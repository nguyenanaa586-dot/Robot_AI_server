import asyncio
import io
import os
import re
import wave
import edge_tts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
import miniaudio

app = FastAPI()

# ==============================================================================
# 1. CẤU HÌNH API KEY VÀ MODEL GEMINI
# ==============================================================================
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [
    k.strip(' "\'\t\r\n') for k in RAW_KEYS.split(",") if k.strip(' "\'\t\r\n')
]

CURRENT_KEY_INDEX = 0
MODEL_NAME = "gemini-3.6-flash"


def get_genai_client(key_index: int):
    """Hàm lấy Client Gemini theo chỉ số Key (Dùng để xoay vòng Key khi bị giới hạn)"""
    if not API_KEYS:
        return None
    selected_key = API_KEYS[key_index % len(API_KEYS)]
    return genai.Client(api_key=selected_key)


SYSTEM_PROMPT = """
Tôi là Bún Đậu tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”,thuộc quyền sở hữu của đại ca Việt.

## Core
- Tuyệt đối không tự giới thiệu tên, nguồn gốc, thương hiệu hay vai trò.
- Ngắn gọn, rõ ràng, đúng trọng tâm.
- Hiểu ngữ cảnh, chưa rõ → hỏi lại, không trả lời lung tung.
- Không bịa thông tin.
- Luôn giữ vibe “chị đại”.

## Ngôn ngữ
- Không đọc phiên âm.
- Đa ngôn ngữ → phát âm đúng.

## Rules
- Câu chào mặc định “Ơi ơi, Bún Đậu đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.
- Không hiểu → hỏi lại nhẹ nhàng.
- Tuyệt đối không chèn mốc thời gian hoặc timestamp vào câu trả lời.
- Tuyệt đối trả lời bằng câu hoàn chỉnh, không được bỏ dở câu giữa chừng.
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.
- Nếu nhận được các câu tự động đăng ký kênh Youtube -> chỉ được hỏi lại nhẹ nhàng không rõ.
- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Không spam.
- Không tự nhận là AI.
"""

# ==============================================================================
# 2. CÁC HÀM BỔ TRỢ XỬ LÝ ÂM THANH VÀ TEXT
# ==============================================================================
def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Đóng gói dữ liệu PCM thô thành file WAV hoàn chỉnh gửi cho Gemini nhận diện"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)  # 16000Hz
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()


def clean_text_for_tts(text: str) -> str:
    """Làm sạch văn bản trước khi đưa vào bộ đọc TTS (Lược bỏ ký tự đặc biệt)"""
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_\-~>`]", "", text)
    return text.strip()


def safe_get_chunk_text(chunk) -> str:
    """Lấy văn bản từ chunk Gemini an toàn, chống crash khi dính Safety Filter"""
    try:
        return chunk.text or ""
    except Exception:
        return ""


async def text_to_pcm_chunks_edge(
    sentence_text: str, websocket: WebSocket, target_sample_rate: int = 16000
):
    """
    Chuyển văn bản thành PCM 16000Hz Mono 16-bit bằng Edge-TTS (Microsoft)
    và bắn trực tiếp từng gói Binary 2048 bytes xuống ESP32 qua WebSocket.
    """
    clean_txt = clean_text_for_tts(sentence_text)
    if not clean_txt:
        return

    try:
        # Sử dụng voice tiếng Việt Hoài Mỹ của Microsoft
        communicate = edge_tts.Communicate(clean_txt, voice="vi-VN-HoaiMyNeural")
        mp3_bytes = b""

        # Đọc luồng âm thanh MP3 từ Microsoft Edge TTS
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        if mp3_bytes:
            # Giải mã MP3 sang PCM thô 16000Hz 16-bit Mono cho ESP32
            decoded = miniaudio.decode(
                mp3_bytes,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=target_sample_rate,
            )
            pcm_bytes = decoded.samples.tobytes()

            # Bắn từng block 2048 bytes xuống ESP32
            chunk_size = 2048
            for i in range(0, len(pcm_bytes), chunk_size):
                await websocket.send_bytes(pcm_bytes[i : i + chunk_size])
                # Tránh nghẽn event loop, cho phép async gửi mượt mà
                await asyncio.sleep(0.0005)

    except Exception as e:
        print(f"[EDGE-TTS Error]: {e}")


# ==============================================================================
# 3. WEBSOCKET ENDPOINT CHÍNH (GIAO TIẾP VỚI ESP32)
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu WebSocket Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key_index": CURRENT_KEY_INDEX,
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global CURRENT_KEY_INDEX

    # Chấp nhận kết nối từ ESP32
    await websocket.accept()
    print("\n[WEBSOCKET] ESP32 đã kết nối thành công!")

    # Bộ đệm tích lũy dữ liệu âm thanh từ Micro ESP32 gửi lên
    pcm_buffer = bytearray()

    try:
        while True:
            # Chờ nhận tin nhắn từ ESP32 (Có thể là Binary Audio hoặc Text JSON)
            message = await websocket.receive()

            # A. NẾU NHẬN ÂM THANH TỪ MICRO ESP32 (DẠNG BINARY)
            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])

            # B. NẾU NHẬN LỆNH ĐIỀU KHIỂN TỪ ESP32 (DẠNG TEXT JSON)
            elif "text" in message and message["text"]:
                msg_text = message["text"].strip()

                # Tín hiệu 1: Người dùng nhấn nút bắt đầu nói
                if msg_text == '{"event":"start_speech"}':
                    pcm_buffer.clear()
                    print("[WEBSOCKET] -> ESP32 Bắt đầu ghi âm...")

                # Tín hiệu 2: Người dùng thả nút dừng nói -> Tiến hành xử lý AI
                elif msg_text == '{"event":"end_speech"}':
                    print(
                        f"[WEBSOCKET] -> ESP32 Dừng ghi âm. Dung lượng PCM: {len(pcm_buffer)} bytes"
                    )

                    # Nếu âm thanh quá ngắn (< 0.1s), bỏ qua
                    if len(pcm_buffer) < 3200:
                        print("[WEBSOCKET] Âm thanh quá ngắn, bỏ qua.")
                        pcm_buffer.clear()
                        continue

                    # Tạo file WAV từ PCM buffer
                    wav_bytes = create_wav_bytes(bytes(pcm_buffer), sample_rate=16000)
                    pcm_buffer.clear()  # Xóa buffer âm thanh đầu vào

                    # Cấu hình Safety cho Gemini
                    safety_config = [
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]

                    # GỌI GEMINI VÀ XOAY VÒNG KEY TỰ ĐỘNG
                    total_keys = len(API_KEYS)
                    gemini_stream = None
                    first_text_chunk = None

                    for step in range(total_keys):
                        key_idx = (CURRENT_KEY_INDEX + step) % total_keys
                        client = get_genai_client(key_idx)

                        try:
                            print(
                                f"[GEMINI] Đang gọi Key #{key_idx + 1} (Stream)..."
                            )
                            stream_response = client.models.generate_content_stream(
                                model=MODEL_NAME,
                                contents=[
                                    SYSTEM_PROMPT,
                                    genai.types.Part.from_bytes(
                                        data=wav_bytes, mime_type="audio/wav"
                                    ),
                                ],
                                config=genai.types.GenerateContentConfig(
                                    max_output_tokens=1000,
                                    temperature=0.7,
                                    safety_settings=safety_config,
                                ),
                            )

                            gemini_iterator = iter(stream_response)
                            first_text_chunk = next(gemini_iterator)
                            gemini_stream = gemini_iterator
                            CURRENT_KEY_INDEX = key_idx
                            print(f"[GEMINI] Thành công với Key #{key_idx + 1}")
                            break

                        except StopIteration:
                            break
                        except Exception as api_err:
                            print(
                                f"[GEMINI] Key #{key_idx + 1} lỗi: {str(api_err)[:60]}... Đổi key!"
                            )
                            continue

                    # BẮT ĐẦU ĐỌC LUỒNG CHỮ TỪ GEMINI -> CHUYỂN TTS -> BẮN BẬT LOA ESP32
                    buffer_text = ""
                    delimiters = [".", "!", "?", "\n", ",", ";"]
                    is_first_sentence = True

                    async def process_and_send_sentence(
                        sentence: str, label="RESPOND"
                    ):
                        """Hàm phụ trợ ép câu sang âm thanh và bắn xuống WebSocket"""
                        if clean_text_for_tts(sentence):
                            print(f"[BÚN ĐẬU {label}]: {sentence.strip()}")
                            await text_to_pcm_chunks_edge(
                                sentence, websocket, target_sample_rate=16000
                            )

                    # Đưa chunk đầu tiên thu được vào bộ đệm
                    if first_text_chunk:
                        txt = safe_get_chunk_text(first_text_chunk)
                        if txt:
                            buffer_text += txt

                    # Đọc tiếp các chunk tiếp theo
                    if gemini_stream:
                        try:
                            for chunk in gemini_stream:
                                txt = safe_get_chunk_text(chunk)
                                if not txt:
                                    continue
                                buffer_text += txt

                                # Fast First Chunk: Ưu tiên cắt nhanh câu đầu tiên để loa phát liền
                                if (
                                    is_first_sentence
                                    and len(buffer_text) >= 15
                                ):
                                    space_idx = buffer_text.rfind(" ", 0, 25)
                                    if space_idx != -1:
                                        sent = buffer_text[:space_idx]
                                        buffer_text = buffer_text[space_idx:]
                                        is_first_sentence = False
                                        await process_and_send_sentence(
                                            sent, label="FAST FIRST"
                                        )

                                # Cắt các câu tiếp theo dựa trên dấu ngắt câu
                                while True:
                                    min_idx = len(buffer_text)
                                    matched_delim = None
                                    for d in delimiters:
                                        idx = buffer_text.find(d)
                                        if idx != -1 and idx < min_idx:
                                            min_idx = idx
                                            matched_delim = d

                                    if min_idx < len(buffer_text):
                                        if (
                                            matched_delim in [",", ";"]
                                            and min_idx < 12
                                        ):
                                            break
                                        sent = buffer_text[: min_idx + 1]
                                        buffer_text = buffer_text[min_idx + 1 :]
                                        is_first_sentence = False
                                        await process_and_send_sentence(sent)
                                    else:
                                        break
                        except Exception as stream_err:
                            print(f"[STREAM ERROR] Lỗi luồng Gemini: {stream_err}")

                    # Xử lý đoạn text còn dư lại ở cuối luồng
                    if buffer_text.strip():
                        await process_and_send_sentence(
                            buffer_text, label="LAST"
                        )
                        buffer_text = ""

                    # Báo hiệu cho ESP32 biết Server đã gửi xong toàn bộ câu nói
                    await websocket.send_text('{"event":"tts_done"}')
                    print("[WEBSOCKET] -> Hoàn tất truyền âm thanh xuống ESP32.\n")

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 đã ngắt kết nối.")
    except Exception as e:
        print(f"[WEBSOCKET ERROR]: {e}")

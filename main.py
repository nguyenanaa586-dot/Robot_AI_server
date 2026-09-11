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
MODEL_NAME = "gemini-3.6-flash"  # Sử dụng model hỗ trợ chuẩn Stable


def get_genai_client(key_index: int):
    """Hàm lấy Client Gemini theo chỉ số Key"""
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
    """Đóng gói dữ liệu PCM thô thành file WAV hoàn chỉnh"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)  # 16000Hz
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()


def clean_text_for_tts(text: str) -> str:
    """Làm sạch văn bản trước khi đưa vào bộ đọc TTS"""
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_\-~>`]", "", text)
    return text.strip()


def safe_get_chunk_text(chunk) -> str:
    """Lấy văn bản từ chunk Gemini an toàn"""
    try:
        return chunk.text or ""
    except Exception:
        return ""


async def text_to_pcm_chunks_edge(
    sentence_text: str, websocket: WebSocket, target_sample_rate: int = 16000
):
    """Chuyển văn bản thành PCM và gửi xuống ESP32 (Đã sửa lỗi No audio was received)"""
    clean_txt = clean_text_for_tts(sentence_text)

    # ĐIỀU CHỈNH 1: Bỏ qua nếu văn bản không chứa bất kỳ chữ cái/chữ số nào để tránh làm Edge-TTS báo lỗi
    if not clean_txt or not re.search(r"\w", clean_txt):
        return

    try:
        communicate = edge_tts.Communicate(clean_txt, voice="vi-VN-HoaiMyNeural")
        mp3_bytes = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        # ĐIỀU CHỈNH 2: Kiểm tra nếu không nhận được dữ liệu MP3 từ Edge-TTS
        if not mp3_bytes:
            print(f"[EDGE-TTS Warning]: Khong nhan duoc audio cho cau: '{clean_txt}'", flush=True)
            return

        # ĐIỀU CHỈNH 3: Chạy giải mã miniaudio trong Thread riêng để không làm nghẽn Event Loop
        def decode_mp3():
            decoded = miniaudio.decode(
                mp3_bytes,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=target_sample_rate,
            )
            return decoded.samples.tobytes()

        pcm_bytes = await asyncio.to_thread(decode_mp3)

        # ĐIỀU CHỈNH 4: Chia gói 1024 bytes vừa đệm ESP32 và nghỉ 1ms giữa các gói giúp âm thanh mượt
        chunk_size = 1024
        for i in range(0, len(pcm_bytes), chunk_size):
            await websocket.send_bytes(pcm_bytes[i : i + chunk_size])
            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"[EDGE-TTS Error]: {e}", flush=True)


# ==============================================================================
# 3. WEBSOCKET ENDPOINT CHÍNH
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

    await websocket.accept()
    print("\n[WEBSOCKET] ESP32 đã kết nối thành công!", flush=True)

    pcm_buffer = bytearray()

    try:
        while True:
            try:
                # Nhận tin nhắn từ ASGI Server
                message = await websocket.receive()
            except RuntimeError:
                print("[WEBSOCKET] ESP32 đã ngắt kết nối (Socket Closed).", flush=True)
                break

            # BẮT BUỘC: Kiểm tra nếu socket báo đóng từ client/proxy
            if message.get("type") == "websocket.disconnect":
                print("[WEBSOCKET] ESP32 đã gửi tín hiệu ngắt kết nối.", flush=True)
                break

            # A. NẾU NHẬN ÂM THANH TỪ MICRO ESP32 (DẠNG BINARY)
            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])

            # B. NẾU NHẬN LỆNH ĐIỀU KHIỂN TỪ ESP32 (DẠNG TEXT JSON)
            elif "text" in message and message["text"]:
                msg_text = message["text"].strip()

                if msg_text == '{"event":"start_speech"}':
                    pcm_buffer.clear()
                    print("[WEBSOCKET] -> ESP32 Bắt đầu ghi âm...", flush=True)

                elif msg_text == '{"event":"end_speech"}':
                    print(
                        f"[WEBSOCKET] -> ESP32 Dừng ghi âm. Dung lượng PCM: {len(pcm_buffer)} bytes",
                        flush=True,
                    )

                    if len(pcm_buffer) < 3200:
                        print("[WEBSOCKET] Âm thanh quá ngắn, bỏ qua.", flush=True)
                        pcm_buffer.clear()
                        continue

                    wav_bytes = create_wav_bytes(bytes(pcm_buffer), sample_rate=16000)
                    pcm_buffer.clear()

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

                    total_keys = len(API_KEYS)
                    gemini_stream = None

                    if total_keys == 0:
                        print("[GEMINI ERROR] Không có GEMINI_API_KEY nào được cài đặt!", flush=True)
                        continue

                    # SỬ DỤNG CLIENT BẤT ĐỒNG BỘ (client.aio) ĐỂ KHÔNG BỊ KHÓA EVENT LOOP
                    for step in range(total_keys):
                        key_idx = (CURRENT_KEY_INDEX + step) % total_keys
                        client = get_genai_client(key_idx)

                        try:
                            print(
                                f"[GEMINI] Đang gọi Key #{key_idx + 1} (Async Stream)...",
                                flush=True,
                            )
                            # Sử dụng client.aio thay vì client.models trực tiếp
                            gemini_stream = await client.aio.models.generate_content_stream(
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
                            CURRENT_KEY_INDEX = key_idx
                            print(f"[GEMINI] Thành công với Key #{key_idx + 1}", flush=True)
                            break

                        except Exception as api_err:
                            print(
                                f"[GEMINI] Key #{key_idx + 1} lỗi: {str(api_err)[:60]}... Đổi key!",
                                flush=True,
                            )
                            continue

                    if not gemini_stream:
                        print("[GEMINI] Tất cả API Keys đều thất bại!", flush=True)
                        continue

                    # ĐỌC LUỒNG CHỮ BẤT ĐỒNG BỘ TỪ GEMINI
                    buffer_text = ""
                    delimiters = [".", "!", "?", "\n", ",", ";"]

                    async def process_and_send_sentence(sentence: str, label="RESPOND"):
                        if clean_text_for_tts(sentence):
                            print(f"[BÚN ĐẬU {label}]: {sentence.strip()}", flush=True)
                            await text_to_pcm_chunks_edge(
                                sentence, websocket, target_sample_rate=16000
                            )

                    try:
                        async for chunk in gemini_stream:
                            txt = safe_get_chunk_text(chunk)
                            if not txt:
                                continue
                            buffer_text += txt

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
                                    await process_and_send_sentence(sent)
                                else:
                                    break
                    except Exception as stream_err:
                        print(f"[STREAM ERROR] Lỗi luồng Gemini: {stream_err}", flush=True)

                    if buffer_text.strip():
                        await process_and_send_sentence(buffer_text, label="LAST")
                        buffer_text = ""

                    await websocket.send_text('{"event":"tts_done"}')
                    print("[WEBSOCKET] -> Hoàn tất truyền âm thanh xuống ESP32.\n", flush=True)

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 đã ngắt kết nối chủ động.", flush=True)
    except Exception as e:
        print(f"[WEBSOCKET ERROR]: {e}", flush=True)

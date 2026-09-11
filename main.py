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

# Tên mô hình chính thức (gemini-2.5-flash hoặc gemini-2.0-flash)
MODEL_NAME = "gemini-3.6-flash" 

# Cấu hình giọng đọc Edge-TTS
TTS_VOICE = "vi-VN-HoaiMyNeural"
TTS_RATE = "+10%"  # Tăng tốc độ đọc lên 10% giúp tự nhiên và phản hồi nhanh hơn


def get_genai_client(key_index: int):
    """Lấy Client Gemini theo chỉ số Key"""
    if not API_KEYS:
        return None
    selected_key = API_KEYS[key_index % len(API_KEYS)]
    return genai.Client(api_key=selected_key)


SYSTEM_PROMPT = """
Tôi là Bún Đậu tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”, thuộc quyền sở hữu của đại ca Việt.

## Core
- Tuyệt đối không tự giới thiệu tên, nguồn gốc, thương hiệu hay vai trò.
- Ngắn gọn, rõ ràng, đúng trọng tâm (tối đa 1-2 câu).
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
- Chỉ nói tiếng Việt, không nói tiếng Anh hệ thống.
- Xưng mày - tao, không xưng bạn - tôi - mình,...
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
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)      # 16-bit PCM
        wav_file.setframerate(sample_rate)  # 16000Hz
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()


def clean_text_for_tts(text: str) -> str:
    """Làm sạch văn bản và loại bỏ ký tự đặc biệt/dấu ngắt thừa để tránh lỗi Edge-TTS"""
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_\-~>`]", "", text)
    return text.strip(" ,;:-_\n\r\t")


def safe_get_chunk_text(chunk) -> str:
    """Lấy văn bản từ chunk Gemini an toàn"""
    try:
        return chunk.text or ""
    except Exception:
        return ""


async def text_to_pcm_chunks_edge(
    sentence_text: str, websocket: WebSocket, target_sample_rate: int = 16000
):
    """Chuyển văn bản thành PCM và stream theo nhịp 25ms tránh tràn RingBuffer ESP32"""
    clean_txt = clean_text_for_tts(sentence_text)

    if not clean_txt or not re.search(r"\w", clean_txt):
        return

    try:
        communicate = edge_tts.Communicate(clean_txt, voice=TTS_VOICE, rate=TTS_RATE)
        mp3_bytes = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes += chunk["data"]

        if not mp3_bytes:
            print(f"[EDGE-TTS Warning]: Không nhận được audio cho câu: '{clean_txt}'", flush=True)
            return

        # Giải mã MP3 -> PCM 16kHz Mono
        def decode_mp3():
            decoded = miniaudio.decode(
                mp3_bytes,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=target_sample_rate,
            )
            return decoded.samples.tobytes()

        pcm_bytes = await asyncio.to_thread(decode_mp3)

        if not pcm_bytes:
            return

        # PACING TỐI ƯU: 1024 bytes = 32ms thời lượng phát loa.
        # Nghỉ 25ms giữa các gói giúp đệm RingBuffer ESP32 không bị tràn.
        chunk_size = 1024
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            await websocket.send_bytes(chunk)
            await asyncio.sleep(0.025)

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
                message = await websocket.receive()
            except RuntimeError:
                print("[WEBSOCKET] ESP32 đã ngắt kết nối (Socket Closed).", flush=True)
                break

            if message.get("type") == "websocket.disconnect":
                print("[WEBSOCKET] ESP32 đã gửi tín hiệu ngắt kết nối.", flush=True)
                break

            # A. NHẬN ÂM THANH BINARY TỪ MICRO ESP32
            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])

            # B. NHẬN LỆNH ĐIỀU KHIỂN TEXT TỪ ESP32
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

                    if len(pcm_buffer) < 3200:  # Nhỏ hơn 0.1 giây (16000 samples/s * 2 bytes * 0.1s)
                        print("[WEBSOCKET] Âm thanh quá ngắn, bỏ qua.", flush=True)
                        pcm_buffer.clear()
                        await websocket.send_text('{"event":"tts_done"}')
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
                        print("[GEMINI ERROR] Không tìm thấy GEMINI_API_KEY nào!", flush=True)
                        await websocket.send_text('{"event":"error", "message":"No API Keys"}')
                        continue

                    # BẮT ĐẦU GỌI GEMINI STREAM VỚI XOAY VÒNG KEY
                    for step in range(total_keys):
                        key_idx = (CURRENT_KEY_INDEX + step) % total_keys
                        client = get_genai_client(key_idx)

                        try:
                            print(
                                f"[GEMINI] Đang gọi Key #{key_idx + 1} (Async Stream)...",
                                flush=True,
                            )
                            gemini_stream = await client.aio.models.generate_content_stream(
                                model=MODEL_NAME,
                                contents=[
                                    genai.types.Part.from_bytes(
                                        data=wav_bytes, mime_type="audio/wav"
                                    ),
                                ],
                                config=genai.types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    max_output_tokens=300,
                                    temperature=0.7,
                                    safety_settings=safety_config,
                                ),
                            )
                            CURRENT_KEY_INDEX = key_idx
                            print(f"[GEMINI] Thành công kết nối với Key #{key_idx + 1}", flush=True)
                            break

                        except Exception as api_err:
                            print(
                                f"[GEMINI] Key #{key_idx + 1} gặp lỗi: {str(api_err)[:80]}... Chuyển sang Key tiếp theo!",
                                flush=True,
                            )
                            continue

                    if not gemini_stream:
                        print("[GEMINI] Tất cả API Keys đều thất bại!", flush=True)
                        await websocket.send_text('{"event":"error", "message":"503 Service Unavailable"}')
                        continue

                    # ==========================================================
                    # LUỒNG XỬ LÝ: NHẬN ĐỦ VĂN BẢN TỪ GEMINI MỚI PHÁT AUDIO
                    # ==========================================================
                    full_response_text = ""

                    try:
                        async for chunk in gemini_stream:
                            txt = safe_get_chunk_text(chunk)
                            if txt:
                                full_response_text += txt

                        cleaned_text = clean_text_for_tts(full_response_text)

                        if cleaned_text:
                            print(f"[BÚN ĐẬU RESPOND]: {cleaned_text}", flush=True)

                            # Tách câu dựa trên dấu ngắt câu (.?!;\n)
                            sentences = re.split(r"(?<=[.?!;\n])\s+", cleaned_text)

                            # Chuyển từng câu thành audio và gửi nối tiếp xuống ESP32
                            for sent in sentences:
                                clean_sent = clean_text_for_tts(sent)
                                if clean_sent:
                                    await text_to_pcm_chunks_edge(
                                        clean_sent, websocket, target_sample_rate=16000
                                    )

                    except Exception as stream_err:
                        print(f"[STREAM ERROR] Lỗi trong quá trình xử lý luồng Gemini: {stream_err}", flush=True)
                        await websocket.send_text('{"event":"error", "message":"Stream failed"}')

                    # Gửi tín hiệu báo hoàn tất lượt nói xuống ESP32
                    await websocket.send_text('{"event":"tts_done"}')
                    print("[WEBSOCKET] -> Hoàn tất gửi dữ liệu âm thanh tới ESP32.\n", flush=True)

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 đã ngắt kết nối chủ động.", flush=True)
    except Exception as e:
        print(f"[WEBSOCKET ERROR]: {e}", flush=True)
    finally:
        pcm_buffer.clear()

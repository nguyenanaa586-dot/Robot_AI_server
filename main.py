import io
import os
import re
import wave
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from gtts import gTTS
import miniaudio

app = FastAPI()

# 1. TỰ ĐỘNG TÁCH VÀ LÀM SẠCH NHIỀU API KEY
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [
    k.strip(' "\'\t\r\n') for k in RAW_KEYS.split(",") if k.strip(' "\'\t\r\n')
]

CURRENT_KEY_INDEX = 0
MODEL_NAME = "gemini-3.6-flash"


def get_genai_client(key_index: int):
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


def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)  # Sample rate
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_\-~>`]", "", text)
    return text.strip()


def safe_get_chunk_text(chunk) -> str:
    """Lấy text an toàn tránh crash server khi Gemini trả về chunk rỗng hoặc bị Safety Filter chặn"""
    try:
        return chunk.text or ""
    except Exception:
        return ""


def text_to_pcm_chunks(sentence_text: str, target_sample_rate: int = 16000):
    """Chuyển từng câu văn ngắn thành PCM raw thô chuẩn 16000Hz (16-bit Mono)

    để ESP32 giải mã mịn, không rè tiếng và ném thẳng xuống Socket HTTPS.
    """
    clean_txt = clean_text_for_tts(sentence_text)
    if not clean_txt:
        return

    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=clean_txt, lang="vi")
        tts.write_to_fp(mp3_fp)
        mp3_bytes = mp3_fp.getvalue()

        # Decode MP3 gốc sang chuẩn PCM 16-bit Mono 16000Hz
        decoded = miniaudio.decode(
            mp3_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=target_sample_rate,
        )
        pcm_bytes = decoded.samples.tobytes()

        # Tăng kích thước chunk lên 2048 bytes giúp ESP32 nhận mượt hơn, chống ngắt DMA
        chunk_size = 2048
        for i in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[i : i + chunk_size]

    except Exception as e:
        print(f"[TTS Chunk Error]: {e}")


@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu Streaming Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key_index": CURRENT_KEY_INDEX,
    }


@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    global CURRENT_KEY_INDEX

    try:
        # 1. NHẬN LUỒNG STREAM AUDIO TỪ ESP32
        pcm_chunks = []
        async for chunk in request.stream():
            pcm_chunks.append(chunk)

        pcm_bytes = b"".join(pcm_chunks)
        print(f"[STREAM RECEIVE] Tong dung luong nhan duoc: {len(pcm_bytes)} bytes")

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return StreamingResponse(io.BytesIO(b""), status_code=400)

        if not API_KEYS:
            return StreamingResponse(io.BytesIO(b""), status_code=500)

        wav_bytes = create_wav_bytes(pcm_bytes, sample_rate=16000)

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

        # 2. GỌI GEMINI STREAMING VÀ XOAY KEY TỰ ĐỘNG
        total_keys = len(API_KEYS)
        gemini_stream = None
        first_text_chunk = None

        for step in range(total_keys):
            key_idx = (CURRENT_KEY_INDEX + step) % total_keys
            client = get_genai_client(key_idx)

            try:
                print(f"[FAST CALL] Dang goi Key #{key_idx + 1} (Chế độ Stream)...")
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

                # Đọc thử chunk đầu tiên để xác nhận API Key hoạt động tốt
                gemini_iterator = iter(stream_response)
                first_text_chunk = next(gemini_iterator)

                gemini_stream = gemini_iterator
                CURRENT_KEY_INDEX = key_idx
                print(f"[SUCCESS] Ket noi thanh cong Key #{key_idx + 1}")
                break

            except StopIteration:
                break
            except Exception as api_err:
                err_str = str(api_err)
                print(
                    f"[KEY FAILURE] Key #{key_idx + 1} bi loi:"
                    f" {err_str[:80]}... Doi sang Key tiep theo!"
                )
                continue

        # 3. ĐỆM TEXT (BUFFER) -> CẮT CÂU NGAY KHI CÓ DẤU NGẮT -> TTS -> PHÁT ÂM THANH XUỐNG ESP32
        def audio_stream_generator():
            buffer_text = ""
            # Danh sách các dấu ngắt câu
            delimiters = [".", "!", "?", "\n", ",", ";"]
            is_first_sentence = True  # Cờ đánh dấu để ưu tiên cắt nhanh đoạn text đầu tiên

            def process_buffer(force_flush=False):
                nonlocal buffer_text, is_first_sentence
                while True:
                    # --- BẮT ĐẦU PHẦN TỐI ƯU FAST FIRST CHUNK ---
                    # Nếu là câu đầu tiên, bộ đệm >= 15 ký tự và không phải ép xả cuối cùng
                    if is_first_sentence and len(buffer_text) >= 15 and not force_flush:
                        # Tìm khoảng trắng gần nhất để không cắt ngang một từ đang viết dở
                        space_idx = buffer_text.rfind(' ', 0, 25)
                        if space_idx != -1:
                            # Cắt lấy đoạn đầu tiên này
                            sentence = buffer_text[:space_idx]
                            # Giữ lại phần còn lại trong buffer
                            buffer_text = buffer_text[space_idx:]
                            is_first_sentence = False  # Đã xong việc ưu tiên câu đầu
                            
                            if clean_text_for_tts(sentence):
                                print(f"[BÚN ĐẬU RESPOND (FAST FIRST)]: {sentence.strip()}")
                                yield from text_to_pcm_chunks(sentence, target_sample_rate=16000)
                            continue  # Chạy lại vòng lặp while để kiểm tra tiếp buffer
                    # --- KẾT THÚC PHẦN TỐI ƯU FAST FIRST CHUNK ---

                    min_idx = len(buffer_text)
                    matched_delim = None

                    for d in delimiters:
                        idx = buffer_text.find(d)
                        if idx != -1 and idx < min_idx:
                            min_idx = idx
                            matched_delim = d

                    # Phát hiện dấu ngắt câu trong Buffer (Xử lý các câu tiếp theo)
                    if min_idx < len(buffer_text):
                        # Nếu gặp dấu phẩy/chấm phẩy, chỉ ngắt khi vế đủ dài (> 12 ký tự)
                        if matched_delim in [",", ";"] and min_idx < 12 and not force_flush:
                            break

                        sentence = buffer_text[: min_idx + 1]
                        buffer_text = buffer_text[min_idx + 1 :]
                        is_first_sentence = False # Hủy cờ ưu tiên nếu gặp dấu câu sớm

                        if clean_text_for_tts(sentence):
                            # IN CÂU TRẢ LỜI CỦA BÚN ĐẬU RA CONSOLE SERVER
                            print(f"[BÚN ĐẬU RESPOND]: {sentence.strip()}")
                            yield from text_to_pcm_chunks(sentence, target_sample_rate=16000)
                    else:
                        break

            # Xử lý phần text còn dư khi LLM đã hoàn thành luồng
            if force_flush and clean_text_for_tts(buffer_text):
                # IN CÂU CUỐI CỦA BÚN ĐẬU RA CONSOLE SERVER
                print(f"[BÚN ĐẬU RESPOND (LAST)]: {buffer_text.strip()}")
                yield from text_to_pcm_chunks(buffer_text, target_sample_rate=16000)
                buffer_text = ""

            # Trường hợp tất cả Key đều hết dung lượng
            if gemini_stream is None and first_text_chunk is None:
                error_msg = (
                    "Hết lượt dùng miễn phí rồi mày, xì tiền ra mua gói vip pro giùm"
                    " tao đi."
                )
                print(f"[BÚN ĐẬU RESPOND ERROR]: {error_msg}")
                yield from text_to_pcm_chunks(error_msg, target_sample_rate=16000)
                return

            # Đưa chunk đầu tiên vào Buffer
            if first_text_chunk:
                text = safe_get_chunk_text(first_text_chunk)
                if text:
                    buffer_text += text
                    yield from process_buffer(force_flush=False)

            # Nhận liên tục các chunk tiếp theo từ luồng (Đã bọc try-except chống nổ server khi ngắt mạng)
            if gemini_stream:
                try:
                    for chunk in gemini_stream:
                        text = safe_get_chunk_text(chunk)
                        if text:
                            buffer_text += text
                            yield from process_buffer(force_flush=False)
                except Exception as stream_err:
                    print(f"[STREAM INTERRUPTED] Luồng kết nối bị ngắt: {stream_err}")

            # Đẩy phần chữ còn lại trong buffer đi TTS
            yield from process_buffer(force_flush=True)

        # Trả về luồng Streaming Response
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "Transfer-Encoding": "chunked",
            },
        )

    except Exception as err:
        print(f"[SERVER ERROR]: {str(err)}")
        return StreamingResponse(io.BytesIO(b""), status_code=500)

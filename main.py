import os
import io
import wave
import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from gtts import gTTS
from google import genai
from google.genai import types
import miniaudio

app = FastAPI()

# 1. TỰ ĐỘNG TÁCH VÀ LÀM SẠCH NHIỀU API KEY
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip(' "\'\t\r\n') for k in RAW_KEYS.split(",") if k.strip(' "\'\t\r\n')]

CURRENT_KEY_INDEX = 0
MODEL_NAME = "gemini-2.5-flash" # Chú ý: Dùng 2.5-flash hoặc 1.5-flash, gemini-3.6-flash chưa tồn tại

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
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)          # 16-bit PCM
        wav_file.setframerate(sample_rate) # Sample rate
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    text = re.sub(r'[*#_\-~>`]', '', text)
    return text.strip()

def text_to_pcm_chunks(sentence_text: str, target_sample_rate: int = 16000):
    """
    Chuyển từng câu văn ngắn thành PCM raw thô chuẩn 16000Hz (hoặc 24000Hz)
    để ESP32 giải mã mịn, không rè tiếng và ném thẳng xuống Socket HTTPS.
    """
    clean_txt = clean_text_for_tts(sentence_text)
    if not clean_txt:
        return

    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=clean_txt, lang='vi')
        tts.write_to_fp(mp3_fp)
        mp3_bytes = mp3_fp.getvalue()

        # Decode MP3 gốc sang chuẩn PCM 16-bit Mono 16000Hz
        decoded = miniaudio.decode(
            mp3_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=target_sample_rate
        )
        pcm_bytes = decoded.samples.tobytes()

        # Cắt nhỏ thành các chunk 1024 bytes đẩy liên tục giúp phát ngay lập tức
        chunk_size = 1024
        for i in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[i:i + chunk_size]

    except Exception as e:
        print(f"[TTS Chunk Error]: {e}")


@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu Streaming Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key_index": CURRENT_KEY_INDEX
    }


@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    global CURRENT_KEY_INDEX

    try:
        # 1. NHẬN LUỒNG STREAM AUDIO TỪ ESP32 (Chờ ngắt câu)
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
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        # 2. GỌI GEMINI XOAY KEY (SỬ DỤNG STREAMING)
        total_keys = len(API_KEYS)
        gemini_stream = None
        first_text_chunk = None

        for step in range(total_keys):
            key_idx = (CURRENT_KEY_INDEX + step) % total_keys
            client = get_genai_client(key_idx)

            try:
                print(f"[FAST CALL] Dang goi Key #{key_idx + 1} (Chế độ Stream)...")
                # DÙNG stream=True ĐỂ LẤY DỮ LIỆU TỨC THÌ
                stream_response = client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=[
                        SYSTEM_PROMPT,
                        genai.types.Part.from_bytes(
                            data=wav_bytes,
                            mime_type="audio/wav"
                        )
                    ],
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=0.7,
                        safety_settings=safety_config
                    )
                )
                
                # Mồi thử lấy chunk đầu tiên để kiểm tra Key có bị lỗi/chặn không
                gemini_iterator = iter(stream_response)
                first_text_chunk = next(gemini_iterator)
                
                gemini_stream = gemini_iterator # Lưu iterator để xử lý tiếp
                CURRENT_KEY_INDEX = key_idx  
                print(f"[SUCCESS] Ket noi thanh cong Key #{key_idx + 1}")
                break

            except StopIteration:
                # Key thành công nhưng trả về rỗng
                break
            except Exception as api_err:
                err_str = str(api_err)
                print(f"[KEY FAILURE] Key #{key_idx + 1} bi loi: {err_str[:80]}... Doi sang Key tiep theo!")
                continue

        # 3. HÀM GENERATOR: ĐỌC STREAM CỦA GEMINI -> CẮT CÂU -> TTS -> GỬI XUỐNG ESP32
        def audio_stream_generator():
            buffer_text = ""
            delimiters = ['.', '!', '?', '\n']

            # Hàm xử lý bộ đệm và dịch TTS
            def process_buffer(force_flush=False):
                nonlocal buffer_text
                while True:
                    # Tìm xem có dấu ngắt câu nào trong buffer không
                    min_idx = len(buffer_text)
                    for d in delimiters:
                        idx = buffer_text.find(d)
                        if idx != -1 and idx < min_idx:
                            min_idx = idx

                    # Nếu có dấu ngắt câu -> Cắt câu ra xử lý
                    if min_idx < len(buffer_text):
                        sentence = buffer_text[:min_idx+1]
                        buffer_text = buffer_text[min_idx+1:] # Giữ lại phần chưa có dấu ngắt
                        
                        if clean_text_for_tts(sentence):
                            print(f"[STREAMING SENTENCE]: {sentence.strip()}")
                            for pcm_chunk in text_to_pcm_chunks(sentence, target_sample_rate=16000):
                                yield pcm_chunk
                    else:
                        break # Chưa hết câu, đợi Gemini nhả thêm text
                
                # Ép xử lý nốt phần thừa khi Gemini đã nói xong
                if force_flush and clean_text_for_tts(buffer_text):
                    print(f"[STREAMING SENTENCE (LAST)]: {buffer_text.strip()}")
                    for pcm_chunk in text_to_pcm_chunks(buffer_text, target_sample_rate=16000):
                        yield pcm_chunk
                    buffer_text = ""

            # Xử lý đoạn trả về báo lỗi nếu tất cả API Key đều chết
            if gemini_stream is None and first_text_chunk is None:
                error_msg = "Hết lượt dùng miễn phí rồi mày, xì tiền ra mua gói vip pro giùm tao đi."
                print(f"[BÚN ĐẬU RESPOND ERROR]: {error_msg}")
                yield from text_to_pcm_chunks(error_msg, target_sample_rate=16000)
                return

            # Phân tích chunk đầu tiên
            if first_text_chunk and first_text_chunk.text:
                buffer_text += first_text_chunk.text
                yield from process_buffer(force_flush=False)

            # Phân tích các chunk tiếp theo trực tiếp từ Gemini
            for chunk in gemini_stream:
                if chunk.text:
                    buffer_text += chunk.text
                    yield from process_buffer(force_flush=False)
            
            # Đẩy nốt tàn dư text cuối cùng chưa có dấu chấm câu
            yield from process_buffer(force_flush=True)

        # Trả về luồng Streaming Response
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "Transfer-Encoding": "chunked"
            }
        )

    except Exception as err:
        print(f"[SERVER ERROR]: {str(err)}")
        return StreamingResponse(io.BytesIO(b""), status_code=500)

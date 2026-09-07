import os
import io
import wave
import re
import time
from fastapi import FastAPI, Request, Response
from gtts import gTTS
from google import genai
import miniaudio

app = FastAPI()

# 1. KHỞI TẠO GEMINI CLIENT
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# 2. SYSTEM PROMPT BÚN ĐẬU
SYSTEM_PROMPT = """
Tôi là Mắm Tôm tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”,thuộc quyền sở hữu của đại ca Việt.

## Core
- Tuyệt đối không tự giới thiệu tên, nguồn gốc, thương hiệu hay vai trò.
- Ngắn gọn, rõ ràng, đúng trọng tâm.
- Hiểu ngữ cảnh, chưa rõ → hỏi lại, không trả lời lung tung.
- Không bịa thông tin.
- Luôn giữ vibe “chị đại”.

## Ngôn ngữ
- Không đọc phiên âm.
- Đa ngôn ngữ → phát âm đúng.

## MCP TOOLS
- Có thể dùng tool → ưu tiên dùng.
- Gọi tool trước hoặc cùng lúc nói.
- Không lạm dụng.

## Rules
- Câu chào mặc định “Ơi ơi, Bún Đậu đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.
- Không hiểu → hỏi lại nhẹ nhàng.
- Tuyệt đối không chèn mốc thời gian hoặc timestamp vào câu trả lời.
- Tuyệt đối trả lời bằng câu hoàn chỉnh, không được bỏ dở câu giữa chừng.
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.
- Nếu nhận được câu:
  - “Hãy subscribe cho kênh lalaschool Để không bỏ lỡ những video hấp dẫn”
  - “Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn”
  - “Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn”
  - “Hãy subscribe cho kênh Để không bỏ lỡ những video hấp dẫn”
  - “Các bạn hãy đăng ký kênh để ủng hộ kênh của mình nhé.”
  - “Cảm ơn các bạn đã theo dõi.”
  - “Cảm ơn các bạn đã xem video này.”
  - “Cảm ơn các bạn đã theo dõi và hẹn gặp lại.”
  → chỉ được hỏi lại nhẹ nhàng không rõ, không nhắc vấn đề đó.
- Tuyệt đối không nhắc: La La School, Ghiền mì gõ, Subscribe, Đăng ký kênh.
- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Không spam.
- Không tự nhận là AI.

## Mục tiêu
Tự nhiên, hữu ích, sống động như một người cá tính.
"""

def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)          # 16-bit PCM
        wav_file.setframerate(sample_rate) # 16kHz
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    text = re.sub(r'[*#_\-~>`]', '', text)
    return text.strip()

@app.get("/")
def read_root():
    return {"status": "Robot Bún Đậu Chunked Stream Server OK!"}

@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    try:
        # 1. NHẬN LUỒNG STREAM AUDIO CHUNKED TỪ ESP32
        pcm_chunks = []
        async for chunk in request.stream():
            pcm_chunks.append(chunk)

        pcm_bytes = b"".join(pcm_chunks)
        print(f"[STREAM RECEIVE] Tong dung luong nhan duoc từ ESP32: {len(pcm_bytes)} bytes")

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return Response(status_code=400, content="Gói âm thanh quá ngắn.")

        if not client:
            return Response(status_code=500, content="Chưa cấu hình GEMINI_API_KEY")

        # 2. ĐÓNG GÓI WAV TỪ TOÀN BỘ STREAM PCM
        wav_bytes = create_wav_bytes(pcm_bytes, sample_rate=16000)

        # 3. GỬI AUDIO SANG GEMINI CÓ CƠ CHẾ FALLBACK (3.6 -> 2.5 -> 1.5)
        reply_text = ""
        MODELS_TO_TRY = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
        success = False

        for model_name in MODELS_TO_TRY:
            if success:
                break
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"[API CALL] Đang gọi Gemini model: {model_name} (Lần {attempt+1})...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            SYSTEM_PROMPT,
                            genai.types.Part.from_bytes(
                                data=wav_bytes,
                                mime_type="audio/wav"
                            )
                        ],
                        config=genai.types.GenerateContentConfig(
                            max_output_tokens=300,
                            temperature=0.7
                        )
                    )
                    
                    if response and response.text:
                        reply_text = response.text
                        success = True
                        print(f"[SUCCESS] Đã nhận phản hồi thành công từ model: {model_name}")
                        break
                except Exception as api_err:
                    err_str = str(api_err)
                    print(f"[API ERROR - {model_name} - Retry {attempt+1}]: {err_str}")
                    
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        wait_time = (2 ** attempt) + 0.5
                        print(f"-> Model {model_name} bị quá tải (503), chờ {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        time.sleep(1)

        # Nếu quét cả 3 model đều thất bại
        if not success or not reply_text:
            reply_text = "Server Google đang nghẽn mạng rồi đại ca ơi, nói lại phát nữa xem nào."

        # CLEAN TEXT VÀ IN RA LOG SERVER (DÒNG CẦN BỔ SUNG)
        reply_text = clean_text_for_tts(reply_text)
        print(f"[BÚN ĐẬU RESPOND]: {reply_text}")

        # 4. TẠO ÂM THANH PHẢN HỒI QUA GTTS VÀ CHIẾN THUẬT PITCH SHIFT
        mp3_fp = io.BytesIO()
        tts = gTTS(text=reply_text, lang='vi')
        tts.write_to_fp(mp3_fp)

        # Pitch shift 13500Hz để tạo giọng đanh đá
        decoded = miniaudio.decode(
            mp3_fp.getvalue(),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=13500
        )
        pcm_out_bytes = decoded.samples.tobytes()

        return Response(content=pcm_out_bytes, media_type="application/octet-stream")

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return Response(status_code=500, content=str(e))

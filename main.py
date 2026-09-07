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

# 1. KHỞI TẠO AI CLIENT
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("[WARNING] Chưa cấu hình GEMINI_API_KEY trên môi trường Render!")

# 2. SYSTEM PROMPT BÚN ĐẬU
SYSTEM_PROMPT = """
Tôi là Bún Đậu tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”. thuộc quyền sở hữu của đại ca Việt.

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
- Tuyệt đối không chèn mốc thời gian hoặc timestamp (dạng 00:08, 00:11) vào câu trả lời.
- Tuyệt đối trả lời bằng câu hoàn chỉnh, không được bỏ dở câu giữa chừng.
- Không hiểu → hỏi lại nhẹ nhàng.
- Tuyệt đối không nhắc: La La School, Ghiền mì gõ, Subscribe, Đăng ký kênh.
- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Không spam.
- Không tự nhận là AI.
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.
"""

# 3. CÁC HÀM BỔ TRỢ (HELPER FUNCTIONS)
def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Đóng gói dữ liệu PCM thô nhận từ ESP32 thành file WAV có header chuẩn"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)          # 16-bit PCM
        wav_file.setframerate(sample_rate) # 16kHz
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

def clean_text_for_tts(text: str) -> str:
    """Lọc sạch ký tự đặc biệt, Markdown và timestamp trước khi tạo giọng nói"""
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    text = re.sub(r'[*#_\-~>`]', '', text)
    return text.strip()

# 4. ROUTE CHECK SỨC KHỎE SERVER
@app.get("/")
def read_root():
    return {"status": "Robot Bún Đậu Server đang hoạt động ổn định!"}

# 5. ENDPOINT XỬ LÝ ÂM THANH CHÍNH
@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    try:
        # 5.1. Nhận luồng byte PCM thô trực tiếp từ body (Không tốn PSRAM ở ESP32)
        pcm_bytes = await request.body()
        print(f"[SERVER] Đã nhận {len(pcm_bytes)} bytes audio PCM từ ESP32-S3.")

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return Response(status_code=400, content="Dữ liệu âm thanh quá ngắn hoặc rỗng.")

        if not client:
            return Response(status_code=500, content="Server chưa được cấu hình GEMINI_API_KEY.")

        # 5.2. Chuyển PCM thô thành WAV binary để Gemini xử lý trực tiếp (Speech-to-Text & LLM)
        wav_bytes = create_wav_bytes(pcm_bytes, sample_rate=16000)

        # 5.3. Gửi Audio trực tiếp lên Gemini 3.6 Flash với cơ chế Retry
        reply_text = ""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
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
                reply_text = response.text if (response and response.text) else "Tao nghe chưa rõ, mày nói lại xem nào."
                break

            except Exception as api_err:
                err_str = str(api_err)
                print(f"[API ERROR - Lần {attempt + 1}/{max_retries}]: {err_str}")

                if "503" in err_str or "UNAVAILABLE" in err_str:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        reply_text = "Server AI đang quá tải, tao chưa nghe kịp. Mày nói lại sau vài giây xem."
                elif "429" in err_str:
                    reply_text = "Mua gói vip pro giùm tao cái, không có tiền mua thì tao đi ngủ, cần hỏi gì thì mai quay lại tìm tao."
                    break
                else:
                    reply_text = "Có lỗi kết nối rồi, mày nói lại lần nữa xem."
                    break

        # 5.4. Làm sạch văn bản trước khi đưa sang TTS
        reply_text = clean_text_for_tts(reply_text)
        print(f"[BÚN ĐẬU RESPOND]: {reply_text}")

        # 5.5. Chuyển văn bản thành Audio MP3 qua gTTS
        mp3_fp = io.BytesIO()
        tts = gTTS(text=reply_text, lang='vi')
        tts.write_to_fp(mp3_fp)
        mp3_data = mp3_fp.getvalue()

        # 5.6. Hạ tông giọng (Pitch Shift) & Chuyển MP3 thành PCM 16-bit Mono gửi về ESP32
        # Mức 13200Hz giúp giọng gTTS đọc nhanh và đanh đá hơn
        PITCH_SHIFT_RATE = 13200 

        decoded = miniaudio.decode(
            mp3_data,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=PITCH_SHIFT_RATE
        )
        pcm_out_bytes = decoded.samples.tobytes()

        print(f"[SERVER] Đã giải mã {len(pcm_out_bytes)} bytes PCM. Đang phản hồi về ESP32-S3...")
        
        # 5.7. Trả về stream binary PCM trực tiếp
        return Response(content=pcm_out_bytes, media_type="application/octet-stream")

    except Exception as e:
        print(f"[ERROR EXCEPTION]: {str(e)}")
        return Response(status_code=500, content=str(e))

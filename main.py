import os
import io
import wave
from fastapi import FastAPI, Request, Response
import edge_tts
from google import genai
import miniaudio

app = FastAPI()

# Khởi tạo Gemini Client từ API Key môi trường
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("[WARNING] Chưa cấu hình GEMINI_API_KEY trên Render!")

VOICE_VIETNAMESE = "vi-VN-HoaiMyNeural"

def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Đóng gói PCM thô thành chuẩn WAV có header"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)          # 16-bit
        wav_file.setframerate(sample_rate) # 16kHz
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

@app.get("/")
def read_root():
    return {"status": "Robot AI Server đang chạy ổn định!"}

@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    try:
        # 1. Nhận luồng byte PCM thô từ ESP32-S3
        pcm_bytes = await request.body()
        print(f"[SERVER] Đã nhận {len(pcm_bytes)} bytes audio từ ESP32-S3.")

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return Response(status_code=400, content="Dữ liệu âm thanh quá ngắn.")

        if not client:
            return Response(status_code=500, content="Server chưa cấu hình GEMINI_API_KEY.")

        # 2. Tạo WAV Header cho khối âm thanh PCM
        wav_bytes = create_wav_bytes(pcm_bytes, sample_rate=16000)

        # 3. Gửi Audio tới Gemini 2.0 Flash
        prompt = (
            "Bạn là một Robot AI thông minh, thân thiện. "
            "Hãy lắng nghe âm thanh này và trả lời bằng văn bản tiếng Việt "
            "ngắn gọn, súc tích (tối đa 2-3 câu)."
        )

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                prompt,
                genai.types.Part.from_bytes(
                    data=wav_bytes,
                    mime_type="audio/wav"
                )
            ]
        )

        reply_text = response.text if (response and response.text) else "Tôi đã nghe nhưng chưa hiểu rõ, bạn nói lại nhé."
        print(f"[GEMINI RESPOND]: {reply_text}")

        # 4. Chuyển văn bản thành dữ liệu MP3 (Edge TTS)
        communicate = edge_tts.Communicate(reply_text, VOICE_VIETNAMESE)
        mp3_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data.extend(chunk["data"])

        # 5. Giải mã MP3 -> PCM 16kHz 16-bit Mono cho ESP32-S3
        decoded = miniaudio.decode(
            bytes(mp3_data),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16000
        )
        pcm_out_bytes = decoded.samples.tobytes()

        print(f"[SERVER] Đã xuất {len(pcm_out_bytes)} bytes PCM. Đang gửi về ESP32-S3...")
        return Response(content=pcm_out_bytes, media_type="application/octet-stream")

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return Response(status_code=500, content=str(e))

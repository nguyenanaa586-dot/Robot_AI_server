import os
import io
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

@app.get("/")
def read_root():
    return {"status": "Robot AI Server đang chạy ổn định!"}

@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    try:
        # Nhận trực tiếp luồng byte âm thanh thô từ ESP32-S3 (Khắc phục triệt để lỗi 422)
        audio_bytes = await request.body()
        print(f"[SERVER] Đã nhận {len(audio_bytes)} bytes audio từ ESP32-S3.")

        if not audio_bytes:
            return Response(status_code=400, content="Không nhận được dữ liệu âm thanh.")

        if not client:
            return Response(status_code=500, content="Server chưa cấu hình GEMINI_API_KEY.")

        # Gửi Audio trực tiếp cho Gemini 2.5 Flash
        prompt = (
            "Bạn là một Robot AI thông minh, thân thiện. "
            "Hãy lắng nghe âm thanh này và trả lời bằng văn bản tiếng Việt "
            "ngắn gọn, súc tích (tối đa 2-3 câu)."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                prompt,
                genai.types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                )
            ]
        )

        reply_text = response.text
        print(f"[GEMINI RESPOND]: {reply_text}")

        # Chuyển văn bản thành dữ liệu MP3 (Edge TTS)
        communicate = edge_tts.Communicate(reply_text, VOICE_VIETNAMESE)
        mp3_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data.extend(chunk["data"])

        # Convert MP3 -> Raw PCM 16kHz 16-bit Mono (Cho loa ESP32 phát)
        decoded = miniaudio.decode(
            bytes(mp3_data),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16000
        )
        pcm_bytes = decoded.samples.tobytes()

        print(f"[SERVER] Đã xuất {len(pcm_bytes)} bytes PCM. Đang gửi về ESP32-S3...")
        return Response(content=pcm_bytes, media_type="application/octet-stream")

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return Response(status_code=500, content=str(e))

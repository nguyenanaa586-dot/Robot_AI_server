import os
import io
import asyncio
from fastapi import FastAPI, UploadFile, File, Response
import edge_tts
import google.generativeai as genai
from pydub import AudioSegment

app = FastAPI()

# 1. Lấy API Key từ biến môi trường của Render (Bảo mật)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("[WARNING] Chưa cấu hình GEMINI_API_KEY trên Render!")

# Giọng đọc Tiếng Việt (Edge TTS - Nữ Hoài My)
VOICE_VIETNAMESE = "vi-VN-HoaiMyNeural"

@app.get("/")
def read_root():
    return {"status": "Robot AI Server đang chạy ổn định!"}

@app.post("/api/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    try:
        # A. Đọc file âm thanh PCM/WAV từ ESP32-S3
        audio_bytes = await file.read()
        print(f"[SERVER] Đã nhận {len(audio_bytes)} bytes audio từ ESP32-S3.")

        if not GEMINI_API_KEY:
            return Response(status_code=500, content="Server chưa cấu hình GEMINI_API_KEY.")

        # B. Gửi Audio trực tiếp cho Gemini AI để hiểu giọng nói và tạo câu trả lời
        prompt = (
            "Bạn là một Robot AI thông minh, thân thiện. "
            "Hãy lắng nghe âm thanh này và trả lời bằng văn bản tiếng Việt "
            "ngắn gọn, xúc tích (tối đa 2-3 câu)."
        )
        
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        
        reply_text = response.text
        print(f"[GEMINI RESPOND]: {reply_text}")

        # C. Chuyển văn bản thành giọng nói (TTS) dạng MP3
        communicate = edge_tts.Communicate(reply_text, VOICE_VIETNAMESE)
        mp3_fp = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_fp.write(chunk["data"])
        
        mp3_fp.seek(0)
        
        # D. Convert MP3 -> Raw PCM 16kHz 16-bit Mono (Chuẩn I2S cho ESP32-S3 phát loa)
        audio_segment = AudioSegment.from_file(mp3_fp, format="mp3")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        pcm_bytes = audio_segment.raw_data
        print(f"[SERVER] Đã xuất {len(pcm_bytes)} bytes PCM. Đang gửi về ESP32-S3...")

        return Response(content=pcm_bytes, media_type="application/octet-stream")

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return Response(status_code=500, content=str(e))

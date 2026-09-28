ROBOT BÚN ĐẬU SERVER V4.10

Mục tiêu bản này: khôi phục đường fallback nhận dạng đã hoạt động ổn định ở V4.8.

Luồng chính:
- Gemini 3.8 Live: não chính + Google Search.
- Khi 3.8 Live hết quota/lỗi: Gemini 3.6 Flash nhận TRỰC TIẾP WAV 16 kHz từ ESP32.
- Không chạy faster-whisper/local STT trong đường fallback.
- Giữ quota guard, MEMORY/ACTION/REPLY, ToF và TTS/Edge-TTS fallback của V4.9.
- Giờ/ngày/thứ vẫn có ngữ cảnh SERVER_TIME_NOW trong system prompt.

Render:
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
Environment Variables hiện tại vẫn đủ với GEMINI_API_KEY và các biến TTS bạn đang dùng.
Không cần cài faster-whisper cho V4.10.

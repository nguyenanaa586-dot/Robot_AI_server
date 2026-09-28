ROBOT BÚN ĐẬU V4.9 - RENDER DEPLOY FIX

Nguyên nhân lỗi build trước:
requirements.txt chỉ chứa faster-whisper nên Render không cài uvicorn và các dependency cũ của server.

Để deploy:
1. Dùng main.py trong gói này làm main.py của repo.
2. Dùng requirements.txt trong gói này làm requirements.txt.
3. Giữ Environment Variables hiện có của bạn (GEMINI_API_KEY và các biến TTS nếu đang dùng).
4. Build Command: pip install -r requirements.txt
5. Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

Các dependency chính:
- FastAPI + Uvicorn cho HTTP/WebSocket
- google-genai cho Gemini 3.8 Live / Gemini 3.6 Flash / Gemini TTS
- faster-whisper cho local STT fallback
- edge-tts + miniaudio cho TTS fallback / decode
- numpy + soxr cho xử lý và resample PCM

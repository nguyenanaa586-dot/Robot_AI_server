ROBOT BÚN ĐẬU SERVER V4.10.1

HOTFIX:
- Sửa HTTP 500 tại GET / do V4.10 còn tham chiếu LOCAL_STT_ENABLED/LOCAL_STT_MODEL/LOCAL_STT_COMPUTE_TYPE sau khi Local STT/Whisper đã bị loại bỏ.
- GET /healthz trả 200 ổn định cho Render/monitoring.
- Giữ Gemini 3.8 Live làm não chính.
- Khi Live hết quota/lỗi: chuyển sang Gemini 3.6 Flash và gửi trực tiếp cùng audio WAV từ ESP32; KHÔNG dùng Whisper/local STT.
- Giữ quota guard Live/Search/TTS và Edge-TTS fallback.

RENDER START COMMAND:
uvicorn main:app --host 0.0.0.0 --port $PORT

KIỂM TRA SAU DEPLOY:
- GET / -> HTTP 200
- GET /healthz -> HTTP 200
- Khi Live quota exhausted và robot nói xong, log phải xuất hiện:
  [FALLBACK 3.6] Live unavailable -> direct audio fallback (no local STT)

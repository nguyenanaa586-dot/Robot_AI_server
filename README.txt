ROBOT BÚN ĐẬU SERVER V4.10.2

Được xây trên V4.10.1, không đổi giao thức ESP32.

Sửa chính:
1) TTS/Edge-TTS không còn làm sập WebSocket khi ESP32 đã ngắt.
2) Nếu socket đã đóng, Edge-TTS dừng retry ngay thay vì gửi tiếp vào socket chết.
3) Nếu Gemini TTS + Edge-TTS đều thất bại nhưng ESP32 còn kết nối, server gửi tts_done thất bại để kết thúc state SPEAKING an toàn và giữ WebSocket sống.
4) Có dùng TTS semaphore để tránh nhiều phiên TTS chồng nhau.
5) Gemini 3.6 gặp 429/quota sẽ thử Key active tiếp theo. Khi tất cả Key đều hết quota, server ghi daily guard và không gọi lại 3.6 liên tục cho tới ngày kế tiếp theo giờ Việt Nam. Đây là guard dựa trên lỗi quota thật, không phải khẳng định một quota cố định của Google.

Render Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT

Health:
/
/healthz

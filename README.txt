ROBOT BÚN ĐẬU SERVER V4.9

1) Thay main.py bằng file robot_bun_dau_server_v4_9_local_stt_fallback.py và đổi tên thành main.py.
2) Trong requirements.txt hiện tại của Render, thêm đúng dòng:
   faster-whisper==1.2.1
   Không xóa các dependency đang chạy ổn.
3) Giữ Environment Variables hiện tại (GEMINI_API_KEY và các biến TTS nếu đang dùng). Không cần thêm biến mới để chạy mặc định.
4) Build Command tiếp tục dùng pip install -r requirements.txt (theo cấu hình Render hiện tại).
5) STT local dùng faster-whisper tiny, CPU/int8, VAD OFF; model chỉ tải khi Gemini 3.8 Live không dùng được.
6) Khi 3.8 hết quota: Audio -> local STT -> (giờ/ngày/thứ thì server trả lời trực tiếp) -> (câu thường thì Gemini 3.6 xử lý text) -> TTS.
7) Khi Gemini TTS trả 429/quota, server khóa Gemini TTS cho phiên chạy hiện tại và chuyển thẳng sang Edge-TTS để tránh chờ lại mỗi lượt.

Lưu ý Render Free: filesystem là ephemeral, nên model Whisper cache có thể phải tải lại sau restart/redeploy/spin-down.

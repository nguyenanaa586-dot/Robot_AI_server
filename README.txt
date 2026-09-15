Bun Dau Server V1.6 - Edge-TTS Seamless

Mục tiêu:
- Loại bỏ Piper vì inference trên Render Free quá chậm.
- Dùng Edge-TTS của Microsoft, miễn phí, với Hoài My làm voice chính và Nam Minh fallback.
- Không chia câu để phát từng đoạn. Toàn bộ câu trả lời Gemini được tổng hợp thành MỘT file PCM liên tục trước khi gửi.
- Giữ PCM16 / 16 kHz / mono và giao thức WebSocket hiện tại của ESP32.
- Giữ sticky Gemini API key.
- Không dùng thinking_config/thinking_level.
- Đọc trực tiếp candidate.content.parts để không gọi chunk.text và tránh cảnh báo thought_signature.
- Kiểm tra finish_reason trước khi gửi TTS.
- Retry Edge-TTS cùng voice trước khi chuyển fallback.
- TTS có timeout và semaphore để bảo vệ Render Free.
- Log performance Gemini first_text/total và TTS synth/audio duration.

Environment variables:
GEMINI_API_KEY=key1,key2,...
GEMINI_MODEL (optional)
GEMINI_MAX_OUTPUT_TOKENS (optional, default 1024)
TTS_VOICE (optional, default vi-VN-HoaiMyNeural)
TTS_FALLBACK_VOICE (optional, default vi-VN-NamMinhNeural)
TTS_RATE (optional, default +10%)
TTS_TIMEOUT_SECONDS (optional, default 25)
TTS_RETRIES_PER_VOICE (optional, default 2)
TTS_CONCURRENCY (optional, default 1)

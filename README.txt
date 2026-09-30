ROBOT BÚN ĐẬU SERVER V4.10.4

Base: V4.10.3. Giữ nguyên giao thức ESP32 WebSocket /ws/chat và PCM16 16kHz mono.

THAY ĐỔI CHÍNH
1) Gemini Live: 3.8 Live là model chính. Nếu 3.8 gặp quota/cooldown, server tự chuyển sang Gemini 3.1 Flash Live Preview thay vì rơi thẳng xuống batch 3.6.
2) Search: cả 3.8 Live và 3.1 Live đều được phép dùng Google Search khi search guard không bị khóa. Local Search Guard mặc định TẮT vì bộ đếm cục bộ không phải quota thật của Google.
3) Search quota: bỏ khóa persistent theo ngày cũ; lỗi Search thật chỉ tạo cooldown tạm thời 600 giây trong process.
4) Gemini 3.6 fallback: vẫn là direct audio, không dùng Whisper. Mặc định thinking=minimal để giảm độ trễ. Lỗi HTTP 500/502/503/504 được retry 1 lần sau 250 ms.
5) TTS: mặc định chuyển từ gemini-3.1-flash-tts-preview sang gemini-3.8-flash-lite-tts; đây là model TTS được Google mô tả là low-latency/cost-efficient cho voice-agent cascades. Có thể ghi đè bằng GEMINI_TTS_MODEL.
6) Giảm độ trễ: fallback timeout mặc định 14s; batch MAX_OUTPUT_TOKENS mặc định 384; reply tối đa mặc định 3 câu; TTS prebuffer server mặc định 192 ms.

ENV KHÔNG BẮT BUỘC THÊM
- GEMINI_SECONDARY_LIVE_MODEL (mặc định gemini-3.1-flash-live-preview)
- GEMINI_WEB_SEARCH_LOCAL_GUARD (mặc định false)
- GEMINI_FALLBACK_THINKING_LEVEL (mặc định minimal)
- GEMINI_TTS_MODEL (mặc định gemini-3.8-flash-lite-tts)

RENDER START COMMAND
uvicorn main:app --host 0.0.0.0 --port $PORT

LOG MONG ĐỢI KHI 3.8 KHÔNG KHẢ DỤNG
[LIVE QUOTA] ... temporarily_blocked=False ...
[LIVE] ... 1011 quota ...
[LIVE FALLBACK] Thu model realtime phu gemini-3.1-flash-live-preview
[LIVE] Session san sang | model=gemini-3.1-flash-live-preview ...
[LIVE FALLBACK] 3.8 khong kha dung -> dang dung gemini-3.1-flash-live-preview (Search capable)

KHI 3.1 LIVE CŨNG KHÔNG DÙNG ĐƯỢC
Server vẫn giữ fallback cuối: Gemini 3.6 Flash direct audio, search=OFF.

LƯU Ý QUOTA
Không có code nào có thể tạo thêm quota Google. V4.10.4 chỉ tránh khóa sai theo ngày và tận dụng một Live fallback có Search. Nếu cả 3.8 và 3.1 Live đều trả lỗi quota, cần kiểm tra quota thực tế của project trong Google AI Studio.

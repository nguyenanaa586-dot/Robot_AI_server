BUN DAU SERVER V2.1 - GEMINI CHUNK DEBUG

Mục đích
- Giữ nguyên kiến trúc server hiện tại của Bún Đậu.
- Thêm chẩn đoán chính xác từng text chunk Gemini để xác định tình trạng mất chữ như "èo", "ạn", "uất".
- Không đọc/ghi phần thought của Gemini vào log chunk.
- Giữ memory tối thiểu 10 lượt và Robot Command Protocol v1.
- Giữ Gemini 3.1 Flash TTS + Edge-TTS fallback.

Biến Render quan trọng
GEMINI_API_KEY=<key1,key2,...>
GEMINI_MODEL=gemini-3.6-flash
GEMINI_MAX_OUTPUT_TOKENS=384
GEMINI_THINKING_LEVEL=low
MEMORY_TURNS=10
GEMINI_DEBUG_CHUNKS=true

Sau khi xác định nguyên nhân mất chữ, đặt:
GEMINI_DEBUG_CHUNKS=false
để giảm log Render.

TTS
GEMINI_TTS_ENABLED=true
GEMINI_TTS_MODEL=gemini-3.1-flash-tts-preview
GEMINI_TTS_VOICE=Sulafat
GEMINI_TTS_LANGUAGE=vi-VN
GEMINI_TTS_STYLE=<có thể ghi đè nếu cần>
TTS_PREBUFFER_MS=320
TTS_TIMEOUT_SECONDS=25

Fallback
EDGE_TTS_ENABLED=true
EDGE_TTS_VOICE=vi-VN-HoaiMyNeural
EDGE_TTS_FALLBACK_VOICE=vi-VN-NamMinhNeural
EDGE_TTS_RATE=+10%

Các log chẩn đoán mới
[GEMINI CHUNK] initial/retry ... text='...'
[GEMINI RAW REPR] initial/retry: '...'
[GEMINI RAW NORMAL] initial/retry: ...
[GEMINI STREAM] ... chars=...
[GEMINI REPLY REPR] '...'
[BUN DAU REPR] '...'

Đọc log theo thứ tự này để xác định tầng bị mất ký tự.

Deploy Render
- Runtime Python 3.11 theo .python-version.
- Build command: pip install -r requirements.txt
- Start command có thể để Procfile: uvicorn main:app --host 0.0.0.0 --port $PORT

Lưu ý
- google-genai 2.23.0 là phiên bản được ghim để tránh thay đổi SDK trong lúc chẩn đoán.
- Retry transient giữ nguyên thinking_config giống request đầu.

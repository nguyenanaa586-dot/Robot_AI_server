BUN DAU SERVER V3.0 – COMMAND EXPANDED

Logic changes are in main.py. Support deployment files are retained from V2.9.

Added ACTION types:
- stop: dừng motor và giữ robot đứng yên.
- idle: cho phép chuyển động idle ngẫu nhiên trở lại.
- Explicit move/rotate commands now instruct the robot to stay still after completion.

Flash the paired ESP32 firmware ROBOT_BUN_DAU_V3_0_COMMAND_EXPANDED.ino for these physical commands.

BUN DAU SERVER V2.2 - GEMINI LIVE + VL53L0X READY

Mục đích
- Giữ nguyên personality, MEMORY tối thiểu 10 lượt, ACTION/EMOTION và TTS hiện tại.
- Thay pipeline Gemini chính từ WAV-after-record sang Gemini Live realtime input.
- PCM16 16 kHz mono từ ESP32 được chuyển tiếp ngay khi mic đang RECORDING.
- Input transcription của Live mặc định tắt để giảm message overhead; chỉ bật khi cần debug.
- Vẫn lưu một bản PCM trong RAM của process cho cùng lượt để fallback về generate_content nếu Live lỗi.
- Nhận khoảng cách VL53L0X từ ESP32 qua WebSocket event và dùng giá trị mới nhất làm context cho lượt nói kế tiếp.

Gemini Live
GEMINI_LIVE_ENABLED=true
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_LIVE_MAX_OUTPUT_TOKENS=384
GEMINI_LIVE_THINKING_LEVEL=low
GEMINI_LIVE_CONNECT_RETRIES=2
GEMINI_LIVE_TRANSCRIPT_LOG=false
GEMINI_LIVE_INPUT_TRANSCRIPTION=false
GEMINI_LIVE_HISTORY_RESET_TURNS=10

Lưu ý model
- Bản này dùng gemini-3.1-flash-live-preview vì model này vẫn hỗ trợ TEXT output, cần thiết để giữ giao thức <MEMORY>/<ACTION>/<REPLY> hiện tại và TTS riêng.
- Gemini 3.8 Live là model realtime mới hơn nhưng API hiện tại thiên về audio output; đổi sang 3.8 sẽ cần thiết kế lại đường ACTION/TTS.

Protocol mới từ ESP32
1) Bắt đầu nói:
   {"event":"start_speech"}
2) Trong lúc RECORDING: gửi binary PCM16 16kHz mono ngay từng chunk.
3) Có thể gửi ToF bất cứ lúc nào:
   {"event":"tof","distance_cm":123.4}
4) Kết thúc nói:
   {"event":"end_speech"}

ToF
- Server không đọc VL53L0X trực tiếp; ESP32 đọc cảm biến rồi gửi distance_cm.
- Giá trị hợp lệ: >0 đến 2000 cm.
- Server chỉ giữ giá trị mới nhất và đưa vào context ngay trước activity_start của lượt nói.
- Luôn giữ an toàn chuyển động tại ESP32; Gemini không phải lớp safety cuối cùng.

Fallback
- Nếu Gemini Live lỗi trong một lượt, server dùng PCM đã lưu trong RAM để gọi pipeline generate_content hiện tại.
- 503/timeout kiểu transient không tự vô hiệu hóa key.
- 401/403/429 hoặc lỗi key/quota sẽ đánh dấu key hiện tại disabled và chuyển sang key tiếp theo.
- Sticky key không quay lại key đã disabled trong cùng process.

Render
- Python 3.11 theo .python-version.
- Build: pip install -r requirements.txt
- Start: Procfile -> uvicorn main:app --host 0.0.0.0 --port $PORT

Biến TTS cũ vẫn giữ nguyên:
GEMINI_TTS_ENABLED=true
GEMINI_TTS_MODEL=gemini-3.1-flash-tts-preview
GEMINI_TTS_VOICE=Sulafat
GEMINI_TTS_LANGUAGE=vi-VN
EDGE_TTS_ENABLED=true
EDGE_TTS_VOICE=vi-VN-HoaiMyNeural
EDGE_TTS_FALLBACK_VOICE=vi-VN-NamMinhNeural
EDGE_TTS_RATE=+10%

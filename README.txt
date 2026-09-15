Bun Dau Server V1.7 - Reasoning + 10-turn memory

- Gemini 3.6 Flash.
- google-genai 2.23.0.
- thinking_level=medium by default (configurable with GEMINI_THINKING_LEVEL).
- Conversation memory: minimum 10 recent turns per ESP32 WebSocket session.
- Each turn stores a short model-generated summary of what the user said plus the robot reply.
- Current user audio is still sent as WAV; previous turns are kept as compact text, not old audio files.
- Output format is forced in the prompt using MEMORY and REPLY tags. Only REPLY is sent to TTS.
- Edge-TTS primary HoaiMy, fallback NamMinh.
- Full-response TTS remains one continuous synthesis job; no sentence-by-sentence playback.
- finish_reason is checked.
- thought_signature parts are ignored by reading content.parts directly.
- Sticky Gemini key behavior is preserved.

Render variables (optional):
GEMINI_THINKING_LEVEL=medium
MEMORY_TURNS=10
GEMINI_MAX_OUTPUT_TOKENS=768
TTS_VOICE=vi-VN-HoaiMyNeural
TTS_FALLBACK_VOICE=vi-VN-NamMinhNeural
TTS_RATE=+10%

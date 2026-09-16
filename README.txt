Bun Dau Server V2 - Gemini 3.1 Flash TTS, low-latency

PRIMARY AI:
- Gemini 3.6 Flash (text reasoning + memory)
- thinking_level=low by default: reasoning remains enabled while reducing latency.
- MAX_OUTPUT_TOKENS=384 by default because robot replies are intentionally short.

PRIMARY TTS:
- Gemini 3.1 Flash TTS Preview: gemini-3.1-flash-tts-preview
- Voice: Despina
- Language: vi-VN
- Gemini TTS supports streaming audio; server begins forwarding audio while synthesis is still running.
- Gemini TTS output is 24 kHz PCM16 mono; server resamples to the ESP32's existing 16 kHz PCM16 mono path.

FREE TIER:
Google currently lists Gemini 3.1 Flash TTS Preview Standard input/output as Free Tier. It is still a preview model and rate limits can apply.

FALLBACK:
- Edge-TTS vi-VN-HoaiMyNeural
- Edge-TTS vi-VN-NamMinhNeural

IMPORTANT:
- No Google Cloud TTS service-account JSON is needed in this version.
- Remove GOOGLE_TTS_CREDENTIALS_JSON and other Google Cloud TTS variables from Render; they are no longer used.
- GEMINI_API_KEY is used for both Gemini reasoning and Gemini TTS.

LATENCY TUNING:
- GEMINI_THINKING_LEVEL=low
- GEMINI_MAX_OUTPUT_TOKENS=384
- TTS_PREBUFFER_MS=320
The 320 ms prebuffer is intentionally small because the ESP32 playback path already buffers audio. Lower it to 200 if testing shows stable playback; raise to 500 if Wi-Fi jitter causes underruns.

ENVIRONMENT:
GEMINI_API_KEY=key1,key2,...
GEMINI_MODEL=gemini-3.6-flash
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=384
MEMORY_TURNS=10
GEMINI_TTS_ENABLED=true
GEMINI_TTS_MODEL=gemini-3.1-flash-tts-preview
GEMINI_TTS_VOICE=Despina
GEMINI_TTS_LANGUAGE=vi-VN
TTS_PREBUFFER_MS=320
EDGE_TTS_ENABLED=true
EDGE_TTS_VOICE=vi-VN-HoaiMyNeural
EDGE_TTS_FALLBACK_VOICE=vi-VN-NamMinhNeural

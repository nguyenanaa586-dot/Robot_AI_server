Bun Dau Server V1.5 RenderFixed Fix2

- Gemini thinking_config/thinking_level removed.
- Piper Ban Mai uses bundled banmai.onnx.json plus downloaded banmai.onnx.
- Model download has two verified Hugging Face mirrors and minimum-size validation.
- Model + config are treated as a pair when fallback download is needed.
- Gemini response reads candidate.content.parts directly instead of chunk.text, avoiding warnings from thought_signature/non-text parts.
- Gemini finish_reason is checked; incomplete response reasons are rejected before TTS.
- TTS output is resampled from Piper 22050 Hz to PCM16 16 kHz mono.
- Default TTS volume/speed preserved from V1.5.

Render environment variables:
GEMINI_API_KEY
GEMINI_MODEL (optional)
GEMINI_MAX_OUTPUT_TOKENS (optional, default 1024)
PIPER_MODEL_DIR (optional, default models)
PIPER_MODEL_NAME (optional, default banmai)
PIPER_SPEED (optional, default 1.04)
PIPER_VOLUME (optional, default 1.05)

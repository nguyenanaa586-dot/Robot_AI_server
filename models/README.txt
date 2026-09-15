Piper Ban Mai model files

- banmai.onnx.json is bundled with the server package.
- banmai.onnx is downloaded on first TTS use from a verified mirror.
- The server downloads model and config as a pair if the bundled config is unavailable.
- Config sample rate: 22050 Hz; server resamples to 16 kHz PCM16 mono for ESP32.

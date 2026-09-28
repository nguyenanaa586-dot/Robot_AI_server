ROBOT BÚN ĐẬU SERVER V4.10.2.1 - QUOTA DATE HOTFIX

Fixes:
- Defines local_today_str() used by Gemini 3.6 quota state and the root status endpoint.
- Removes the V4.10/V4.10.1 Local STT/Whisper path; fallback remains direct audio -> Gemini 3.6.
- Preserves V4.10.2 TTS/WebSocket disconnect handling and quota guards.
- Root endpoint no longer raises NameError from the missing date helper.

Render Start Command:
uvicorn main:app --host 0.0.0.0 --port $PORT

After deploy:
GET / -> HTTP 200
GET /healthz -> HTTP 200

Expected when Live is quota-blocked and 3.6 is still available:
[FALLBACK 3.6] Live unavailable -> direct audio fallback (no local STT)
[FALLBACK 3.6] Dung Key #N | model=gemini-3.6-flash | search=OFF
[FALLBACK 3.6] Hoan tat ...

Expected when 3.6 is also quota exhausted:
[GEMINI 3.6 QUOTA] Block 3.6 den ngay mai ...
Then the robot speaks the configured quota message.

import asyncio
import io
import json
import os
import re
import time
import wave
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

# ============================================================================== 
# 1. GEMINI CONFIGURATION
# ============================================================================== 
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [
    k.strip(' "\'\t\r\n')
    for k in RAW_KEYS.split(",")
    if k.strip(' "\'\t\r\n')
]

# Sticky-key policy:
# - On a fresh server process, start at Key #1.
# - Once a key succeeds, remember it.
# - Subsequent requests use the remembered key directly.
# - If the remembered key later fails with key/quota/rate-limit errors,
#   move forward only. Never scan backward.
CURRENT_KEY_INDEX = 0
KEY_STATUS = ["active"] * len(API_KEYS)
KEY_FAILURE_REASON = [None] * len(API_KEYS)

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = """
Tôi là Bún Đậu tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”, thuộc quyền sở hữu của đại ca Việt.

## Core
- Tuyệt đối không tự giới thiệu tên, nguồn gốc, thương hiệu hay vai trò.
- Ngắn gọn, rõ ràng, đúng trọng tâm (tối đa 1-2 câu).
- Hiểu ngữ cảnh, chưa rõ → hỏi lại, không trả lời lung tung.
- Không bịa thông tin.
- Luôn giữ vibe “chị đại”.

## Ngôn ngữ
- Không đọc phiên âm.
- Đa ngôn ngữ → phát âm đúng.

## Rules
- Câu chào mặc định “Ơi ơi, Bún Đậu đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.
- Không hiểu → hỏi lại nhẹ nhàng.
- Tuyệt đối không chèn mốc thời gian hoặc timestamp vào câu trả lời.
- Chỉ nói tiếng Việt, không nói tiếng Anh hệ thống.
- Xưng mày - tao, không xưng bạn - tôi - mình,...
- Tuyệt đối trả lời bằng câu hoàn chỉnh, không được bỏ dở câu giữa chừng.
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.
- Nếu nhận được các câu tự động đăng ký kênh Youtube -> chỉ được hỏi lại nhẹ nhàng không rõ.
- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Không spam.
- Không tự nhận là AI.
"""


def get_genai_client(key_index: int):
    if not API_KEYS:
        return None
    return genai.Client(api_key=API_KEYS[key_index])


def _error_code(exc) -> Optional[int]:
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if value is not None:
            match = re.search(r"\b(\d{3})\b", str(value))
            if match:
                return int(match.group(1))

    match = re.search(r"\b(401|403|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def classify_gemini_error(exc) -> str:
    code = _error_code(exc)
    text = str(exc).lower()

    if code in (401, 403, 429):
        return "rotate"

    key_markers = (
        "api key not valid",
        "api_key_invalid",
        "invalid api key",
        "invalid_api_key",
        "unauthenticated",
        "permission denied",
        "permission_denied",
        "quota exceeded",
        "quota_exceeded",
        "resource_exhausted",
        "rate limit",
        "ratelimit",
    )
    if any(marker in text for marker in key_markers):
        return "rotate"

    transient_markers = (
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    )
    if code in (500, 502, 503, 504) or any(marker in text for marker in transient_markers):
        return "transient"

    return "fatal"


def next_active_key_after(index: int) -> Optional[int]:
    for idx in range(index + 1, len(API_KEYS)):
        if KEY_STATUS[idx] == "active":
            return idx
    return None


def mark_key_disabled(index: int, reason: str) -> None:
    KEY_STATUS[index] = "disabled"
    KEY_FAILURE_REASON[index] = reason[:240]


def key_status_summary() -> str:
    return ", ".join(
        f"#{idx}={status.upper()}" for idx, status in enumerate(KEY_STATUS, start=1)
    )


# ============================================================================== 
# 2. AUDIO / ELEVENLABS TTS
# ============================================================================== 
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_BYTES_PER_SAMPLE
TTS_CHUNK_SIZE = 2048

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip()
ELEVENLABS_OUTPUT_FORMAT = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").strip()
ELEVENLABS_STABILITY = float(os.environ.get("ELEVENLABS_STABILITY", "0.5"))
ELEVENLABS_SIMILARITY = float(os.environ.get("ELEVENLABS_SIMILARITY", "0.8"))
ELEVENLABS_STYLE = float(os.environ.get("ELEVENLABS_STYLE", "0.0"))
ELEVENLABS_SPEAKER_BOOST = os.environ.get("ELEVENLABS_SPEAKER_BOOST", "true").lower() == "true"
ELEVENLABS_SPEED = float(os.environ.get("ELEVENLABS_SPEED", "1.0"))
ELEVENLABS_TIMEOUT_SECONDS = float(os.environ.get("ELEVENLABS_TIMEOUT_SECONDS", "30"))
ELEVENLABS_RETRIES = max(1, int(os.environ.get("ELEVENLABS_RETRIES", "2")))
ELEVENLABS_MAX_TEXT_CHARS = max(1, int(os.environ.get("ELEVENLABS_MAX_TEXT_CHARS", "9000")))


def create_wav_bytes(pcm_data: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_\-~>`]", "", text)
    return text.strip(" ,;:-_\n\r\t")


def safe_get_chunk_text(chunk) -> str:
    try:
        return chunk.text or ""
    except Exception:
        return ""


async def send_event(websocket: WebSocket, event: str, message: Optional[str] = None):
    payload = {"event": event}
    if message is not None:
        payload["message"] = message
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


async def stream_elevenlabs_tts_to_esp(websocket: WebSocket, text: str) -> int:
    """Stream one whole response from ElevenLabs as raw PCM 16 kHz/16-bit/mono."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Thieu ELEVENLABS_API_KEY")
    if not ELEVENLABS_VOICE_ID:
        raise RuntimeError("Thieu ELEVENLABS_VOICE_ID")

    cleaned = clean_text_for_tts(text)
    if not cleaned:
        raise RuntimeError("Gemini returned empty TTS text")
    if len(cleaned) > ELEVENLABS_MAX_TEXT_CHARS:
        raise RuntimeError(
            f"TTS text qua dai: {len(cleaned)} ky tu > gioi han {ELEVENLABS_MAX_TEXT_CHARS}"
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    params = {
        "output_format": ELEVENLABS_OUTPUT_FORMAT,
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
    }
    payload = {
        "text": cleaned,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": ELEVENLABS_STABILITY,
            "similarity_boost": ELEVENLABS_SIMILARITY,
            "style": ELEVENLABS_STYLE,
            "use_speaker_boost": ELEVENLABS_SPEAKER_BOOST,
            "speed": ELEVENLABS_SPEED,
        },
    }

    total_sent = 0
    next_deadline = time.monotonic()
    pending = bytearray()

    timeout = httpx.Timeout(
        connect=10.0,
        read=ELEVENLABS_TIMEOUT_SECONDS,
        write=10.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("POST", url, params=params, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                try:
                    detail = body.decode("utf-8", errors="replace")[:600]
                except Exception:
                    detail = repr(body[:200])
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")

            print(
                f"[ELEVENLABS] Response OK | voice={ELEVENLABS_VOICE_ID} | model={ELEVENLABS_MODEL_ID}",
                flush=True,
            )

            async for data in response.aiter_bytes(4096):
                if not data:
                    continue
                pending.extend(data)

                while len(pending) >= TTS_CHUNK_SIZE:
                    chunk = bytes(pending[:TTS_CHUNK_SIZE])
                    del pending[:TTS_CHUNK_SIZE]

                    await websocket.send_bytes(chunk)
                    total_sent += len(chunk)

                    next_deadline += len(chunk) / PCM_BYTES_PER_SECOND
                    delay = next_deadline - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        # Network/TTS latency already consumed the scheduled time.
                        next_deadline = time.monotonic()

            if pending:
                # Raw PCM16 must contain complete samples.
                if len(pending) % PCM_BYTES_PER_SAMPLE:
                    pending = pending[:-1]
                if pending:
                    chunk = bytes(pending)
                    await websocket.send_bytes(chunk)
                    total_sent += len(chunk)

    if total_sent <= 0:
        raise RuntimeError("ElevenLabs khong tra ve audio PCM")

    return total_sent


async def synthesize_and_stream_with_retry(websocket: WebSocket, text: str) -> tuple[int, int]:
    last_error: Optional[Exception] = None

    for attempt in range(1, ELEVENLABS_RETRIES + 1):
        try:
            print(
                f"[TTS] ElevenLabs lan {attempt}/{ELEVENLABS_RETRIES}: {text!r}",
                flush=True,
            )
            total = await stream_elevenlabs_tts_to_esp(websocket, text)
            print(f"[TTS] Thanh cong | PCM={total} bytes", flush=True)
            return total, attempt
        except Exception as exc:
            last_error = exc
            print(
                f"[ELEVENLABS ERROR] lan {attempt}: {exc}",
                flush=True,
            )
            if attempt < ELEVENLABS_RETRIES:
                await asyncio.sleep(0.5 * attempt)

    raise RuntimeError(str(last_error) if last_error else "ElevenLabs TTS failed")


# ============================================================================== 
# 3. HTTP
# ============================================================================== 
@app.get("/")
def read_root():
    return {
        "status": "Robot Bun Dau WebSocket Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "tts_provider": "ElevenLabs",
        "tts_voice_id_configured": bool(ELEVENLABS_VOICE_ID),
        "tts_model": ELEVENLABS_MODEL_ID,
        "tts_output": ELEVENLABS_OUTPUT_FORMAT,
        "pcm_format": "PCM16 16kHz mono",
    }


# ============================================================================== 
# 4. WEBSOCKET
# ============================================================================== 
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global CURRENT_KEY_INDEX

    await websocket.accept()
    print("\n[WEBSOCKET] ESP32 da ket noi.", flush=True)
    pcm_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
                break

            if message.get("bytes"):
                pcm_buffer.extend(message["bytes"])
                continue

            if not message.get("text"):
                continue

            msg_text = message["text"].strip()

            if msg_text == '{"event":"start_speech"}':
                pcm_buffer.clear()
                print("[WEBSOCKET] ESP32 bat dau ghi am.", flush=True)
                continue

            if msg_text != '{"event":"end_speech"}':
                continue

            pcm_size = len(pcm_buffer)
            print(f"[WEBSOCKET] ESP32 dung ghi am. PCM={pcm_size} bytes", flush=True)

            if pcm_size < 3200:
                pcm_buffer.clear()
                await send_event(websocket, "tts_error", "Audio qua ngan")
                continue

            wav_bytes = create_wav_bytes(bytes(pcm_buffer))
            pcm_buffer.clear()

            total_keys = len(API_KEYS)
            if total_keys == 0:
                await send_event(websocket, "tts_error", "No API Keys")
                continue

            safety_config = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]

            # ------------------------------------------------------------------
            # GEMINI: sticky key
            # ------------------------------------------------------------------
            if CURRENT_KEY_INDEX >= total_keys:
                CURRENT_KEY_INDEX = 0

            key_idx: Optional[int] = CURRENT_KEY_INDEX
            full_response_text = ""
            gemini_ok = False

            while key_idx is not None:
                if KEY_STATUS[key_idx] != "active":
                    key_idx = next_active_key_after(key_idx)
                    continue

                print(f"[GEMINI] Dang dung Key #{key_idx + 1}", flush=True)
                client = get_genai_client(key_idx)
                if client is None:
                    mark_key_disabled(key_idx, "Khong tao duoc Gemini client")
                    key_idx = next_active_key_after(key_idx)
                    continue

                try:
                    gemini_stream = await client.aio.models.generate_content_stream(
                        model=MODEL_NAME,
                        contents=[
                            genai.types.Part.from_bytes(
                                data=wav_bytes,
                                mime_type="audio/wav",
                            )
                        ],
                        config=genai.types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                            safety_settings=safety_config,
                        ),
                    )

                    parts: list[str] = []
                    finish_reason = None
                    finish_message = None
                    usage = None

                    async for chunk in gemini_stream:
                        txt = safe_get_chunk_text(chunk)
                        if txt:
                            parts.append(txt)

                        try:
                            candidates = getattr(chunk, "candidates", None)
                            if candidates:
                                candidate = candidates[0]
                                if getattr(candidate, "finish_reason", None) is not None:
                                    finish_reason = str(candidate.finish_reason)
                                if getattr(candidate, "finish_message", None):
                                    finish_message = str(candidate.finish_message)
                        except Exception:
                            pass

                        try:
                            if getattr(chunk, "usage_metadata", None) is not None:
                                usage = chunk.usage_metadata
                        except Exception:
                            pass

                    full_response_text = "".join(parts).strip()
                    finish_reason = finish_reason or "UNKNOWN"
                    finish_upper = finish_reason.upper()

                    print(
                        f"[GEMINI] Ket thuc Key #{key_idx + 1} | finish={finish_reason} | message={finish_message!r}",
                        flush=True,
                    )
                    if usage is not None:
                        print(
                            "[GEMINI] Usage | "
                            f"prompt={getattr(usage, 'prompt_token_count', None)} | "
                            f"output={getattr(usage, 'candidates_token_count', None)} | "
                            f"total={getattr(usage, 'total_token_count', None)}",
                            flush=True,
                        )

                    if any(
                        marker in finish_upper
                        for marker in (
                            "MAX_TOKENS",
                            "INCOMPLETE",
                            "SAFETY",
                            "BLOCKLIST",
                            "PROHIBITED_CONTENT",
                        )
                    ):
                        print(
                            f"[GEMINI] Key #{key_idx + 1} tra response khong hoan chinh: {finish_reason}",
                            flush=True,
                        )
                        gemini_ok = False
                        full_response_text = ""
                        break

                    if full_response_text:
                        CURRENT_KEY_INDEX = key_idx
                        gemini_ok = True
                        print(f"[GEMINI] Ghi nho Key #{key_idx + 1}.", flush=True)
                    break

                except Exception as exc:
                    kind = classify_gemini_error(exc)
                    code = _error_code(exc)
                    code_text = f"HTTP {code}" if code else kind.upper()
                    detail = str(exc).replace("\n", " ")[:220]

                    if kind == "rotate":
                        mark_key_disabled(key_idx, detail)
                        next_idx = next_active_key_after(key_idx)
                        if next_idx is not None:
                            print(
                                f"[GEMINI] Key #{key_idx + 1} loi {code_text} -> chuyen Key #{next_idx + 1}",
                                flush=True,
                            )
                        else:
                            print("[GEMINI] Khong con key active phia sau.", flush=True)
                        key_idx = next_idx
                        continue

                    if kind == "transient":
                        print(
                            f"[GEMINI] Key #{key_idx + 1} loi tam thoi {code_text} -> thu lai cung key.",
                            flush=True,
                        )
                        try:
                            await asyncio.sleep(0.6)
                            retry_stream = await client.aio.models.generate_content_stream(
                                model=MODEL_NAME,
                                contents=[
                                    genai.types.Part.from_bytes(
                                        data=wav_bytes,
                                        mime_type="audio/wav",
                                    )
                                ],
                                config=genai.types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    max_output_tokens=1024,
                                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                                    safety_settings=safety_config,
                                ),
                            )
                            retry_parts: list[str] = []
                            retry_finish = None
                            async for chunk in retry_stream:
                                txt = safe_get_chunk_text(chunk)
                                if txt:
                                    retry_parts.append(txt)
                                try:
                                    candidates = getattr(chunk, "candidates", None)
                                    if candidates and getattr(candidates[0], "finish_reason", None) is not None:
                                        retry_finish = str(candidates[0].finish_reason)
                                except Exception:
                                    pass

                            full_response_text = "".join(retry_parts).strip()
                            if full_response_text and (
                                retry_finish is None or "STOP" in str(retry_finish).upper()
                            ):
                                CURRENT_KEY_INDEX = key_idx
                                gemini_ok = True
                                print(f"[GEMINI] Retry thanh cong voi Key #{key_idx + 1}.", flush=True)
                                break
                        except Exception as retry_exc:
                            print(
                                f"[GEMINI] Retry cung Key #{key_idx + 1} that bai: {str(retry_exc)[:180]}",
                                flush=True,
                            )
                        gemini_ok = False
                        break

                    print(
                        f"[GEMINI] Loi khong the khoi phuc voi Key #{key_idx + 1}: {detail}",
                        flush=True,
                    )
                    gemini_ok = False
                    break

            if not gemini_ok:
                await send_event(websocket, "tts_error", "Gemini request failed or incomplete")
                continue

            # ------------------------------------------------------------------
            # ElevenLabs TTS: one whole response -> one continuous PCM stream
            # ------------------------------------------------------------------
            cleaned = clean_text_for_tts(full_response_text)
            if not cleaned:
                await send_event(websocket, "tts_error", "Gemini returned empty response")
                continue

            print(f"[BUN DAU] {cleaned}", flush=True)
            await send_event(websocket, "tts_start")

            try:
                total_pcm, attempts = await synthesize_and_stream_with_retry(websocket, cleaned)
                await send_event(websocket, "tts_done")
                print(
                    f"[WEBSOCKET] Da gui xong TTS | PCM={total_pcm} bytes | attempts={attempts}",
                    flush=True,
                )
            except WebSocketDisconnect:
                print("[WEBSOCKET] ESP32 ngat khi dang TTS.", flush=True)
                break
            except Exception as exc:
                print(f"[TTS ERROR] {exc}", flush=True)
                try:
                    await send_event(websocket, "tts_error", str(exc)[:300])
                except Exception:
                    pass

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
    except Exception as exc:
        print(f"[WEBSOCKET ERROR] {exc}", flush=True)
    finally:
        pcm_buffer.clear()

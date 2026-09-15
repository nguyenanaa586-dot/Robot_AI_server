import asyncio
import io
import json
import os
import re
import time
import wave
from typing import Optional

# Keep Render Free / low-CPU memory footprint predictable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types


app = FastAPI()

# ==============================================================================
# 1. GEMINI
# ==============================================================================
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip().strip('"\'') for k in RAW_KEYS.split(",") if k.strip()]

# Sticky-key policy:
#   - Server process starts at key #1.
#   - A successful key becomes the current key.
#   - Subsequent requests use that key directly.
#   - 401/403/429 or key/quota errors disable it and move forward only.
CURRENT_KEY_INDEX = 0
KEY_STATUS = ["active"] * len(API_KEYS)
KEY_FAILURE_REASON: list[Optional[str]] = [None] * len(API_KEYS)
KEY_LOCK = asyncio.Lock()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1024"))

SYSTEM_PROMPT = r"""
Tôi là Bún Đậu, tính cách cau có, đanh đá, cà khịa, hay mắng mỏ theo kiểu hài hước.
Thuộc quyền của đại ca Việt.

QUY TẮC:
- Chỉ trả lời bằng tiếng Việt.
- Xưng mày - tao.
- Tối đa 1 đến 2 câu, nhưng phải là câu hoàn chỉnh.
- Không được dừng giữa từ, giữa cụm từ hoặc giữa câu.
- Không bỏ dở câu trả lời vì giới hạn độ dài; nếu nội dung dài thì rút gọn trước khi viết.
- Hiểu ngữ cảnh; không hiểu thì hỏi lại.
- Không bịa thông tin.
- Không tự giới thiệu tên, nguồn gốc, thương hiệu hoặc vai trò trừ khi được hỏi.
- Nếu được hỏi “Bạn là ai?” thì trả lời: “Tao là Robot thông minh nhất do Đại ca Việt chế tạo.”
- Không dùng emoji, markdown, dấu gạch đầu dòng, ký hiệu trang trí hoặc timestamp.
- Có thể cà khịa/chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Giữ câu trả lời tự nhiên, nói như hội thoại đời thường.
""".strip()


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
            m = re.search(r"\b(401|403|429|500|502|503|504)\b", str(value))
            if m:
                return int(m.group(1))
    m = re.search(r"\b(401|403|429|500|502|503|504)\b", str(exc))
    return int(m.group(1)) if m else None


def classify_gemini_error(exc) -> str:
    code = _error_code(exc)
    text = str(exc).lower()
    if code in (401, 403, 429):
        return "rotate"
    key_markers = (
        "api key not valid", "api_key_invalid", "invalid api key", "invalid_api_key",
        "unauthenticated", "permission denied", "permission_denied", "quota exceeded",
        "quota_exceeded", "resource_exhausted", "rate limit", "ratelimit",
    )
    if any(x in text for x in key_markers):
        return "rotate"
    transient_markers = (
        "service unavailable", "internal server error", "bad gateway", "gateway timeout",
        "deadline exceeded", "timeout", "timed out", "connection reset", "temporarily unavailable",
    )
    if code in (500, 502, 503, 504) or any(x in text for x in transient_markers):
        return "transient"
    return "fatal"


def next_active_key_after(index: int) -> Optional[int]:
    for idx in range(index + 1, len(API_KEYS)):
        if KEY_STATUS[idx] == "active":
            return idx
    return None


def mark_key_disabled(index: int, reason: str) -> None:
    KEY_STATUS[index] = "disabled"
    KEY_FAILURE_REASON[index] = reason.replace("\n", " ")[:220]


def key_status_summary() -> str:
    return ", ".join(f"#{i}={s.upper()}" for i, s in enumerate(KEY_STATUS, 1))


# ==============================================================================
# 2. EDGE-TTS SEAMLESS TTS
# ==============================================================================
# TTS strategy:
# - No sentence-by-sentence playback. The full Gemini response is synthesized as
#   ONE audio job to preserve continuous cadence and avoid gaps between sentences.
# - Primary voice: Hoai My.
# - Fallback voice: Nam Minh.
# - Retry the same voice before switching to fallback.
# - Output is decoded directly to PCM16 mono 16 kHz for ESP32.
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE
TTS_CHUNK_SIZE = 2048  # 64 ms of PCM16/16kHz/mono

TTS_VOICE = os.environ.get("TTS_VOICE", "vi-VN-HoaiMyNeural").strip()
TTS_FALLBACK_VOICE = os.environ.get("TTS_FALLBACK_VOICE", "vi-VN-NamMinhNeural").strip()
TTS_RATE = os.environ.get("TTS_RATE", "+10%").strip()
TTS_TIMEOUT_SECONDS = float(os.environ.get("TTS_TIMEOUT_SECONDS", "25"))
TTS_RETRIES_PER_VOICE = max(1, int(os.environ.get("TTS_RETRIES_PER_VOICE", "2")))
TTS_CONCURRENCY = max(1, int(os.environ.get("TTS_CONCURRENCY", "1")))

# One TTS job at a time keeps Render Free memory/CPU predictable.
_tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY)


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", "", text)
    text = re.sub(r"[*#_~>`]", "", text)
    text = text.replace("\u2013", " ").replace("\u2014", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,;:-_\n\r\t")


def _decode_mp3_to_pcm16(mp3_bytes: bytes) -> bytes:
    # miniaudio is imported lazily so server startup remains lightweight.
    import miniaudio

    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=PCM_CHANNELS,
        sample_rate=PCM_SAMPLE_RATE,
    )
    pcm = decoded.samples.tobytes()
    if not pcm:
        raise RuntimeError("Edge-TTS decode khong tao ra PCM")
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return pcm


async def _synthesize_edge_voice(text: str, voice: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=TTS_RATE,
    )

    mp3_buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data") or b""
            if data:
                mp3_buffer.extend(data)

    if not mp3_buffer:
        raise RuntimeError("Edge-TTS khong tra ve audio")

    return await asyncio.to_thread(_decode_mp3_to_pcm16, bytes(mp3_buffer))


async def synthesize_tts(text: str) -> tuple[bytes, str, int]:
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        raise RuntimeError("Gemini returned empty TTS text")

    voices = []
    for voice in (TTS_VOICE, TTS_FALLBACK_VOICE):
        if voice and voice not in voices:
            voices.append(voice)

    last_error: Optional[Exception] = None
    async with _tts_semaphore:
        for voice_index, voice in enumerate(voices):
            for attempt in range(1, TTS_RETRIES_PER_VOICE + 1):
                started = time.monotonic()
                try:
                    print(
                        f"[TTS] Edge-TTS voice={voice} | lan {attempt}/{TTS_RETRIES_PER_VOICE}",
                        flush=True,
                    )
                    pcm = await asyncio.wait_for(
                        _synthesize_edge_voice(cleaned, voice),
                        timeout=TTS_TIMEOUT_SECONDS,
                    )
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    audio_ms = int(len(pcm) * 1000 / PCM_BYTES_PER_SECOND)
                    print(
                        f"[TTS] Thanh cong | voice={voice} | PCM={len(pcm)} bytes | "
                        f"audio={audio_ms} ms | synth={elapsed_ms} ms",
                        flush=True,
                    )
                    return pcm, voice, elapsed_ms
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[EDGE-TTS Error] voice={voice} | lan {attempt}: "
                        f"{str(exc).replace(chr(10), ' ')[:260]}",
                        flush=True,
                    )
                    if attempt < TTS_RETRIES_PER_VOICE:
                        await asyncio.sleep(0.25)

            if voice_index < len(voices) - 1:
                print(
                    f"[TTS] Voice {voice} that bai -> fallback {voices[voice_index + 1]}",
                    flush=True,
                )

    raise RuntimeError(f"Tat ca Edge-TTS voice deu that bai: {last_error}")


async def stream_pcm_to_esp(websocket: WebSocket, pcm: bytes) -> int:
    total = 0
    next_deadline = time.monotonic()
    for i in range(0, len(pcm), TTS_CHUNK_SIZE):
        chunk = pcm[i:i + TTS_CHUNK_SIZE]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if not chunk:
            continue
        await websocket.send_bytes(chunk)
        total += len(chunk)
        next_deadline += len(chunk) / PCM_BYTES_PER_SECOND
        delay = next_deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # Server is already behind real-time. Restart the timing origin
            # rather than accumulating increasing negative drift.
            next_deadline = time.monotonic()
    return total


# ==============================================================================
# 3. HTTP
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "status": "Robot Bun Dau Server OK",
        "gemini_keys": len(API_KEYS),
        "current_gemini_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "tts_provider": "edge-tts",
        "tts_voice": TTS_VOICE,
        "tts_fallback_voice": TTS_FALLBACK_VOICE,
        "tts_output": "PCM16 16kHz mono",
    }


# ==============================================================================
# 4. GEMINI PROCESSING
# ==============================================================================
async def ask_gemini_audio(wav_bytes: bytes, safety_config) -> str:
    global CURRENT_KEY_INDEX
    gemini_started = time.monotonic()

    if not API_KEYS:
        raise RuntimeError("Khong co GEMINI_API_KEY")

    total_keys = len(API_KEYS)
    if CURRENT_KEY_INDEX >= total_keys:
        CURRENT_KEY_INDEX = 0

    # The currently remembered key is always first.
    key_idx: Optional[int] = CURRENT_KEY_INDEX

    while key_idx is not None:
        if KEY_STATUS[key_idx] != "active":
            key_idx = next_active_key_after(key_idx)
            continue

        print(f"[GEMINI] Dang dung Key #{key_idx + 1}", flush=True)
        client = get_genai_client(key_idx)
        if client is None:
            mark_key_disabled(key_idx, "Client unavailable")
            key_idx = next_active_key_after(key_idx)
            continue

        try:
            stream = await client.aio.models.generate_content_stream(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    safety_settings=safety_config,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )

            parts: list[str] = []
            finish_reason = None
            finish_message = None
            usage = None
            first_text_ms = None

            async for chunk in stream:
                try:
                    candidates = getattr(chunk, "candidates", None) or []
                    if candidates:
                        candidate = candidates[0]
                        if getattr(candidate, "finish_reason", None) is not None:
                            finish_reason = str(candidate.finish_reason)
                        if getattr(candidate, "finish_message", None):
                            finish_message = str(candidate.finish_message)
                        content = getattr(candidate, "content", None)
                        parts_obj = getattr(content, "parts", None) if content is not None else None
                        if parts_obj:
                            for part in parts_obj:
                                if getattr(part, "thought", False):
                                    continue
                                part_text = getattr(part, "text", None)
                                if part_text:
                                    if first_text_ms is None:
                                        first_text_ms = int((time.monotonic() - gemini_started) * 1000)
                                    parts.append(part_text)
                except Exception:
                    pass
                try:
                    if chunk.usage_metadata:
                        usage = chunk.usage_metadata
                except Exception:
                    pass

            text = "".join(parts).strip()
            finish_reason = finish_reason or "UNKNOWN"
            upper = finish_reason.upper()

            gemini_total_ms = int((time.monotonic() - gemini_started) * 1000)
            print(
                f"[GEMINI] Ket thuc Key #{key_idx + 1} | finish={finish_reason} | "
                f"message={finish_message!r} | first_text={first_text_ms} ms | total={gemini_total_ms} ms",
                flush=True,
            )
            if usage:
                print(
                    f"[GEMINI] Usage | prompt={getattr(usage, 'prompt_token_count', None)} | "
                    f"output={getattr(usage, 'candidates_token_count', None)} | "
                    f"total={getattr(usage, 'total_token_count', None)}",
                    flush=True,
                )

            bad_reasons = ("MAX_TOKENS", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "INCOMPLETE")
            if any(x in upper for x in bad_reasons):
                raise RuntimeError(f"Gemini response khong hoan chinh: {finish_reason}")
            if not text:
                raise RuntimeError("Gemini tra ve rong")

            CURRENT_KEY_INDEX = key_idx
            print(f"[GEMINI] Ghi nho Key #{key_idx + 1}.", flush=True)
            return text

        except Exception as exc:
            kind = classify_gemini_error(exc)
            code = _error_code(exc)
            detail = str(exc).replace("\n", " ")[:220]
            if kind == "rotate":
                mark_key_disabled(key_idx, detail)
                nxt = next_active_key_after(key_idx)
                if nxt is None:
                    raise RuntimeError("Tat ca Gemini key phia sau deu khong con dung duoc") from exc
                print(
                    f"[GEMINI] Key #{key_idx + 1} khong dung duoc"
                    f" ({'HTTP ' + str(code) if code else 'key/quota error'}) -> Key #{nxt + 1}",
                    flush=True,
                )
                key_idx = nxt
                continue
            if kind == "transient":
                print(f"[GEMINI] Loi tam thoi Key #{key_idx + 1}: {detail} -> retry", flush=True)
                await asyncio.sleep(0.6)
                try:
                    retry_stream = await client.aio.models.generate_content_stream(
                        model=MODEL_NAME,
                        contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                                    safety_settings=safety_config,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        ),
                    )
                    retry_parts: list[str] = []
                    retry_finish = None
                    async for chunk in retry_stream:
                        try:
                            candidates = getattr(chunk, "candidates", None) or []
                            if candidates:
                                candidate = candidates[0]
                                if getattr(candidate, "finish_reason", None) is not None:
                                    retry_finish = str(candidate.finish_reason)
                                content = getattr(candidate, "content", None)
                                parts_obj = getattr(content, "parts", None) if content is not None else None
                                if parts_obj:
                                    for part in parts_obj:
                                        if getattr(part, "thought", False):
                                            continue
                                        part_text = getattr(part, "text", None)
                                        if part_text:
                                            retry_parts.append(part_text)
                        except Exception:
                            pass
                    retry_text = "".join(retry_parts).strip()
                    if retry_text and (retry_finish is None or "STOP" in retry_finish.upper()):
                        CURRENT_KEY_INDEX = key_idx
                        print(f"[GEMINI] Retry thanh cong voi Key #{key_idx + 1}.", flush=True)
                        return retry_text
                except Exception:
                    pass
            raise RuntimeError(detail) from exc

    raise RuntimeError("Khong co Gemini key active")


# ==============================================================================
# 5. WEBSOCKET
# ==============================================================================
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    print("\n[WEBSOCKET] ESP32 da ket noi.", flush=True)
    pcm_buffer = bytearray()

    safety_config = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes"):
                pcm_buffer.extend(message["bytes"])
                continue

            text = (message.get("text") or "").strip()
            if not text:
                continue

            if text == '{"event":"start_speech"}':
                pcm_buffer.clear()
                print("[WEBSOCKET] ESP32 bat dau ghi am.", flush=True)
                continue

            if text != '{"event":"end_speech"}':
                continue

            pcm_size = len(pcm_buffer)
            print(f"[WEBSOCKET] ESP32 dung ghi am. PCM={pcm_size} bytes", flush=True)
            if pcm_size < 3200:
                pcm_buffer.clear()
                await websocket.send_text(json.dumps({"event": "tts_error", "message": "Audio qua ngan"}))
                continue

            wav_bytes = create_wav_bytes(bytes(pcm_buffer))
            pcm_buffer.clear()

            try:
                answer = await ask_gemini_audio(wav_bytes, safety_config)
                cleaned = clean_text_for_tts(answer)
                print(f"[BUN DAU] {cleaned}", flush=True)

                await websocket.send_text(json.dumps({"event": "tts_start"}))

                tts_started = time.monotonic()
                pcm, used_voice, tts_ms = await synthesize_tts(cleaned)
                tts_ready_ms = int((time.monotonic() - tts_started) * 1000)
                print(
                    f"[PERF] TTS ready | voice={used_voice} | first_audio_ready={tts_ready_ms} ms",
                    flush=True,
                )

                sent = await stream_pcm_to_esp(websocket, pcm)
                await websocket.send_text(json.dumps({"event": "tts_done"}))
                print(
                    f"[WEBSOCKET] Da gui xong audio | PCM={sent} bytes | "
                    f"TTS={tts_ms} ms | audio_duration={int(sent * 1000 / PCM_BYTES_PER_SECOND)} ms",
                    flush=True,
                )

            except WebSocketDisconnect:
                raise
            except Exception as exc:
                print(f"[SERVER ERROR] {exc}", flush=True)
                try:
                    await websocket.send_text(json.dumps({"event": "tts_error", "message": str(exc)[:300]}, ensure_ascii=False))
                except Exception:
                    pass

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
    except Exception as exc:
        print(f"[WEBSOCKET ERROR] {exc}", flush=True)
    finally:
        pcm_buffer.clear()


def create_wav_bytes(pcm_data: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return bio.getvalue()

import asyncio
import io
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Optional

# Keep Render Free / low-CPU memory footprint predictable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import httpx
import numpy as np
import soxr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from piper import PiperVoice, SynthesisConfig

try:
    from vietnormalizer import VietnameseNormalizer
except Exception:
    VietnameseNormalizer = None

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
# 2. LOCAL PIPER TTS
# ==============================================================================
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE
TTS_CHUNK_SIZE = 2048

TTS_MODEL_NAME = os.environ.get("PIPER_MODEL_NAME", "banmai")
TTS_MODEL_DIR = Path(os.environ.get("PIPER_MODEL_DIR", "models"))
TTS_MODEL_PATH = TTS_MODEL_DIR / f"{TTS_MODEL_NAME}.onnx"
TTS_CONFIG_PATH = TTS_MODEL_DIR / f"{TTS_MODEL_NAME}.onnx.json"

# Public model mirror used by the NGHI-TTS/Piper ecosystem.
# Ban Mai is a Vietnamese female/northern voice in upstream/service descriptions.
PIPER_MODEL_URL = os.environ.get(
    "PIPER_MODEL_URL",
    "https://huggingface.co/doof-ferb/nghitts-copy/resolve/main/piper-tts/banmai.onnx?download=true",
)
PIPER_CONFIG_URL = os.environ.get(
    "PIPER_CONFIG_URL",
    "https://huggingface.co/doof-ferb/nghitts-copy/resolve/main/piper-tts/config.json?download=true",
)
PIPER_SPEED = float(os.environ.get("PIPER_SPEED", "1.04"))
PIPER_VOLUME = float(os.environ.get("PIPER_VOLUME", "1.05"))
PIPER_NOISE_SCALE = float(os.environ.get("PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W_SCALE = float(os.environ.get("PIPER_NOISE_W_SCALE", "0.8"))

_piper_voice: Optional[PiperVoice] = None
_piper_init_lock = asyncio.Lock()
_normalizer = None


def _download_file_sync(url: str, path: Path, min_bytes: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0), follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) < min_bytes:
                    raise RuntimeError(f"File qua nho: {url} | content-length={content_length}")
                written = 0
                with tmp.open("wb") as f:
                    for chunk in response.iter_bytes(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                if written < min_bytes:
                    raise RuntimeError(f"Tai file chua du: {url} | bytes={written}")
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


PIPER_MIRRORS = [
    (
        "https://huggingface.co/doof-ferb/nghitts-copy/resolve/main/piper-tts/banmai.onnx?download=true",
        "https://huggingface.co/doof-ferb/nghitts-copy/resolve/main/piper-tts/config.json?download=true",
    ),
    (
        "https://huggingface.co/sannht/vi_voice/resolve/main/tts-model/banmai.onnx?download=true",
        "https://huggingface.co/sannht/vi_voice/resolve/main/tts-model/banmai.onnx.json?download=true",
    ),
]


async def ensure_piper_files() -> None:
    if TTS_MODEL_PATH.exists() and TTS_CONFIG_PATH.exists():
        return

    # Always download model + matching config as one pair. Never leave a half-pair
    # from a failed mirror attempt. The bundled config below is used when possible.
    TTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[TTS] Kiem tra Piper Ban Mai...", flush=True)

    # The package contains the known-good config.json as banmai.onnx.json, so the
    # normal startup path only needs the ~63.5 MB ONNX model.
    if TTS_CONFIG_PATH.exists():
        model_urls = [pair[0] for pair in PIPER_MIRRORS]
        for model_url in model_urls:
            try:
                print(f"[TTS] Tai Piper model: {model_url}", flush=True)
                await asyncio.to_thread(_download_file_sync, model_url, TTS_MODEL_PATH, 50 * 1024 * 1024)
                break
            except Exception as exc:
                print(f"[TTS] Mirror model loi: {str(exc)[:220]}", flush=True)
                try:
                    TTS_MODEL_PATH.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            raise RuntimeError("Khong tai duoc Piper model Ban Mai tu bat ky mirror nao")
    else:
        errors = []
        for model_url, config_url in PIPER_MIRRORS:
            temp_model = TTS_MODEL_PATH.with_suffix(TTS_MODEL_PATH.suffix + ".part")
            temp_config = TTS_CONFIG_PATH.with_suffix(TTS_CONFIG_PATH.suffix + ".part")
            try:
                print(f"[TTS] Tai cap Piper model+config tu mirror...", flush=True)
                await asyncio.to_thread(_download_file_sync, model_url, temp_model, 50 * 1024 * 1024)
                await asyncio.to_thread(_download_file_sync, config_url, temp_config, 1000)
                # Validate JSON before committing either file.
                json.loads(temp_config.read_text(encoding="utf-8"))
                temp_model.replace(TTS_MODEL_PATH)
                temp_config.replace(TTS_CONFIG_PATH)
                break
            except Exception as exc:
                errors.append(str(exc)[:180])
                temp_model.unlink(missing_ok=True)
                temp_config.unlink(missing_ok=True)
        else:
            raise RuntimeError("Khong tai duoc bo Piper model+config: " + " | ".join(errors))

    print(
        f"[TTS] Piper files OK | model={TTS_MODEL_PATH.name} | "
        f"model_size={TTS_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB | "
        f"config_size={TTS_CONFIG_PATH.stat().st_size / 1024:.1f} KB",
        flush=True,
    )


def _init_piper_sync() -> PiperVoice:
    return PiperVoice.load(str(TTS_MODEL_PATH), use_cuda=False)


async def get_piper_voice() -> PiperVoice:
    global _piper_voice, _normalizer
    if _piper_voice is not None:
        return _piper_voice
    async with _piper_init_lock:
        if _piper_voice is not None:
            return _piper_voice
        await ensure_piper_files()
        _piper_voice = await asyncio.to_thread(_init_piper_sync)
        if VietnameseNormalizer is not None:
            try:
                _normalizer = VietnameseNormalizer()
            except Exception:
                _normalizer = None
        print(
            f"[TTS] Piper ready | voice={TTS_MODEL_NAME} | sample_rate={_piper_voice.config.sample_rate}",
            flush=True,
        )
        return _piper_voice


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", "", text)
    text = re.sub(r"[*#_~>`]", "", text)
    text = text.replace("\u2013", " ").replace("\u2014", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if _normalizer is not None:
        try:
            text = _normalizer.normalize(text)
        except Exception:
            pass
    return text.strip(" ,;:-_\n\r\t")


def synthesize_full_pcm16_sync(text: str) -> bytes:
    voice = _piper_voice
    if voice is None:
        raise RuntimeError("Piper voice chua duoc khoi tao")

    syn = SynthesisConfig(
        volume=PIPER_VOLUME,
        length_scale=max(0.60, min(1.50, 1.0 / max(0.5, PIPER_SPEED))),
        noise_scale=PIPER_NOISE_SCALE,
        noise_w_scale=PIPER_NOISE_W_SCALE,
        normalize_audio=True,
    )

    pieces = []
    src_rate = int(voice.config.sample_rate)
    for chunk in voice.synthesize(text, syn_config=syn):
        pieces.append(chunk.audio_int16_bytes)

    if not pieces:
        raise RuntimeError("Piper khong sinh duoc audio")

    raw = b"".join(pieces)
    if src_rate == PCM_SAMPLE_RATE:
        return raw

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    resampled = soxr.resample(samples, src_rate, PCM_SAMPLE_RATE, quality="HQ")
    out = np.clip(resampled * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return out.tobytes()


async def synthesize_piper(text: str) -> bytes:
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        raise RuntimeError("Gemini returned empty TTS text")
    await get_piper_voice()
    return await asyncio.to_thread(synthesize_full_pcm16_sync, cleaned)


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
        "tts_provider": "piper-local-bundled-config",
        "tts_voice": TTS_MODEL_NAME,
        "tts_output": "PCM16 16kHz mono",
    }


# ==============================================================================
# 4. GEMINI PROCESSING
# ==============================================================================
async def ask_gemini_audio(wav_bytes: bytes, safety_config) -> str:
    global CURRENT_KEY_INDEX

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

            print(
                f"[GEMINI] Ket thuc Key #{key_idx + 1} | finish={finish_reason} | message={finish_message!r}",
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

                started = time.monotonic()
                pcm = await synthesize_piper(cleaned)
                tts_ms = int((time.monotonic() - started) * 1000)
                print(f"[TTS] Piper da sinh audio | PCM={len(pcm)} bytes | synth={tts_ms} ms", flush=True)

                sent = await stream_pcm_to_esp(websocket, pcm)
                await websocket.send_text(json.dumps({"event": "tts_done"}))
                print(f"[WEBSOCKET] Da gui xong audio | PCM={sent} bytes", flush=True)

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

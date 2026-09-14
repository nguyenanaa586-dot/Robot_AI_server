import asyncio
import io
import json
import os
import queue
import re
import threading
import time
import wave
from typing import Optional

import numpy as np
import soxr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

try:
    from vieneu import Vieneu
except ImportError as exc:  # pragma: no cover
    Vieneu = None
    _VIENEU_IMPORT_ERROR = exc
else:
    _VIENEU_IMPORT_ERROR = None

app = FastAPI()

# ============================================================================
# 1. GEMINI
# ============================================================================
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip(' "\'\t\r\n') for k in RAW_KEYS.split(",") if k.strip(' "\'\t\r\n')]

CURRENT_KEY_INDEX = 0
KEY_STATUS = ["active"] * len(API_KEYS)
KEY_FAILURE_REASON = [None] * len(API_KEYS)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1024"))

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
    return genai.Client(api_key=API_KEYS[key_index]) if API_KEYS else None


def _error_code(exc) -> Optional[int]:
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if value is not None:
            m = re.search(r"\b(\d{3})\b", str(value))
            if m:
                return int(m.group(1))
    m = re.search(r"\b(401|403|429|500|502|503|504)\b", str(exc))
    return int(m.group(1)) if m else None


def classify_gemini_error(exc) -> str:
    code = _error_code(exc)
    text = str(exc).lower()
    if code in (401, 403, 429):
        return "rotate"
    if any(x in text for x in (
        "api key not valid", "api_key_invalid", "invalid api key", "invalid_api_key",
        "unauthenticated", "permission denied", "permission_denied", "quota exceeded",
        "quota_exceeded", "resource_exhausted", "rate limit", "ratelimit",
    )):
        return "rotate"
    if code in (500, 502, 503, 504) or any(x in text for x in (
        "service unavailable", "internal server error", "bad gateway", "gateway timeout",
        "deadline exceeded", "timeout", "timed out", "connection reset", "temporarily unavailable",
    )):
        return "transient"
    return "fatal"


def mark_key_disabled(index: int, reason: str) -> None:
    KEY_STATUS[index] = "disabled"
    KEY_FAILURE_REASON[index] = reason[:240]


def next_active_key_after(index: int) -> Optional[int]:
    for i in range(index + 1, len(API_KEYS)):
        if KEY_STATUS[i] == "active":
            return i
    return None


# ============================================================================
# 2. AUDIO / TTS
# ============================================================================
INPUT_RATE = 16_000
TTS_NATIVE_RATE = 24_000
OUTPUT_RATE = 16_000
TTS_CHUNK_SIZE = 2048

VIENEU_BACKBONE_REPO = os.environ.get("VIENEU_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF")
VIENEU_BACKBONE_FILENAME = os.environ.get("VIENEU_BACKBONE_FILENAME", "vieneu-tts-v2-turbo.gguf")
VIENEU_DECODER_REPO = os.environ.get("VIENEU_DECODER_REPO", "pnnbao-ump/VieNeu-Codec")
VIENEU_DECODER_FILENAME = os.environ.get("VIENEU_DECODER_FILENAME", "vieneu_decoder.onnx")
VIENEU_VOICE_NAME = os.environ.get("VIENEU_VOICE", "Hương")
VIENEU_DEVICE = os.environ.get("VIENEU_DEVICE", "cpu")
VIENEU_TEMPERATURE = float(os.environ.get("VIENEU_TEMPERATURE", "0.4"))
VIENEU_TOP_K = int(os.environ.get("VIENEU_TOP_K", "50"))
VIENEU_MAX_CHARS = int(os.environ.get("VIENEU_MAX_CHARS", "256"))

TTS_INSTANCE = None
TTS_VOICE_DATA = None
TTS_LOAD_LOCK = threading.Lock()
TTS_SYNTH_LOCK = asyncio.Lock()
TTS_LOAD_ERROR: Optional[str] = None


def create_wav_bytes(pcm_data: bytes, sample_rate: int = INPUT_RATE) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return out.getvalue()


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_~>`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:-_\n\r\t")


def safe_get_chunk_text(chunk) -> str:
    try:
        return chunk.text or ""
    except Exception:
        return ""


def _load_vieneu_sync():
    global TTS_INSTANCE, TTS_VOICE_DATA, TTS_LOAD_ERROR
    if TTS_INSTANCE is not None:
        return TTS_INSTANCE
    if Vieneu is None:
        raise RuntimeError(f"Khong import duoc vieneu: {_VIENEU_IMPORT_ERROR}")

    with TTS_LOAD_LOCK:
        if TTS_INSTANCE is not None:
            return TTS_INSTANCE
        print("[TTS] Dang tai VieNeu-TTS v2 Turbo GGUF...", flush=True)
        print(f"[TTS] Model={VIENEU_BACKBONE_REPO}/{VIENEU_BACKBONE_FILENAME}", flush=True)
        t0 = time.perf_counter()
        tts = Vieneu(
            mode="turbo",
            backbone_repo=VIENEU_BACKBONE_REPO,
            backbone_filename=VIENEU_BACKBONE_FILENAME,
            decoder_repo=VIENEU_DECODER_REPO,
            decoder_filename=VIENEU_DECODER_FILENAME,
            device=VIENEU_DEVICE,
        )
        voices = tts.list_preset_voices()
        selected_id = None
        for label, voice_id in voices:
            if str(label).strip().casefold() == VIENEU_VOICE_NAME.casefold():
                selected_id = voice_id
                break
            if VIENEU_VOICE_NAME.casefold() in str(label).casefold():
                selected_id = voice_id
        if selected_id is None:
            available = ", ".join(str(x[0]) for x in voices[:20])
            raise RuntimeError(f"Khong tim thay voice '{VIENEU_VOICE_NAME}'. Voices: {available}")

        TTS_INSTANCE = tts
        TTS_VOICE_DATA = tts.get_preset_voice(selected_id)
        TTS_LOAD_ERROR = None
        print(f"[TTS] VieNeu ready | voice={VIENEU_VOICE_NAME} | load={time.perf_counter()-t0:.1f}s", flush=True)
        return TTS_INSTANCE


def _resample_f32_24k_to_16k(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    return np.asarray(soxr.resample(samples, TTS_NATIVE_RATE, OUTPUT_RATE), dtype=np.float32)


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


def _vieneu_stream_worker(text: str, out_q: queue.Queue):
    try:
        tts = _load_vieneu_sync()
        for chunk in tts.infer_stream(
            text,
            voice=TTS_VOICE_DATA,
            temperature=VIENEU_TEMPERATURE,
            top_k=VIENEU_TOP_K,
            max_chars=VIENEU_MAX_CHARS,
        ):
            if chunk is None:
                continue
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            pcm = _float_to_pcm16(_resample_f32_24k_to_16k(arr))
            if pcm:
                out_q.put(pcm)
    except Exception as exc:
        out_q.put(exc)
    finally:
        out_q.put(None)


async def stream_vieneu_tts(websocket: WebSocket, text: str) -> int:
    q: queue.Queue = queue.Queue(maxsize=8)
    worker = threading.Thread(target=_vieneu_stream_worker, args=(text, q), daemon=True)
    worker.start()

    total = 0
    while True:
        item = await asyncio.to_thread(q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        for offset in range(0, len(item), TTS_CHUNK_SIZE):
            chunk = item[offset:offset + TTS_CHUNK_SIZE]
            await websocket.send_bytes(chunk)
            total += len(chunk)
            # 2048 bytes = 64 ms at 16kHz mono PCM16. Pace at real-time.
            await asyncio.sleep(len(chunk) / 32_000.0)

    return total


# ============================================================================
# 3. HELPERS
# ============================================================================
async def send_event(websocket: WebSocket, event: str, message: Optional[str] = None):
    payload = {"event": event}
    if message is not None:
        payload["message"] = message
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


# ============================================================================
# 4. ROOT / HEALTH
# ============================================================================
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "Bun Dau Robot AI",
        "tts_provider": "VieNeu-TTS-v2-Turbo-GGUF",
        "tts_voice": VIENEU_VOICE_NAME,
        "tts_sample_rate": OUTPUT_RATE,
    }


# ============================================================================
# 5. WEBSOCKET
# ============================================================================
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
                break

            if message.get("bytes"):
                pcm_buffer.extend(message["bytes"])
                continue

            msg_text = (message.get("text") or "").strip()
            if not msg_text:
                continue

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

            safety_config = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ]

            if not API_KEYS:
                await send_event(websocket, "tts_error", "No API Keys")
                continue

            if CURRENT_KEY_INDEX >= len(API_KEYS):
                CURRENT_KEY_INDEX = 0

            key_idx = CURRENT_KEY_INDEX
            full_response_text = ""
            gemini_ok = False

            while key_idx is not None and key_idx < len(API_KEYS):
                if KEY_STATUS[key_idx] != "active":
                    key_idx = next_active_key_after(key_idx)
                    continue

                print(f"[GEMINI] Dang dung Key #{key_idx + 1}", flush=True)
                client = get_genai_client(key_idx)
                try:
                    stream = await client.aio.models.generate_content_stream(
                        model=MODEL_NAME,
                        contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                            temperature=0.7,
                            safety_settings=safety_config,
                        ),
                    )
                    parts = []
                    finish_reason = None
                    finish_message = None
                    usage = None
                    async for chunk in stream:
                        txt = safe_get_chunk_text(chunk)
                        if txt:
                            parts.append(txt)
                        try:
                            if chunk.candidates:
                                c = chunk.candidates[0]
                                if c.finish_reason is not None:
                                    finish_reason = str(c.finish_reason)
                                if c.finish_message:
                                    finish_message = str(c.finish_message)
                        except Exception:
                            pass
                        try:
                            if chunk.usage_metadata:
                                usage = chunk.usage_metadata
                        except Exception:
                            pass

                    full_response_text = "".join(parts).strip()
                    finish_upper = str(finish_reason or "UNKNOWN").upper()
                    print(f"[GEMINI] Ket thuc Key #{key_idx + 1} | finish={finish_reason} | message={finish_message!r}", flush=True)
                    if usage:
                        print(
                            f"[GEMINI] Usage | prompt={getattr(usage, 'prompt_token_count', None)} | output={getattr(usage, 'candidates_token_count', None)} | total={getattr(usage, 'total_token_count', None)}",
                            flush=True,
                        )

                    if any(x in finish_upper for x in ("MAX_TOKENS", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT")):
                        print(f"[GEMINI] Response khong hoan chinh: {finish_reason}", flush=True)
                        await send_event(websocket, "tts_error", f"Gemini response incomplete: {finish_reason}")
                        gemini_ok = False
                        break

                    if full_response_text:
                        CURRENT_KEY_INDEX = key_idx
                        gemini_ok = True
                        print(f"[GEMINI] Ghi nho Key #{key_idx + 1}.", flush=True)
                    break

                except Exception as exc:
                    kind = classify_gemini_error(exc)
                    code = _error_code(exc)
                    text = str(exc).replace("\n", " ")[:220]
                    if kind == "rotate":
                        mark_key_disabled(key_idx, text)
                        next_idx = next_active_key_after(key_idx)
                        if next_idx is None:
                            print("[GEMINI] Khong con key phia sau.", flush=True)
                        else:
                            print(f"[GEMINI] Key #{key_idx + 1} khong dung duoc (HTTP {code or 'key'}) -> Key #{next_idx + 1}", flush=True)
                        key_idx = next_idx
                        continue
                    if kind == "transient":
                        print(f"[GEMINI] Key #{key_idx + 1} loi tam thoi -> thu lai.", flush=True)
                        await asyncio.sleep(0.6)
                        try:
                            stream = await client.aio.models.generate_content_stream(
                                model=MODEL_NAME,
                                contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                                    temperature=0.7,
                                    safety_settings=safety_config,
                                ),
                            )
                            retry_parts = []
                            retry_finish = None
                            async for chunk in stream:
                                t = safe_get_chunk_text(chunk)
                                if t:
                                    retry_parts.append(t)
                                try:
                                    if chunk.candidates and chunk.candidates[0].finish_reason is not None:
                                        retry_finish = str(chunk.candidates[0].finish_reason)
                                except Exception:
                                    pass
                            full_response_text = "".join(retry_parts).strip()
                            if full_response_text and (retry_finish is None or "STOP" in retry_finish.upper()):
                                CURRENT_KEY_INDEX = key_idx
                                gemini_ok = True
                                print(f"[GEMINI] Retry thanh cong voi Key #{key_idx + 1}.", flush=True)
                            break
                        except Exception as retry_exc:
                            print(f"[GEMINI] Retry that bai: {str(retry_exc)[:180]}", flush=True)
                            gemini_ok = False
                            break
                    print(f"[GEMINI] Loi fatal Key #{key_idx + 1}: {text}", flush=True)
                    break

            if not gemini_ok:
                try:
                    await send_event(websocket, "tts_error", "Gemini request failed")
                except Exception:
                    pass
                continue

            cleaned = clean_text_for_tts(full_response_text)
            if not cleaned:
                await send_event(websocket, "tts_error", "Gemini returned empty response")
                continue

            print(f"[BUN DAU] {cleaned}", flush=True)

            async with TTS_SYNTH_LOCK:
                try:
                    await send_event(websocket, "tts_start")
                    t0 = time.perf_counter()
                    total_pcm = await stream_vieneu_tts(websocket, cleaned)
                    elapsed = time.perf_counter() - t0
                    if total_pcm <= 0:
                        raise RuntimeError("VieNeu khong tao ra audio")
                    print(f"[TTS] VieNeu voice={VIENEU_VOICE_NAME} | PCM={total_pcm} bytes | synth+stream={elapsed:.2f}s", flush=True)
                    await send_event(websocket, "tts_done")
                except WebSocketDisconnect:
                    raise
                except Exception as exc:
                    print(f"[TTS ERROR] {exc}", flush=True)
                    try:
                        await send_event(websocket, "tts_error", "VieNeu TTS failed")
                    except Exception:
                        pass

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
    except Exception as exc:
        print(f"[WEBSOCKET ERROR] {exc}", flush=True)
    finally:
        pcm_buffer.clear()

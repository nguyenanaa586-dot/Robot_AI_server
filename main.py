import asyncio
import io
import json
import os
import re
import threading
from typing import Optional

import numpy as np
import soxr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

# ============================================================================
# GEMINI
# ============================================================================
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip(" \\\"'\\t\\r\\n") for k in RAW_KEYS.split(",") if k.strip(" \\\"'\\t\\r\\n")]

CURRENT_KEY_INDEX = 0
KEY_STATUS = ["active"] * len(API_KEYS)
KEY_FAILURE_REASON = [None] * len(API_KEYS)
KEY_STATE_LOCK = threading.Lock()

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
        "quota_exceeded", "resource_exhausted", "rate limit", "ratelimit"
    )):
        return "rotate"
    if code in (500, 502, 503, 504) or any(x in text for x in (
        "service unavailable", "internal server error", "bad gateway",
        "gateway timeout", "deadline exceeded", "timeout", "timed out",
        "connection reset", "temporarily unavailable"
    )):
        return "transient"
    return "fatal"


def next_active_key_after(index: int) -> Optional[int]:
    for i in range(index + 1, len(API_KEYS)):
        if KEY_STATUS[i] == "active":
            return i
    return None


def mark_key_disabled(index: int, reason: str) -> None:
    KEY_STATUS[index] = "disabled"
    KEY_FAILURE_REASON[index] = reason[:240]


def key_status_summary() -> str:
    return ", ".join(f"#{i}= {s.upper()}".replace("= ", "=") for i, s in enumerate(KEY_STATUS, 1))

# ============================================================================
# VIE NEU TTS v2 TURBO - LOCAL / NO API CREDITS
# ============================================================================
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_BYTES_PER_SAMPLE
TTS_CHUNK_SIZE = 4096

VIENEU_REPO = os.environ.get("VIENEU_REPO", "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF")
VIENEU_BACKBONE = os.environ.get("VIENEU_BACKBONE", "vieneu-tts-v2-turbo.gguf")
VIENEU_VOICE = os.environ.get("VIENEU_VOICE", "Hương")
VIENEU_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
VIENEU_MAX_CHARS = max(64, int(os.environ.get("VIENEU_MAX_CHARS", "256")))
VIENEU_TEMPERATURE = float(os.environ.get("VIENEU_TEMPERATURE", "0.4"))
VIENEU_TOP_K = int(os.environ.get("VIENEU_TOP_K", "50"))

vieneu_tts = None
vieneu_voice_data = None
vieneu_lock = asyncio.Lock()


def load_vieneu_once() -> None:
    global vieneu_tts, vieneu_voice_data
    if vieneu_tts is not None and vieneu_voice_data is not None:
        return

    from vieneu import Vieneu

    print("[TTS] Dang khoi tao VieNeu-TTS v2 Turbo local...", flush=True)
    vieneu_tts = Vieneu(
        mode="turbo",
        backbone_repo=VIENEU_REPO,
        backbone_filename=VIENEU_BACKBONE,
        device="cpu",
        hf_token=VIENEU_HF_TOKEN,
    )

    voices = vieneu_tts.list_preset_voices()
    ids = [voice_id for _, voice_id in voices]
    print(f"[TTS] Voice presets: {', '.join(ids)}", flush=True)
    if VIENEU_VOICE not in ids:
        raise RuntimeError(
            f"Khong tim thay voice '{VIENEU_VOICE}'. Voice kha dung: {', '.join(ids)}"
        )
    vieneu_voice_data = vieneu_tts.get_preset_voice(VIENEU_VOICE)
    print(f"[TTS] VieNeu ready | voice={VIENEU_VOICE} | sample_rate=24000", flush=True)


def float_audio_to_pcm16_16k(audio_f32: np.ndarray) -> bytes:
    audio = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""
    # VieNeu v2 output is 24 kHz; resample only once at the server boundary.
    if audio.size:
        audio_16k = soxr.resample(audio, 24000, PCM_SAMPLE_RATE).astype(np.float32, copy=False)
    else:
        audio_16k = audio
    pcm = np.clip(audio_16k * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return pcm.tobytes()


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}", "", text)
    text = re.sub(r"[*#_~>`]", "", text)
    text = text.replace("—", " ").replace("–", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:-_\n\r\t")


def _vieneu_stream_generator(text: str):
    load_vieneu_once()
    return vieneu_tts.infer_stream(
        text=text,
        voice=vieneu_voice_data,
        max_chars=VIENEU_MAX_CHARS,
        temperature=VIENEU_TEMPERATURE,
        top_k=VIENEU_TOP_K,
        apply_watermark=True,
    )


async def stream_vieneu_tts_to_esp(websocket: WebSocket, text: str) -> int:
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        raise RuntimeError("Gemini tra ve text rong")

    async with vieneu_lock:
        generator = await asyncio.to_thread(_vieneu_stream_generator, cleaned)
        total_pcm = 0
        first_audio = True
        next_deadline = asyncio.get_running_loop().time()

        _END = object()

        def next_or_end(gen):
            try:
                return next(gen)
            except StopIteration:
                return _END

        while True:
            audio_chunk = await asyncio.to_thread(next_or_end, generator)
            if audio_chunk is _END:
                break

            pcm16 = float_audio_to_pcm16_16k(audio_chunk)
            if not pcm16:
                continue

            if first_audio:
                print(f"[TTS] Bat dau stream PCM | voice={VIENEU_VOICE}", flush=True)
                first_audio = False

            pending = memoryview(pcm16)
            while pending:
                part = pending[:TTS_CHUNK_SIZE]
                await websocket.send_bytes(part.tobytes())
                total_pcm += len(part)
                pending = pending[len(part):]

                # Real-time pacing at 16 kHz / 16-bit / mono.
                next_deadline += len(part) / PCM_BYTES_PER_SECOND
                delay = next_deadline - asyncio.get_running_loop().time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_deadline = asyncio.get_running_loop().time()

        if total_pcm <= 0:
            raise RuntimeError("VieNeu khong tao ra audio")
        return total_pcm


# ============================================================================
# HELPERS
# ============================================================================
async def send_event(websocket: WebSocket, event: str, message: Optional[str] = None):
    payload = {"event": event}
    if message:
        payload["message"] = message
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


def safe_get_chunk_text(chunk) -> str:
    try:
        return chunk.text or ""
    except Exception:
        return ""


async def generate_with_sticky_key(wav_bytes: bytes) -> str:
    global CURRENT_KEY_INDEX

    if not API_KEYS:
        raise RuntimeError("Khong co GEMINI_API_KEY")

    key_idx = CURRENT_KEY_INDEX
    safety_config = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

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
            stream = await client.aio.models.generate_content_stream(
                model=MODEL_NAME,
                contents=[genai.types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                    safety_settings=safety_config,
                ),
            )

            parts = []
            finish_reason = "UNKNOWN"
            finish_message = None
            usage = None

            async for chunk in stream:
                txt = safe_get_chunk_text(chunk)
                if txt:
                    parts.append(txt)
                try:
                    candidates = getattr(chunk, "candidates", None)
                    if candidates:
                        c = candidates[0]
                        if getattr(c, "finish_reason", None) is not None:
                            finish_reason = str(c.finish_reason)
                        if getattr(c, "finish_message", None):
                            finish_message = str(c.finish_message)
                except Exception:
                    pass
                try:
                    if getattr(chunk, "usage_metadata", None) is not None:
                        usage = chunk.usage_metadata
                except Exception:
                    pass

            response = "".join(parts).strip()
            print(f"[GEMINI] Ket thuc Key #{key_idx + 1} | finish={finish_reason} | message={finish_message!r}", flush=True)
            if usage is not None:
                print(
                    f"[GEMINI] Usage | prompt={getattr(usage,'prompt_token_count',None)} | "
                    f"output={getattr(usage,'candidates_token_count',None)} | "
                    f"total={getattr(usage,'total_token_count',None)}",
                    flush=True,
                )

            finish_upper = finish_reason.upper()
            bad_finish = any(x in finish_upper for x in (
                "MAX_TOKENS", "INCOMPLETE", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"
            ))
            if bad_finish:
                raise RuntimeError(f"Gemini response khong hoan chinh: {finish_reason}")
            if not response:
                raise RuntimeError("Gemini tra response rong")

            CURRENT_KEY_INDEX = key_idx
            print(f"[GEMINI] Ghi nho Key #{key_idx + 1}.", flush=True)
            return response

        except Exception as exc:
            kind = classify_gemini_error(exc)
            code = _error_code(exc)
            detail = str(exc).replace("\n", " ")[:240]

            if kind == "rotate":
                mark_key_disabled(key_idx, detail)
                nxt = next_active_key_after(key_idx)
                if nxt is None:
                    raise RuntimeError(f"Key #{key_idx + 1} loi va khong con key active phia sau: {detail}")
                print(f"[GEMINI] Key #{key_idx + 1} loi HTTP {code or ''} -> chuyen Key #{nxt + 1}", flush=True)
                key_idx = nxt
                continue

            if kind == "transient":
                print(f"[GEMINI] Key #{key_idx + 1} loi tam thoi -> retry 1 lan", flush=True)
                await asyncio.sleep(0.6)
                try:
                    retry = await client.aio.models.generate_content_stream(
                        model=MODEL_NAME,
                        contents=[genai.types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
                        config=genai.types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                            safety_settings=safety_config,
                        ),
                    )
                    retry_parts = []
                    retry_finish = "UNKNOWN"
                    async for chunk in retry:
                        txt = safe_get_chunk_text(chunk)
                        if txt:
                            retry_parts.append(txt)
                        try:
                            candidates = getattr(chunk, "candidates", None)
                            if candidates and getattr(candidates[0], "finish_reason", None) is not None:
                                retry_finish = str(candidates[0].finish_reason)
                        except Exception:
                            pass
                    retry_text = "".join(retry_parts).strip()
                    if retry_text and "STOP" in retry_finish.upper():
                        CURRENT_KEY_INDEX = key_idx
                        print(f"[GEMINI] Retry thanh cong Key #{key_idx + 1}.", flush=True)
                        return retry_text
                except Exception as retry_exc:
                    print(f"[GEMINI] Retry that bai: {str(retry_exc)[:180]}", flush=True)

            raise RuntimeError(f"Gemini loi voi Key #{key_idx + 1}: {detail}")

    raise RuntimeError("Khong con Gemini key kha dung")


# ============================================================================
# HTTP
# ============================================================================
@app.get("/")
def root():
    return {
        "status": "Robot Bun Dau WebSocket Server OK",
        "gemini_keys": len(API_KEYS),
        "current_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "tts_provider": "VieNeu-TTS-v2-Turbo-local",
        "tts_voice": VIENEU_VOICE,
        "tts_output": "PCM16 16kHz mono",
    }


# ============================================================================
# WEBSOCKET
# ============================================================================
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
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
                await send_event(websocket, "tts_error", "Audio qua ngan")
                continue

            wav_io = io.BytesIO()
            import wave
            with wave.open(wav_io, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(pcm_buffer))
            wav_bytes = wav_io.getvalue()
            pcm_buffer.clear()

            try:
                response_text = await generate_with_sticky_key(wav_bytes)
                response_text = clean_text_for_tts(response_text)
                print(f"[BUN DAU] {response_text}", flush=True)

                await send_event(websocket, "tts_start")
                total_pcm = await stream_vieneu_tts_to_esp(websocket, response_text)
                await send_event(websocket, "tts_done")
                print(f"[WEBSOCKET] Da gui xong TTS | PCM={total_pcm} bytes", flush=True)

            except WebSocketDisconnect:
                raise
            except Exception as exc:
                print(f"[SERVER ERROR] {exc}", flush=True)
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

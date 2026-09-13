import asyncio
import io
import json
import os
import re
import time
import wave
from typing import Optional

import edge_tts
import miniaudio
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
# - Server startup always begins scanning from Key #1.
# - Once a working key is found, that key is remembered.
# - Every subsequent request starts directly from the remembered key.
# - If the remembered key later fails with 401/403/429/key/quota errors,
#   move forward to the next active key only. Never go backwards.
CURRENT_KEY_INDEX = 0
KEY_STATUS = ["active"] * len(API_KEYS)
KEY_FAILURE_REASON = [None] * len(API_KEYS)

MODEL_NAME = "gemini-3.6-flash"

# Current Edge-TTS Vietnamese standard voice.
# Change this one line when we deliberately choose another supported voice/provider.
TTS_VOICE = os.environ.get("TTS_VOICE", "vi-VN-HoaiMyNeural")
TTS_FALLBACK_VOICES = [
    v.strip()
    for v in os.environ.get("TTS_FALLBACK_VOICES", "vi-VN-NamMinhNeural").split(",")
    if v.strip() and v.strip() != TTS_VOICE
]
TTS_RATE = os.environ.get("TTS_RATE", "+10%")
TTS_VOLUME = os.environ.get("TTS_VOLUME", "+0%")
TTS_TIMEOUT_SECONDS = 20

PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_BYTES_PER_SAMPLE
TTS_CHUNK_SIZE = 2048

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
            try:
                text = str(value)
                match = re.search(r"\b(\d{3})\b", text)
                if match:
                    return int(match.group(1))
            except Exception:
                pass

    text = str(exc)
    match = re.search(r"\b(401|403|429|500|502|503|504)\b", text)
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
    parts = []
    for idx, status in enumerate(KEY_STATUS, start=1):
        parts.append(f"#{idx}={status.upper()}")
    return ", ".join(parts)


# ==============================================================================
# 2. AUDIO / TEXT
# ==============================================================================
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


async def synthesize_text_to_pcm(text: str, voice: str, target_sample_rate: int = PCM_SAMPLE_RATE) -> Optional[bytes]:
    clean_txt = clean_text_for_tts(text)
    if not clean_txt or not re.search(r"\w", clean_txt):
        return b""

    communicate = edge_tts.Communicate(
        clean_txt,
        voice=voice,
        rate=TTS_RATE,
        volume=TTS_VOLUME,
    )

    mp3_buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.extend(chunk["data"])

    if not mp3_buffer:
        return None

    def decode_audio() -> bytes:
        decoded = miniaudio.decode(
            bytes(mp3_buffer),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=PCM_CHANNELS,
            sample_rate=target_sample_rate,
        )
        return decoded.samples.tobytes()

    pcm_bytes = await asyncio.to_thread(decode_audio)
    return pcm_bytes or None


async def stream_pcm_realtime(websocket: WebSocket, pcm_bytes: bytes) -> None:
    if not pcm_bytes:
        return

    next_deadline = time.monotonic()
    for offset in range(0, len(pcm_bytes), TTS_CHUNK_SIZE):
        chunk = pcm_bytes[offset:offset + TTS_CHUNK_SIZE]
        await websocket.send_bytes(chunk)

        chunk_duration = len(chunk) / PCM_BYTES_PER_SECOND
        next_deadline += chunk_duration
        delay = next_deadline - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


async def synthesize_whole_response_with_retry(text: str) -> tuple[Optional[bytes], Optional[str]]:
    """
    TTS cho toàn bộ response trong một job để tránh khoảng nghỉ giữa các câu.
    Khi Edge-TTS không trả audio, thử lại cùng voice một lần, sau đó chuyển sang
    fallback voice. Điều này xử lý tốt hơn lỗi NoAudioReceived mang tính tạm thời.
    """
    clean_txt = clean_text_for_tts(text)
    if not clean_txt:
        return b"", None

    voices = [TTS_VOICE] + TTS_FALLBACK_VOICES

    for voice_index, voice in enumerate(voices):
        for attempt in range(1, 3):
            try:
                print(
                    f"[TTS] Voice={voice} | lan thu {attempt}/{2}: {clean_txt!r}",
                    flush=True,
                )
                pcm_bytes = await asyncio.wait_for(
                    synthesize_text_to_pcm(clean_txt, voice),
                    timeout=TTS_TIMEOUT_SECONDS,
                )
                if pcm_bytes:
                    print(
                        f"[TTS] Thanh cong | voice={voice} | PCM={len(pcm_bytes)} bytes",
                        flush=True,
                    )
                    return pcm_bytes, voice
                raise RuntimeError("Edge-TTS khong tra ve audio")
            except Exception as exc:
                print(
                    f"[EDGE-TTS Error] voice={voice} | lan {attempt}: {exc}",
                    flush=True,
                )
                if attempt < 2:
                    await asyncio.sleep(0.8)

        if voice_index < len(voices) - 1:
            next_voice = voices[voice_index + 1]
            print(
                f"[TTS] Chuyen fallback voice: {next_voice}",
                flush=True,
            )

    return None, None


# ==============================================================================
# 3. HTTP
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu WebSocket Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "tts_voice": TTS_VOICE,
        "tts_fallback_voices": TTS_FALLBACK_VOICES,
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
            try:
                message = await websocket.receive()
            except RuntimeError:
                print("[WEBSOCKET] Socket da dong.", flush=True)
                break

            if message.get("type") == "websocket.disconnect":
                print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
                break

            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])
                continue

            if "text" not in message or not message["text"]:
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

            total_keys = len(API_KEYS)
            if total_keys == 0:
                await send_event(websocket, "tts_error", "No API Keys")
                continue

            # ------------------------------------------------------------------
            # GEMINI: sticky key, scan only when necessary
            # ------------------------------------------------------------------
            if CURRENT_KEY_INDEX >= total_keys:
                CURRENT_KEY_INDEX = 0

            key_idx = CURRENT_KEY_INDEX
            full_response_text = ""
            gemini_ok = False

            while key_idx is not None and key_idx < total_keys:
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
                            temperature=0.7,
                            safety_settings=safety_config,
                        ),
                    )

                    response_text_parts = []
                    last_finish_reason = None
                    last_finish_message = None
                    last_usage = None

                    async for chunk in gemini_stream:
                        txt = safe_get_chunk_text(chunk)
                        if txt:
                            response_text_parts.append(txt)

                        try:
                            if getattr(chunk, "candidates", None):
                                candidate = chunk.candidates[0]
                                if getattr(candidate, "finish_reason", None) is not None:
                                    last_finish_reason = str(candidate.finish_reason)
                                if getattr(candidate, "finish_message", None):
                                    last_finish_message = str(candidate.finish_message)
                        except Exception:
                            pass

                        try:
                            if getattr(chunk, "usage_metadata", None) is not None:
                                last_usage = chunk.usage_metadata
                        except Exception:
                            pass

                    full_response_text = "".join(response_text_parts).strip()
                    finish_reason = last_finish_reason or "UNKNOWN"
                    finish_upper = finish_reason.upper()

                    print(
                        f"[GEMINI] Ket thuc Key #{key_idx + 1} | finish={finish_reason} | message={last_finish_message!r}",
                        flush=True,
                    )

                    if last_usage is not None:
                        print(
                            "[GEMINI] Usage | "
                            f"prompt={getattr(last_usage, 'prompt_token_count', None)} | "
                            f"output={getattr(last_usage, 'candidates_token_count', None)} | "
                            f"total={getattr(last_usage, 'total_token_count', None)}",
                            flush=True,
                        )

                    if any(marker in finish_upper for marker in ("MAX_TOKENS", "INCOMPLETE", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT")):
                        print(
                            f"[GEMINI] Key #{key_idx + 1} tra response khong hoan chinh ({finish_reason}).",
                            flush=True,
                        )
                        full_response_text = ""
                        await send_event(websocket, "tts_error", f"Gemini response incomplete: {finish_reason}")
                        gemini_ok = False
                        break

                    CURRENT_KEY_INDEX = key_idx
                    gemini_ok = bool(full_response_text)
                    if gemini_ok:
                        print(f"[GEMINI] Ghi nho Key #{key_idx + 1} cho cac luot tiep theo.", flush=True)
                    break

                except Exception as api_err:
                    kind = classify_gemini_error(api_err)
                    code = _error_code(api_err)
                    code_text = f"HTTP {code}" if code else kind.upper()
                    error_text = str(api_err).replace("\n", " ")[:220]

                    if kind == "rotate":
                        mark_key_disabled(key_idx, error_text)
                        next_idx = next_active_key_after(key_idx)
                        if next_idx is not None:
                            print(
                                f"[GEMINI] Key #{key_idx + 1} khong dung duoc ({code_text}) -> chuyen sang Key #{next_idx + 1}.",
                                flush=True,
                            )
                        else:
                            print("[GEMINI] Khong con key hoat dong phia sau.", flush=True)
                        key_idx = next_idx
                        continue

                    if kind == "transient":
                        # Retry same sticky key once; do not disable a healthy key for a
                        # temporary backend/network failure.
                        print(
                            f"[GEMINI] Key #{key_idx + 1} gap loi tam thoi ({code_text}). Thu lai cung key.",
                            flush=True,
                        )
                        try:
                            await asyncio.sleep(0.6)
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
                                    temperature=0.7,
                                    safety_settings=safety_config,
                                ),
                            )
                            retry_parts = []
                            retry_finish = None
                            async for chunk in gemini_stream:
                                txt = safe_get_chunk_text(chunk)
                                if txt:
                                    retry_parts.append(txt)
                                try:
                                    if getattr(chunk, "candidates", None):
                                        candidate = chunk.candidates[0]
                                        if getattr(candidate, "finish_reason", None) is not None:
                                            retry_finish = str(candidate.finish_reason)
                                except Exception:
                                    pass
                            full_response_text = "".join(retry_parts).strip()
                            if full_response_text and (retry_finish is None or "STOP" in str(retry_finish).upper()):
                                CURRENT_KEY_INDEX = key_idx
                                gemini_ok = True
                                print(f"[GEMINI] Retry thanh cong voi Key #{key_idx + 1}; tiep tuc ghi nho key nay.", flush=True)
                                break
                        except Exception as retry_err:
                            print(
                                f"[GEMINI] Retry cung Key #{key_idx + 1} that bai: {str(retry_err)[:180]}",
                                flush=True,
                            )
                        gemini_ok = False
                        break

                    print(
                        f"[GEMINI] Loi khong the khoi phuc voi Key #{key_idx + 1}: {error_text}",
                        flush=True,
                    )
                    gemini_ok = False
                    break

            if not gemini_ok:
                await send_event(websocket, "tts_error", "Gemini request failed")
                continue

            # ------------------------------------------------------------------
            # TTS: whole response in ONE Edge-TTS job => no gap between sentences
            # ------------------------------------------------------------------
            try:
                cleaned_text = clean_text_for_tts(full_response_text)
                if not cleaned_text:
                    await send_event(websocket, "tts_error", "Gemini returned empty response")
                    continue

                print(f"[BUN DAU] {cleaned_text}", flush=True)
                await send_event(websocket, "tts_start")

                pcm_bytes, used_voice = await synthesize_whole_response_with_retry(cleaned_text)
                if pcm_bytes is None:
                    print("[TTS] Khong tao duoc audio sau 3 lan thu.", flush=True)
                    await send_event(websocket, "tts_error", "Edge-TTS failed")
                    continue

                await stream_pcm_realtime(websocket, pcm_bytes)
                await send_event(websocket, "tts_done")
                print(
                    f"[WEBSOCKET] Da gui xong TTS | PCM={len(pcm_bytes)} bytes | voice={used_voice}",
                    flush=True,
                )

            except WebSocketDisconnect:
                print("[WEBSOCKET] ESP32 ngat khi dang TTS.", flush=True)
                break
            except Exception as stream_err:
                print(f"[STREAM ERROR] {stream_err}", flush=True)
                try:
                    await send_event(websocket, "tts_error", "Stream failed")
                except Exception:
                    pass

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
    except Exception as exc:
        print(f"[WEBSOCKET ERROR] {exc}", flush=True)
    finally:
        pcm_buffer.clear()

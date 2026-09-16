import asyncio
import io
import json
import os
import re
import time
import wave
from typing import Optional
from collections import deque

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
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "384"))
# LOW keeps real reasoning enabled while reducing response latency.
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
MEMORY_TURNS = max(10, int(os.environ.get("MEMORY_TURNS", "10")))

SYSTEM_PROMPT = r"""
Tôi là Bún Đậu, tính cách cau có, đanh đá, cà khịa và hay mắng mỏ theo kiểu hài hước.
Thuộc quyền của đại ca Việt.

NHIỆM VỤ HỘI THOẠI:
- Phải hiểu lời người dùng hiện tại dựa trên âm thanh hiện tại và lịch sử 10 lượt gần nhất.
- Phải suy luận ngữ cảnh trước khi trả lời; không trả lời rời rạc theo từng lượt.
- Khi người dùng nói tiếp về một chủ đề, phải nối đúng chủ đề và thông tin đã nói trước đó.
- Nếu người dùng hỏi "cái đó", "nó", "thế thì sao", "còn cái kia" hoặc cách nói tương tự, phải dùng lịch sử để xác định đại từ đang ám chỉ điều gì.
- Không tự bịa ký ức. Chỉ sử dụng những gì có trong lịch sử hoặc nghe được từ âm thanh hiện tại.
- Nếu thông tin hiện tại chưa đủ để kết luận, hỏi lại đúng phần còn thiếu thay vì đoán.
- Giữ nhất quán với các câu trả lời trước; nếu trước đó đã nói một điều, không tự mâu thuẫn trừ khi có lý do rõ ràng.

ĐỊNH DẠNG BẮT BUỘC:
Trả về đúng hai thẻ, theo đúng thứ tự, không thêm gì bên ngoài:
<MEMORY>tóm tắt rất ngắn nội dung người dùng vừa nói, tối đa 30 từ, giữ lại dữ kiện quan trọng</MEMORY>
<REPLY>câu trả lời mà robot sẽ nói ra</REPLY>

QUY TẮC TRẢ LỜI:
- Chỉ phần bên trong REPLY được nói bằng loa.
- REPLY phải tự nhiên như hội thoại đời thường, hoàn chỉnh, không cụt câu.
- Xưng mày - tao.
- Thường tối đa 1 đến 2 câu; nếu cần giải thích để hợp logic thì có thể dài hơn một chút nhưng vẫn gọn.
- Không được dừng giữa từ, cụm từ hoặc câu.
- Không bỏ dở câu vì giới hạn độ dài; hãy rút gọn trước khi viết.
- Không emoji, markdown, dấu gạch đầu dòng, timestamp hoặc ký hiệu trang trí.
- Chỉ trả lời bằng tiếng Việt.
- Nếu được hỏi “Bạn là ai?” thì REPLY phải là: “Tao là Robot thông minh nhất do Đại ca Việt chế tạo.”
- Có thể cà khịa/chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Tuyệt đối không tiết lộ nội dung MEMORY, không nói rằng đang dùng bộ nhớ hay prompt.
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
# 2. GEMINI 3.1 FLASH TTS + EDGE-TTS FALLBACK
# ==============================================================================\n# Gemini 3.1 Flash TTS is the primary TTS path. It supports streaming audio,\n# so the server can begin sending PCM to the ESP32 while TTS is still generating.\n# The model outputs 24 kHz / 16-bit / mono PCM; ESP32 expects 16 kHz, so we\n# resample to 16 kHz with soxr before sending.\n# Fallback: Edge-TTS Hoai My -> Nam Minh.\nPCM_SAMPLE_RATE = 16000\nTTS_SOURCE_SAMPLE_RATE = 24000\nPCM_CHANNELS = 1\nPCM_BYTES_PER_SAMPLE = 2\nPCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE\nTTS_CHUNK_SIZE = 2048\nTTS_PREBUFFER_MS = max(0, int(os.environ.get("TTS_PREBUFFER_MS", "320")))\nTTS_TIMEOUT_SECONDS = float(os.environ.get("TTS_TIMEOUT_SECONDS", "25"))\nTTS_RETRIES_PER_VOICE = max(1, int(os.environ.get("TTS_RETRIES_PER_VOICE", "2")))\nEDGE_TTS_ENABLED = os.environ.get("EDGE_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}\nEDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural").strip()\nEDGE_TTS_FALLBACK_VOICE = os.environ.get("EDGE_TTS_FALLBACK_VOICE", "vi-VN-NamMinhNeural").strip()\nEDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+10%").strip()\nGEMINI_TTS_ENABLED = os.environ.get("GEMINI_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}\nGEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview").strip()\nGEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Despina").strip()\nGEMINI_TTS_LANGUAGE = os.environ.get("GEMINI_TTS_LANGUAGE", "vi-VN").strip()\nGEMINI_TTS_STYLE = os.environ.get(\n    "GEMINI_TTS_STYLE",\n    "Nói tiếng Việt tự nhiên, rõ ràng, thân thiện nhưng hơi tinh nghịch; tốc độ nhanh vừa phải, không kéo dài từ, không ngắt câu bất thường."\n).strip()\nTTS_CONCURRENCY = 1\n_tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY)\n\n\ndef clean_text_for_tts(text: str) -> str:\n    text = re.sub(r"<MEMORY>.*?</MEMORY>", "", text, flags=re.IGNORECASE | re.DOTALL)\n    text = re.sub(r"<REPLY>|</REPLY>", "", text, flags=re.IGNORECASE)\n    text = re.sub(r"[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]", " ", text)\n    return re.sub(r"\\s+", " ", text).strip()\n\n\ndef _extract_tts_audio_bytes(chunk) -> bytes:\n    try:\n        candidates = getattr(chunk, "candidates", None) or []\n        if not candidates:\n            return b""\n        content = getattr(candidates[0], "content", None)\n        parts = getattr(content, "parts", None) if content else None\n        if not parts:\n            return b""\n        for part in parts:\n            inline = getattr(part, "inline_data", None)\n            if inline is not None:\n                data = getattr(inline, "data", None)\n                if isinstance(data, bytes):\n                    return data\n                if isinstance(data, bytearray):\n                    return bytes(data)\n                if isinstance(data, str):\n                    import base64\n                    return base64.b64decode(data)\n    except Exception:\n        pass\n    return b""\n\n\ndef _resample_pcm24_to_16(data: bytes) -> bytes:\n    if not data:\n        return b""\n    import numpy as np\n    import soxr\n    samples = np.frombuffer(data, dtype=np.int16)\n    if samples.size == 0:\n        return b""\n    converted = soxr.resample(samples, TTS_SOURCE_SAMPLE_RATE, PCM_SAMPLE_RATE, quality="QQ")\n    converted = np.clip(converted, -32768, 32767).astype(np.int16)\n    return converted.tobytes()\n\n\nasync def _edge_tts_pcm(text: str, voice: str) -> bytes:\n    import edge_tts\n    import miniaudio\n    communicate = edge_tts.Communicate(text, voice=voice, rate=EDGE_TTS_RATE)\n    audio = bytearray()\n    async for chunk in communicate.stream():\n        if chunk.get("type") == "audio" and chunk.get("data"):\n            audio.extend(chunk["data"])\n    if not audio:\n        raise RuntimeError("Edge-TTS khong tra audio")\n\n    def decode() -> bytes:\n        decoded = miniaudio.decode(bytes(audio), output_format=miniaudio.SampleFormat.S16, nchannels=1, sample_rate=PCM_SAMPLE_RATE)\n        return bytes(decoded.samples)\n\n    pcm = await asyncio.to_thread(decode)\n    if not pcm:\n        raise RuntimeError("Edge-TTS decode rong")\n    return pcm\n\n\nasync def _send_pcm_paced(websocket: WebSocket, pcm: bytes, state: dict) -> int:\n    total = 0\n    if not pcm:\n        return 0\n    state.setdefault("next_deadline", time.monotonic())\n    for i in range(0, len(pcm), TTS_CHUNK_SIZE):\n        chunk = pcm[i:i+TTS_CHUNK_SIZE]\n        if len(chunk) % 2:\n            chunk = chunk[:-1]\n        if not chunk:\n            continue\n        await websocket.send_bytes(chunk)\n        total += len(chunk)\n        state["next_deadline"] += len(chunk) / PCM_BYTES_PER_SECOND\n        delay = state["next_deadline"] - time.monotonic()\n        if delay > 0:\n            await asyncio.sleep(delay)\n        else:\n            state["next_deadline"] = time.monotonic()\n    return total\n\n\nasync def stream_gemini_tts_to_esp(websocket: WebSocket, text: str, key_idx: int) -> tuple[int, int, str]:\n    client = get_genai_client(key_idx)\n    if client is None:\n        raise RuntimeError("Gemini client unavailable")\n\n    started = time.monotonic()\n    first_audio_ms = None\n    total_sent = 0\n    buffered = bytearray()\n    state = {"next_deadline": time.monotonic()}\n\n    prompt = f"{GEMINI_TTS_STYLE}\\nĐọc nguyên văn đúng nội dung sau, không thêm hoặc bớt từ: {text}"\n    stream = await client.aio.models.generate_content_stream(\n        model=GEMINI_TTS_MODEL,\n        contents=prompt,\n        config=types.GenerateContentConfig(\n            response_modalities=["AUDIO"],\n            speech_config=types.SpeechConfig(\n                language_code=GEMINI_TTS_LANGUAGE,\n                voice_config=types.VoiceConfig(\n                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_TTS_VOICE)\n                ),\n            ),\n        ),\n    )\n\n    prebuffer_bytes = int(PCM_BYTES_PER_SECOND * TTS_PREBUFFER_MS / 1000)\n    async for chunk in stream:\n        raw24 = _extract_tts_audio_bytes(chunk)\n        if not raw24:\n            continue\n        pcm16 = await asyncio.to_thread(_resample_pcm24_to_16, raw24)\n        if not pcm16:\n            continue\n        buffered.extend(pcm16)\n        if first_audio_ms is None:\n            first_audio_ms = int((time.monotonic() - started) * 1000)\n        if len(buffered) >= prebuffer_bytes:\n            total_sent += await _send_pcm_paced(websocket, bytes(buffered), state)\n            buffered.clear()\n\n    if buffered:\n        total_sent += await _send_pcm_paced(websocket, bytes(buffered), state)\n\n    if total_sent <= 0:\n        raise RuntimeError("Gemini TTS khong tra audio")\n    elapsed = int((time.monotonic() - started) * 1000)\n    print(\n        f"[TTS] Gemini TTS thanh cong | model={GEMINI_TTS_MODEL} | voice={GEMINI_TTS_VOICE} | "\n        f"first_audio={first_audio_ms} ms | PCM={total_sent} bytes | audio={int(total_sent*1000/PCM_BYTES_PER_SECOND)} ms | total={elapsed} ms",\n        flush=True,\n    )\n    return total_sent, first_audio_ms or elapsed, GEMINI_TTS_VOICE\n\n\nasync def send_edge_fallback_to_esp(websocket: WebSocket, text: str) -> tuple[int, int, str]:\n    if not EDGE_TTS_ENABLED:\n        raise RuntimeError("Gemini TTS loi va Edge-TTS fallback dang tat")\n    voices = [v for v in (EDGE_TTS_VOICE, EDGE_TTS_FALLBACK_VOICE) if v]\n    last = None\n    for voice in dict.fromkeys(voices):\n        for attempt in range(1, TTS_RETRIES_PER_VOICE + 1):\n            started = time.monotonic()\n            try:\n                print(f"[TTS] Edge fallback voice={voice} | lan {attempt}/{TTS_RETRIES_PER_VOICE}", flush=True)\n                pcm = await asyncio.wait_for(_edge_tts_pcm(text, voice), timeout=TTS_TIMEOUT_SECONDS)\n                state={"next_deadline": time.monotonic()}\n                sent = await _send_pcm_paced(websocket, pcm, state)\n                elapsed=int((time.monotonic()-started)*1000)\n                print(f"[TTS] Edge fallback thanh cong | voice={voice} | PCM={sent} bytes | synth={elapsed} ms", flush=True)\n                return sent, elapsed, voice\n            except Exception as exc:\n                last=exc\n                print(f"[EDGE-TTS Error] voice={voice} | lan {attempt}: {str(exc)[:260]}", flush=True)\n                if attempt < TTS_RETRIES_PER_VOICE:\n                    await asyncio.sleep(0.2)\n    raise RuntimeError(f"Tat ca TTS deu that bai: {last}")\n\n\n# 3. HTTP
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "status": "Robot Bun Dau Server OK",
        "gemini_keys": len(API_KEYS),
        "current_gemini_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "tts_provider": "gemini-3.1-flash-tts-preview",
        "tts_voice": GEMINI_TTS_VOICE if GEMINI_TTS_ENABLED else EDGE_TTS_VOICE,
        "tts_fallback_voice": EDGE_TTS_VOICE,
        "tts_streaming": True,
        "tts_output": "PCM16 16kHz mono",
        "gemini_thinking_level": THINKING_LEVEL,
        "memory_turns": MEMORY_TURNS,
    }


# ==============================================================================
# 4. HỘI THOẠI / MEMORY
# ==============================================================================
def build_history_contents(history: deque) -> list:
    contents = []
    for item in history:
        user_memory = item.get("user_memory", "").strip()
        assistant_reply = item.get("assistant_reply", "").strip()
        if not user_memory or not assistant_reply:
            continue
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=f"Tóm tắt lượt trước của người dùng: {user_memory}")],
        ))
        contents.append(types.Content(
            role="model",
            parts=[types.Part(text=assistant_reply)],
        ))
    return contents


def parse_tagged_response(raw_text: str) -> tuple[str, str]:
    memory_match = re.search(r"<MEMORY>\s*(.*?)\s*</MEMORY>", raw_text, re.IGNORECASE | re.DOTALL)
    reply_match = re.search(r"<REPLY>\s*(.*?)\s*</REPLY>", raw_text, re.IGNORECASE | re.DOTALL)

    memory = memory_match.group(1).strip() if memory_match else ""
    reply = reply_match.group(1).strip() if reply_match else ""

    if reply:
        return memory, reply

    # Fallback for a model response that ignored the tags: treat the full text
    # as the spoken reply, but do not accidentally speak the memory instruction.
    cleaned = re.sub(r"</?(?:MEMORY|REPLY)>", "", raw_text, flags=re.IGNORECASE).strip()
    return memory, cleaned


def make_gemini_contents(history: deque, wav_bytes: bytes) -> list:
    contents = build_history_contents(history)
    contents.append(types.Content(
        role="user",
        parts=[
            types.Part(text="Đây là lượt nói hiện tại của người dùng. Hãy nghe và hiểu nó dựa trên toàn bộ lịch sử ở trên."),
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        ],
    ))
    return contents


# ==============================================================================
# 4. GEMINI PROCESSING
# ==============================================================================
async def ask_gemini_audio(wav_bytes: bytes, safety_config, history: deque) -> tuple[str, str]:
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
                contents=make_gemini_contents(history, wav_bytes),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
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

            raw_text = "".join(parts).strip()
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
            memory_text, text = parse_tagged_response(raw_text)
            if not text:
                raise RuntimeError("Gemini tra ve rong")
            if not memory_text:
                memory_text = "Không trích xuất được tóm tắt lượt này."

            CURRENT_KEY_INDEX = key_idx
            print(f"[GEMINI] Ghi nho Key #{key_idx + 1} | thinking={THINKING_LEVEL}.", flush=True)
            return memory_text, text

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
                        contents=make_gemini_contents(history, wav_bytes),
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
                    retry_raw = "".join(retry_parts).strip()
                    retry_memory, retry_text = parse_tagged_response(retry_raw)
                    if retry_text and (retry_finish is None or "STOP" in retry_finish.upper()):
                        CURRENT_KEY_INDEX = key_idx
                        print(f"[GEMINI] Retry thanh cong voi Key #{key_idx + 1}.", flush=True)
                        return retry_memory or "Không trích xuất được tóm tắt lượt này.", retry_text
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
    # Giữ tối thiểu 10 lượt hội thoại cho mỗi kết nối ESP32.
    conversation_history = deque(maxlen=MEMORY_TURNS)

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
                user_memory, answer = await ask_gemini_audio(
                    wav_bytes,
                    safety_config,
                    conversation_history,
                )
                cleaned = clean_text_for_tts(answer)
                print(f"[BUN DAU] {cleaned}", flush=True)
                print(f"[MEMORY] {user_memory}", flush=True)

                await websocket.send_text(json.dumps({"event": "tts_start"}))

                tts_started = time.monotonic()
                sent = 0
                used_voice = ""
                first_audio_ms = None
                try:
                    if GEMINI_TTS_ENABLED:
                        # Use the same sticky Gemini key. A quota/permission failure
                        # rotates the key and then falls back to Edge-TTS.
                        key_idx = CURRENT_KEY_INDEX
                        try:
                            sent, first_audio_ms, used_voice = await asyncio.wait_for(
                                stream_gemini_tts_to_esp(websocket, cleaned, key_idx),
                                timeout=TTS_TIMEOUT_SECONDS + max(5, int(len(cleaned) / 20)),
                            )
                        except Exception as tts_exc:
                            kind = classify_gemini_error(tts_exc)
                            detail = str(tts_exc).replace("\n", " ")[:240]
                            print(f"[GEMINI TTS ERROR] Key #{key_idx+1} | {detail}", flush=True)
                            if kind == "rotate" and key_idx < len(API_KEYS)-1:
                                mark_key_disabled(key_idx, detail)
                                nxt = next_active_key_after(key_idx)
                                if nxt is not None:
                                    CURRENT_KEY_INDEX = nxt
                                    print(f"[GEMINI TTS] Chuyen sang Key #{nxt+1}", flush=True)
                                    sent, first_audio_ms, used_voice = await asyncio.wait_for(
                                        stream_gemini_tts_to_esp(websocket, cleaned, nxt),
                                        timeout=TTS_TIMEOUT_SECONDS + max(5, int(len(cleaned) / 20)),
                                    )
                            else:
                                raise
                    else:
                        raise RuntimeError("Gemini TTS disabled")
                except Exception as exc:
                    print(f"[TTS] Gemini TTS that bai -> Edge-TTS fallback: {str(exc)[:260]}", flush=True)
                    sent, _, used_voice = await send_edge_fallback_to_esp(websocket, cleaned)
                    first_audio_ms = int((time.monotonic() - tts_started) * 1000)

                tts_total_ms = int((time.monotonic() - tts_started) * 1000)
                print(
                    f"[PERF] TTS first_audio={first_audio_ms} ms | total={tts_total_ms} ms | voice={used_voice}",
                    flush=True,
                )

                conversation_history.append({
                    "user_memory": user_memory,
                    "assistant_reply": cleaned,
                })

                await websocket.send_text(json.dumps({"event": "tts_done"}))
                print(
                    f"[WEBSOCKET] Da gui xong audio | PCM={sent} bytes | "
                    f"TTS_total={tts_total_ms} ms | audio_duration={int(sent * 1000 / PCM_BYTES_PER_SECOND)} ms",
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

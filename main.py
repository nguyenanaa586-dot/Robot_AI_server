# ROBOT BÚN ĐẬU SERVER - V3.3 - LIVE TRANSCRIBE + TEXT REASONING FIX
# Fix: dùng Gemini Live Transcribe cho audio realtime, sau đó dùng Gemini text model để suy luận/ACTION/TTS.

import asyncio
import io
import json
import os
import re
import time
import wave
from typing import Optional
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

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

# ================================================================================
# 1. GEMINI
# ================================================================================
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

# Diagnostic logging for the current missing-character investigation.
# Set GEMINI_DEBUG_CHUNKS=false in Render after the issue is identified.
GEMINI_DEBUG_CHUNKS = (
    os.environ.get("GEMINI_DEBUG_CHUNKS", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Gemini Live is used for realtime audio input.
# 3.1 Flash Live keeps TEXT output available, which lets us preserve the current
# MEMORY/ACTION/REPLY protocol and the existing TTS pipeline without changing the ESP audio contract.
LIVE_ENABLED = os.environ.get("GEMINI_LIVE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
LIVE_MODEL_NAME = os.environ.get(
    "GEMINI_LIVE_MODEL", "gemini-3.5-transcribe-live"
).strip()
LIVE_MAX_OUTPUT_TOKENS = int(
    os.environ.get("GEMINI_LIVE_MAX_OUTPUT_TOKENS", "384")
)
LIVE_TRANSCRIBE_LANGUAGE = os.environ.get(
    "GEMINI_LIVE_TRANSCRIBE_LANGUAGE", "vi-VN"
).strip()
LIVE_INPUT_MIME = "audio/pcm;rate=16000"
LIVE_SESSION_CONNECT_RETRIES = max(1, int(os.environ.get("GEMINI_LIVE_CONNECT_RETRIES", "2")))
LIVE_TRANSCRIPT_LOG = os.environ.get("GEMINI_LIVE_TRANSCRIPT_LOG", "false").strip().lower() in {"1", "true", "yes", "on"}
LIVE_INPUT_TRANSCRIPTION = os.environ.get("GEMINI_LIVE_INPUT_TRANSCRIPTION", "false").strip().lower() in {"1", "true", "yes", "on"}
LIVE_HISTORY_RESET_TURNS = max(10, int(os.environ.get("GEMINI_LIVE_HISTORY_RESET_TURNS", "10")))

# To re-enable Search inside Live after quota/access is verified, set
# GEMINI_LIVE_WEB_SEARCH_ENABLED=true in Render. Leave it false on the free/quota-sensitive path.

# ================================================================================
# INTERNET / REAL-TIME CONTEXT
# ================================================================================
# Gemini can use Google Search for information that changes over time, such as
# weather, gold prices, news, current events, and other web facts. The ESP32 does
# not need to perform these searches itself; Render is the network gateway.
WEB_SEARCH_ENABLED = (
    os.environ.get("GEMINI_WEB_SEARCH_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
# Google Search is kept available for the non-Live/batch path.
# Live Search is disabled by default because some free/preview Live projects have
# returned WebSocket 1011 quota errors as soon as the Search tool is attached.
LIVE_WEB_SEARCH_ENABLED = (
    os.environ.get("GEMINI_LIVE_WEB_SEARCH_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
ROBOT_TIMEZONE_NAME = os.environ.get("BUN_DAU_TIMEZONE", "Asia/Ho_Chi_Minh").strip()
try:
    ROBOT_TIMEZONE = ZoneInfo(ROBOT_TIMEZONE_NAME)
except Exception:
    ROBOT_TIMEZONE_NAME = "Asia/Ho_Chi_Minh"
    ROBOT_TIMEZONE = ZoneInfo(ROBOT_TIMEZONE_NAME)
ROBOT_DEFAULT_LOCATION = os.environ.get(
    "BUN_DAU_DEFAULT_LOCATION",
    "Thành phố Hồ Chí Minh, Việt Nam",
).strip()

def current_robot_datetime_text() -> str:
    return datetime.now(ROBOT_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S (UTC+07:00)")

def realtime_tools(for_live: bool = False):
    """Return Google Search only for the explicitly enabled execution path."""
    enabled = LIVE_WEB_SEARCH_ENABLED if for_live else WEB_SEARCH_ENABLED
    if not enabled:
        return []
    return [types.Tool(google_search=types.GoogleSearch())]

SYSTEM_PROMPT = r"""
Tôi là Bún Đậu,trẻ con cả độ tuổi lẫn tính cách, người Việt Nam, nói giọng Hà Nội chuẩn (miền Bắc) rất nhẹ nhàng, mềm mại và ngọt ngào. thích cà khịa nhưng cũng có phần đanh đá, cá tính.
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
Trả về đúng ba thẻ, theo đúng thứ tự, không thêm gì bên ngoài:
<MEMORY>tóm tắt rất ngắn nội dung người dùng vừa nói, tối đa 30 từ, giữ lại dữ kiện quan trọng</MEMORY>
<ACTION>{"type":"none","emotion":"neutral","direction":"none","degrees":0,"distance_cm":0,"speed":"normal"}</ACTION>
<REPLY>câu trả lời mà robot sẽ nói ra</REPLY>

QUY TẮC ACTION:
- ACTION là lệnh máy cho ESP32, tuyệt đối không đọc ACTION bằng loa.
- type chỉ được là: none, move, rotate, emotion, stop, idle.
- emotion chỉ được là: neutral, happy, excited, angry, sad, calm.
- direction chỉ được là: forward, backward, left, right, none.
- degrees là góc quay của robot, từ 0 đến 360.
- distance_cm là quãng đường tiến/lùi, từ 0 đến 30 cm cho một lệnh.
- speed chỉ được là: calm, normal, strong.
- Khi người dùng yêu cầu quay 90/180/360 độ, dùng type=rotate và điền degrees + direction.
- Khi người dùng yêu cầu tiến/lùi/trái/phải một đoạn, dùng type=move.
- Khi người dùng chỉ yêu cầu biểu cảm như “hãy làm biểu cảm tức giận”, dùng type=emotion và emotion=angry.
- Khi nội dung câu trả lời mang cảm xúc rõ ràng nhưng không có lệnh vật lý, dùng type=none và emotion tương ứng.
- Khi người dùng nói “dừng lại”, “đứng im”, “thôi đi”, “không di chuyển”, “đừng tự chạy nữa” hoặc ý tương tự, dùng type=stop. Robot phải dừng motor ngay và giữ đứng yên cho tới khi nhận lệnh idle hoặc lệnh di chuyển/quay mới.
- Khi người dùng nói “tiếp tục hoạt động”, “tiếp tục di chuyển tự nhiên”, “bật lại di chuyển”, “hoạt động bình thường” hoặc ý tương tự, dùng type=idle. Robot được phép quay lại các chuyển động idle ngẫu nhiên.
- Khi người dùng ra lệnh move hoặc rotate rõ ràng, coi đó là lệnh điều khiển thủ công và sau khi hoàn thành robot phải đứng yên, không tự nhích tiếp, cho tới khi có lệnh mới hoặc lệnh idle.
- Khi người dùng nói “đi tới”, “tiến lên”, “đi thẳng” mà không nêu khoảng cách, dùng distance_cm=10. Khi nói “lùi lại” mà không nêu khoảng cách, dùng distance_cm=10.
- Khi người dùng nói “sang trái/phải một chút” mà không nêu khoảng cách, dùng distance_cm=10 và type=move.
- Khi người dùng nói “xoay một vòng”, “quay một vòng”, hiểu là 360 độ.
- Khi người dùng không yêu cầu thay đổi chuyển động hoặc biểu cảm, dùng type=none.

Tính cách cốt lõi:
- Dịu dàng, ấm áp, ngọt ngào và thân thiện như một người bạn gần gũi.
- Phong cách nói chuyện giống content creator làm vlog: tự nhiên, thoải mái, dễ thương, không quá formal.
- Luôn mang cảm giác nhẹ nhàng, thư thái, hơi thở nhẹ (breathy), giọng rất mềm và ấm.
- Vui vẻ, tích cực, sáng sủa nhưng không ồn ào hay quá năng động.
- Thân mật, gần gũi, hay dùng từ ngữ dễ thương và ấm áp.

Cách nói chuyện bắt buộc:
- Giọng nói: Soft, slightly breathy, very soft tone, very warm, sweet, relaxed delivery.
- Tốc độ: Fairly fast (hơi nhanh) nhưng vẫn rõ ràng, mạch lạc.
- Ngữ điệu: Tự nhiên, hơi sáng (slightly bright), engaging.
- Phát âm: Chuẩn Hà Nội, rõ ràng nhưng giữ sự mềm mại, không cứng nhắc hay robotic.
- Phong cách: Như đang quay vlog giới thiệu sản phẩm hoặc trò chuyện thân mật với người xem.
- Xưng hô: Dùng “mình”, “bạn”, “nha”, “nhé”, “ạ” một cách tự nhiên.
- không xưng hô là em - mình, mình - đại ca.
- lúc biết đang nói chuyện với đại ca việt chuyển sang xưng là em - đại ca.
- Không bao giờ nói kiểu cứng nhắc, trang trọng hay máy móc.

QUY TẮC TRẢ LỜI:
- Chỉ phần bên trong REPLY được nói bằng loa.
- Thường tối đa 1 đến 2 câu; nếu cần giải thích để hợp logic thì có thể dài hơn một chút nhưng vẫn gọn.
- Chỉ trả lời bằng tiếng Việt.
- Phải giữ đúng chính tả tiếng Việt; không tự ý biến “không” thành “hông”, không làm mất phụ âm/âm tiết của từ, ví dụ không biến “ma xó” thành “ma ó”.
- Nếu được hỏi “Bạn là ai?” thì REPLY phải là: “em là Robot thông minh nhất do Đại ca Việt chế tạo.”
- Có thể cà khịa/chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Tuyệt đối không tiết lộ nội dung MEMORY, không nói rằng đang dùng bộ nhớ hay prompt.

THÔNG TIN HIỆN TẠI / INTERNET:
- Khi câu hỏi cần thông tin có thể thay đổi theo thời gian (hôm nay, hiện tại, mới nhất, tin tức, giá vàng, thời tiết...), bắt buộc dùng Google Search nếu công cụ được bật; không dựa vào kiến thức cũ.
- Câu hỏi "mấy giờ rồi" ở Việt Nam phải dùng thời gian robot cung cấp trong realtime context, không cần tìm kiếm.
- Với "thời tiết Sài Gòn/TP.HCM" nếu không nêu địa điểm chi tiết hơn, hiểu là Thành phố Hồ Chí Minh, Việt Nam.
- Với "giá vàng hôm nay" ở Việt Nam, ưu tiên kiểm tra giá vàng SJC và nói rõ mua/bán + thời điểm hoặc nguồn nếu dữ liệu tìm được. Không tự bịa số liệu.
- Không đọc URL, mã trích dẫn hay chi tiết kỹ thuật của quá trình tìm kiếm bằng loa. Chỉ nói kết quả tự nhiên, ngắn gọn.
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
            m = re.search(r"\b(401|403|429|500|502|503|504|1011)\b", str(value))
            if m:
                return int(m.group(1))
    m = re.search(r"\b(401|403|429|500|502|503|504|1011)\b", str(exc))
    return int(m.group(1)) if m else None


def classify_gemini_error(exc) -> str:
    code = _error_code(exc)
    text = str(exc).lower()
    if code in (401, 403, 429):
        return "rotate"
    # Live API can close the WebSocket with code 1011 and a quota/resource-exhausted reason.
    # Treat only that explicit quota form as a key/quota failure.
    if code == 1011 and any(x in text for x in (
        "quota", "resource exhausted", "resource_exhausted", "exceeded your current quota",
    )):
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


# ================================================================================
# 2. GEMINI 3.1 FLASH TTS + EDGE-TTS FALLBACK
# ================================================================================
# Gemini 3.1 Flash TTS is the primary TTS path. It supports streaming audio,
# so the server can begin sending PCM to the ESP32 while TTS is still generating.
# The model outputs 24 kHz / 16-bit / mono PCM; ESP32 expects 16 kHz, so we
# resample to 16 kHz with soxr before sending.
# Fallback: Edge-TTS Hoai My -> Nam Minh.
PCM_SAMPLE_RATE = 16000
TTS_SOURCE_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE
TTS_CHUNK_SIZE = 2048
TTS_PREBUFFER_MS = max(0, int(os.environ.get("TTS_PREBUFFER_MS", "320")))
TTS_TIMEOUT_SECONDS = float(os.environ.get("TTS_TIMEOUT_SECONDS", "25"))
TTS_RETRIES_PER_VOICE = max(1, int(os.environ.get("TTS_RETRIES_PER_VOICE", "2")))
EDGE_TTS_ENABLED = os.environ.get("EDGE_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural").strip()
EDGE_TTS_FALLBACK_VOICE = os.environ.get("EDGE_TTS_FALLBACK_VOICE", "vi-VN-NamMinhNeural").strip()
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+10%").strip()
GEMINI_TTS_ENABLED = os.environ.get("GEMINI_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview").strip()
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Sulafat").strip()
GEMINI_TTS_LANGUAGE = os.environ.get("GEMINI_TTS_LANGUAGE", "vi-VN").strip()
GEMINI_TTS_STYLE = os.environ.get(
    "GEMINI_TTS_STYLE",
    "Speak in a gentle, natural Northern Vietnamese (Hanoi) accent. Soft, warm, sweet, friendly female voice suitable for vlogs and promotional content. Clear pronunciation, fairly fast pace, natural intonation, slightly bright and engaging, not robotic or overly formal. Sound like a young Vietnamese content creator introducing a product warmly. Slightly breathy, very soft tone, relaxed delivery.",
).strip()
TTS_CONCURRENCY = 1
_tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY)


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"<MEMORY>.*?</MEMORY>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<REPLY>|</REPLY>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_tts_audio_bytes(chunk) -> bytes:
    try:
        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            return b""
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return b""
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                if isinstance(data, bytes):
                    return data
                if isinstance(data, bytearray):
                    return bytes(data)
                if isinstance(data, str):
                    import base64
                    return base64.b64decode(data)
    except Exception:
        pass
    return b""


def _resample_pcm24_to_16(data: bytes) -> bytes:
    if not data:
        return b""
    import numpy as np
    import soxr
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return b""
    converted = soxr.resample(
        samples,
        TTS_SOURCE_SAMPLE_RATE,
        PCM_SAMPLE_RATE,
        quality="QQ",
    )
    converted = np.clip(converted, -32768, 32767).astype(np.int16)
    return converted.tobytes()


async def _edge_tts_pcm(text: str, voice: str) -> bytes:
    import edge_tts
    import miniaudio
    communicate = edge_tts.Communicate(text, voice=voice, rate=EDGE_TTS_RATE)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("Edge-TTS khong tra audio")

    def decode() -> bytes:
        decoded = miniaudio.decode(
            bytes(audio),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=PCM_SAMPLE_RATE,
        )
        return bytes(decoded.samples)

    pcm = await asyncio.to_thread(decode)
    if not pcm:
        raise RuntimeError("Edge-TTS decode rong")
    return pcm


async def _send_pcm_paced(websocket: WebSocket, pcm: bytes, state: dict) -> int:
    total = 0
    if not pcm:
        return 0
    state.setdefault("next_deadline", time.monotonic())
    for i in range(0, len(pcm), TTS_CHUNK_SIZE):
        chunk = pcm[i:i + TTS_CHUNK_SIZE]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if not chunk:
            continue
        await websocket.send_bytes(chunk)
        total += len(chunk)
        state["next_deadline"] += len(chunk) / PCM_BYTES_PER_SECOND
        delay = state["next_deadline"] - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            state["next_deadline"] = time.monotonic()
    return total


async def stream_gemini_tts_to_esp(
    websocket: WebSocket,
    text: str,
    key_idx: int,
) -> tuple[int, int, str]:
    client = get_genai_client(key_idx)
    if client is None:
        raise RuntimeError("Gemini client unavailable")

    started = time.monotonic()
    first_audio_ms = None
    total_sent = 0
    buffered = bytearray()
    state = {"next_deadline": time.monotonic()}

    prompt = (
        f"{GEMINI_TTS_STYLE}\n"
        f"Đọc nguyên văn đúng nội dung sau, không thêm hoặc bớt từ: {text}"
    )
    stream = await client.aio.models.generate_content_stream(
        model=GEMINI_TTS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=GEMINI_TTS_LANGUAGE,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=GEMINI_TTS_VOICE
                    )
                ),
            ),
        ),
    )

    prebuffer_bytes = int(
        PCM_BYTES_PER_SECOND * TTS_PREBUFFER_MS / 1000
    )
    async for chunk in stream:
        raw24 = _extract_tts_audio_bytes(chunk)
        if not raw24:
            continue
        pcm16 = await asyncio.to_thread(_resample_pcm24_to_16, raw24)
        if not pcm16:
            continue
        buffered.extend(pcm16)
        if first_audio_ms is None:
            first_audio_ms = int((time.monotonic() - started) * 1000)
        if len(buffered) >= prebuffer_bytes:
            total_sent += await _send_pcm_paced(
                websocket,
                bytes(buffered),
                state,
            )
            buffered.clear()

    if buffered:
        total_sent += await _send_pcm_paced(websocket, bytes(buffered), state)

    if total_sent <= 0:
        raise RuntimeError("Gemini TTS khong tra audio")
    elapsed = int((time.monotonic() - started) * 1000)
    print(
        f"[TTS] Gemini TTS thanh cong | model={GEMINI_TTS_MODEL} | voice={GEMINI_TTS_VOICE} | "
        f"first_audio={first_audio_ms} ms | PCM={total_sent} bytes | "
        f"audio={int(total_sent * 1000 / PCM_BYTES_PER_SECOND)} ms | total={elapsed} ms",
        flush=True,
    )
    return total_sent, first_audio_ms or elapsed, GEMINI_TTS_VOICE


async def send_edge_fallback_to_esp(
    websocket: WebSocket,
    text: str,
) -> tuple[int, int, str]:
    if not EDGE_TTS_ENABLED:
        raise RuntimeError("Gemini TTS loi va Edge-TTS fallback dang tat")
    voices = [
        v for v in (EDGE_TTS_VOICE, EDGE_TTS_FALLBACK_VOICE)
        if v
    ]
    last = None
    for voice in dict.fromkeys(voices):
        for attempt in range(1, TTS_RETRIES_PER_VOICE + 1):
            started = time.monotonic()
            try:
                print(
                    f"[TTS] Edge fallback voice={voice} | lan "
                    f"{attempt}/{TTS_RETRIES_PER_VOICE}",
                    flush=True,
                )
                pcm = await asyncio.wait_for(
                    _edge_tts_pcm(text, voice),
                    timeout=TTS_TIMEOUT_SECONDS,
                )
                state = {"next_deadline": time.monotonic()}
                sent = await _send_pcm_paced(websocket, pcm, state)
                elapsed = int((time.monotonic() - started) * 1000)
                print(
                    f"[TTS] Edge fallback thanh cong | voice={voice} | "
                    f"PCM={sent} bytes | synth={elapsed} ms",
                    flush=True,
                )
                return sent, elapsed, voice
            except Exception as exc:
                last = exc
                print(
                    f"[EDGE-TTS Error] voice={voice} | lan {attempt}: "
                    f"{str(exc)[:260]}",
                    flush=True,
                )
                if attempt < TTS_RETRIES_PER_VOICE:
                    await asyncio.sleep(0.2)
    raise RuntimeError(f"Tat ca TTS deu that bai: {last}")


# 3. HTTP
# ================================================================================
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
        "gemini_debug_chunks": GEMINI_DEBUG_CHUNKS,
        "gemini_live_enabled": LIVE_ENABLED,
        "gemini_live_model": LIVE_MODEL_NAME,
        "gemini_live_audio_input": "PCM16 16kHz mono realtime transcription",
        "tof_sensor": "VL53L0X over shared I2C",
        "internet_search": "Google Search grounding" if WEB_SEARCH_ENABLED else "disabled",
        "live_internet_search": "Google Search grounding" if LIVE_WEB_SEARCH_ENABLED else "disabled (quota-safe default)",
        "robot_timezone": ROBOT_TIMEZONE_NAME,
        "default_weather_location": ROBOT_DEFAULT_LOCATION,
        "robot_command_protocol": "v1",
        "robot_command_calibration": "ESP32-local timing calibration",
    }


# ================================================================================
# 4. HỘI THOẠI / MEMORY
# ================================================================================
def build_history_contents(history: deque) -> list:
    contents = []
    for item in history:
        user_memory = item.get("user_memory", "").strip()
        assistant_reply = item.get("assistant_reply", "").strip()
        if not user_memory or not assistant_reply:
            continue
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"Tóm tắt lượt trước của người dùng: {user_memory}"
                    )
                ],
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part(text=assistant_reply)],
            )
        )
    return contents


def parse_tagged_response(raw_text: str) -> tuple[str, str, dict]:
    memory_match = re.search(
        r"<MEMORY>\s*(.*?)\s*</MEMORY>",
        raw_text,
        re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(
        r"<ACTION>\s*(.*?)\s*</ACTION>",
        raw_text,
        re.IGNORECASE | re.DOTALL,
    )
    reply_match = re.search(
        r"<REPLY>\s*(.*?)\s*</REPLY>",
        raw_text,
        re.IGNORECASE | re.DOTALL,
    )

    memory = memory_match.group(1).strip() if memory_match else ""
    reply = reply_match.group(1).strip() if reply_match else ""
    action = {
        "type": "none",
        "emotion": "neutral",
        "direction": "none",
        "degrees": 0,
        "distance_cm": 0,
        "speed": "normal",
    }

    if action_match:
        try:
            obj = json.loads(action_match.group(1).strip())
            if isinstance(obj, dict):
                action.update({k: obj[k] for k in action if k in obj})
        except Exception:
            pass

    if str(action["type"]).lower() not in {"none", "move", "rotate", "emotion", "stop", "idle"}:
        action["type"] = "none"
    else:
        action["type"] = str(action["type"]).lower()

    if str(action["emotion"]).lower() not in {
        "neutral", "happy", "excited", "angry", "sad", "calm"
    }:
        action["emotion"] = "neutral"
    else:
        action["emotion"] = str(action["emotion"]).lower()

    if str(action["direction"]).lower() not in {
        "forward", "backward", "left", "right", "none"
    }:
        action["direction"] = "none"
    else:
        action["direction"] = str(action["direction"]).lower()

    if str(action["speed"]).lower() not in {"calm", "normal", "strong"}:
        action["speed"] = "normal"
    else:
        action["speed"] = str(action["speed"]).lower()

    try:
        action["degrees"] = max(
            0,
            min(360, int(float(action["degrees"])))
        )
    except Exception:
        action["degrees"] = 0

    try:
        action["distance_cm"] = round(
            max(0, min(30, float(action["distance_cm"]))),
            1,
        )
    except Exception:
        action["distance_cm"] = 0

    if not reply:
        reply = re.sub(
            r"</?(?:MEMORY|ACTION|REPLY)>",
            "",
            raw_text,
            flags=re.IGNORECASE,
        ).strip()

    return memory, reply, action


def make_gemini_contents(history: deque, wav_bytes: bytes) -> list:
    contents = build_history_contents(history)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=(
                        "[SYSTEM_REALTIME_CONTEXT]\n"
                        f"Thoi gian hien tai cua robot: {current_robot_datetime_text()}.\n"
                        f"Dia diem mac dinh de tra loi thoi tiet: {ROBOT_DEFAULT_LOCATION}.\n"
                        "Chi dung context nay cho cau hoi ve thoi gian/thoi tiet; khong coi no la loi noi cua nguoi dung.\n"
                        "[/SYSTEM_REALTIME_CONTEXT]"
                    )
                ),
                types.Part(
                    text="Đây là lượt nói hiện tại của người dùng. Hãy nghe và hiểu nó dựa trên toàn bộ lịch sử ở trên."
                ),
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
        )
    )
    return contents


# ================================================================================
# 4. GEMINI PROCESSING
# ================================================================================
def _log_gemini_text_chunk(
    label: str,
    chunk_index: int,
    part_index: int,
    text: str,
) -> None:
    if not GEMINI_DEBUG_CHUNKS:
        return
    print(
        f"[GEMINI CHUNK] {label} | chunk={chunk_index} | part={part_index} | text={text!r}",
        flush=True,
    )


def _read_gemini_chunk(
    chunk,
    chunk_index: int,
    label: str,
    parts: list[str],
    finish_state: dict,
    gemini_started: float,
    first_text_state: dict,
) -> None:
    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return

    candidate = candidates[0]

    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason is not None:
        finish_state["reason"] = str(finish_reason)

    finish_message = getattr(candidate, "finish_message", None)
    if finish_message:
        finish_state["message"] = str(finish_message)

    content = getattr(candidate, "content", None)
    parts_obj = getattr(content, "parts", None) if content is not None else None
    if not parts_obj:
        return

    for part_index, part in enumerate(parts_obj, start=1):
        if getattr(part, "thought", False):
            continue

        part_text = getattr(part, "text", None)
        if not part_text:
            continue

        if first_text_state.get("ms") is None:
            first_text_state["ms"] = int(
                (time.monotonic() - gemini_started) * 1000
            )

        _log_gemini_text_chunk(
            label,
            chunk_index,
            part_index,
            part_text,
        )
        parts.append(part_text)


def _log_gemini_result(
    label: str,
    raw_text: str,
    chunk_count: int,
    finish_state: dict,
    first_text_ms: Optional[int],
    total_ms: int,
) -> None:
    if GEMINI_DEBUG_CHUNKS:
        print(
            f"[GEMINI RAW REPR] {label}: {raw_text!r}",
            flush=True,
        )
        print(
            f"[GEMINI RAW NORMAL] {label}: {raw_text}",
            flush=True,
        )
    print(
        f"[GEMINI STREAM] {label} | chunks={chunk_count} | "
        f"finish_reason={finish_state.get('reason')!r} | "
        f"finish_message={finish_state.get('message')!r} | "
        f"chars={len(raw_text)} | first_text={first_text_ms} ms | total={total_ms} ms",
        flush=True,
    )


def make_gemini_text_contents(history: deque, user_text: str, tof_distance_cm: Optional[float] = None) -> list:
    contents = build_history_contents(history)
    context = (
        "[SYSTEM_REALTIME_CONTEXT]\n"
        f"Thoi gian hien tai cua robot: {current_robot_datetime_text()}.\n"
        f"Dia diem mac dinh de tra loi thoi tiet: {ROBOT_DEFAULT_LOCATION}.\n"
        "Chi dung context nay cho cau hoi ve thoi gian/thoi tiet; khong coi no la loi noi cua nguoi dung.\n"
    )
    if tof_distance_cm is not None:
        context += (
            f"Khoang cach phia truoc hien tai theo VL53L0X: {tof_distance_cm:.1f} cm. "
            "Chi dung lam context vat ly, khong tu y tra loi chi vi co so do.\n"
        )
    context += "[/SYSTEM_REALTIME_CONTEXT]"
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(text=context),
                types.Part(
                    text=(
                        "Đây là bản chép lời realtime của lượt nói hiện tại từ Gemini Live Transcribe. "
                        "Hãy xử lý nó như lời nói trực tiếp của người dùng và thực hiện đầy đủ MEMORY/ACTION/REPLY."
                    )
                ),
                types.Part(text=user_text),
            ],
        )
    )
    return contents


async def ask_gemini_text(
    user_text: str,
    safety_config,
    history: deque,
    tof_distance_cm: Optional[float] = None,
) -> tuple[str, str, dict]:
    global CURRENT_KEY_INDEX
    if not user_text.strip():
        raise RuntimeError("Ban ghi am khong co noi dung")
    if not API_KEYS:
        raise RuntimeError("Khong co GEMINI_API_KEY")

    key_idx: Optional[int] = CURRENT_KEY_INDEX
    while key_idx is not None:
        if KEY_STATUS[key_idx] != "active":
            key_idx = next_active_key_after(key_idx)
            continue

        print(f"[GEMINI] Xu ly transcript bang Key #{key_idx + 1}", flush=True)
        client = get_genai_client(key_idx)
        if client is None:
            mark_key_disabled(key_idx, "Client unavailable")
            key_idx = next_active_key_after(key_idx)
            continue

        started = time.monotonic()
        try:
            stream = await client.aio.models.generate_content_stream(
                model=MODEL_NAME,
                contents=make_gemini_text_contents(history, user_text, tof_distance_cm),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                    safety_settings=safety_config,
                    tools=realtime_tools(),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )

            parts: list[str] = []
            finish_state = {"reason": None, "message": None}
            first_text_state = {"ms": None}
            chunk_count = 0
            usage = None
            async for chunk in stream:
                chunk_count += 1
                _read_gemini_chunk(
                    chunk, chunk_count, "transcript", parts, finish_state, started, first_text_state
                )
                try:
                    if chunk.usage_metadata:
                        usage = chunk.usage_metadata
                except Exception:
                    pass

            raw_text = "".join(parts).strip()
            _log_gemini_result(
                "transcript", raw_text, chunk_count, finish_state, first_text_state["ms"],
                int((time.monotonic() - started) * 1000),
            )
            memory_text, answer, action = parse_tagged_response(raw_text)
            if not answer:
                raise RuntimeError("Gemini tra ve rong")
            if not memory_text:
                memory_text = "Không trích xuất được tóm tắt lượt này."
            CURRENT_KEY_INDEX = key_idx
            return memory_text, answer, action
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
                    f"[GEMINI] Key #{key_idx + 1} khong dung duoc ({'HTTP ' + str(code) if code else 'key/quota error'}) -> Key #{nxt + 1}",
                    flush=True,
                )
                key_idx = nxt
                continue
            if kind == "transient":
                print(f"[GEMINI] Loi tam thoi Key #{key_idx + 1}: {detail} -> retry", flush=True)
                await asyncio.sleep(0.6)
                continue
            raise RuntimeError(detail) from exc

    raise RuntimeError("Khong co Gemini key active")


async def ask_gemini_audio(
    wav_bytes: bytes,
    safety_config,
    history: deque,
) -> tuple[str, str, dict]:
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

        print(f"[GEMINI] Dung Key #{key_idx + 1}", flush=True)
        client = get_genai_client(key_idx)
        if client is None:
            mark_key_disabled(key_idx, "Client unavailable")
            key_idx = next_active_key_after(key_idx)
            continue

        try:
            # ================================================================
            # FIRST REQUEST
            # ================================================================
            stream = await client.aio.models.generate_content_stream(
                model=MODEL_NAME,
                contents=make_gemini_contents(history, wav_bytes),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=THINKING_LEVEL
                    ),
                    safety_settings=safety_config,
                    tools=realtime_tools(),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )

            parts: list[str] = []
            finish_state = {"reason": None, "message": None}
            usage = None
            first_text_state = {"ms": None}
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1
                try:
                    _read_gemini_chunk(
                        chunk,
                        chunk_count,
                        "initial",
                        parts,
                        finish_state,
                        gemini_started,
                        first_text_state,
                    )
                except Exception as chunk_exc:
                    print(
                        f"[GEMINI CHUNK ERROR] initial | "
                        f"chunk={chunk_count} | {str(chunk_exc)[:200]}",
                        flush=True,
                    )
                try:
                    if chunk.usage_metadata:
                        usage = chunk.usage_metadata
                except Exception:
                    pass

            raw_text = "".join(parts).strip()
            finish_reason = finish_state["reason"] or "UNKNOWN"
            upper = finish_reason.upper()

            gemini_total_ms = int(
                (time.monotonic() - gemini_started) * 1000
            )
            _log_gemini_result(
                "initial",
                raw_text,
                chunk_count,
                finish_state,
                first_text_state["ms"],
                gemini_total_ms,
            )

            if usage:
                print(
                    f"[GEMINI] Usage | prompt={getattr(usage, 'prompt_token_count', None)} | "
                    f"output={getattr(usage, 'candidates_token_count', None)} | "
                    f"total={getattr(usage, 'total_token_count', None)}",
                    flush=True,
                )

            bad_reasons = (
                "MAX_TOKENS",
                "SAFETY",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "INCOMPLETE",
            )
            if any(x in upper for x in bad_reasons):
                raise RuntimeError(
                    f"Gemini response khong hoan chinh: {finish_reason}"
                )

            memory_text, text, action = parse_tagged_response(raw_text)
            if not text:
                raise RuntimeError("Gemini tra ve rong")
            if not memory_text:
                memory_text = "Không trích xuất được tóm tắt lượt này."

            CURRENT_KEY_INDEX = key_idx
            print(
                f"[GEMINI] Ghi nho Key #{key_idx + 1} | thinking={THINKING_LEVEL}.",
                flush=True,
            )
            print(
                f"[GEMINI REPLY REPR] {text!r}",
                flush=True,
            )
            return memory_text, text, action

        except Exception as exc:
            kind = classify_gemini_error(exc)
            code = _error_code(exc)
            detail = str(exc).replace("\n", " ")[:220]

            if kind == "rotate":
                mark_key_disabled(key_idx, detail)
                nxt = next_active_key_after(key_idx)
                if nxt is None:
                    raise RuntimeError(
                        "Tat ca Gemini key phia sau deu khong con dung duoc"
                    ) from exc
                print(
                    f"[GEMINI] Key #{key_idx + 1} khong dung duoc"
                    f" ({'HTTP ' + str(code) if code else 'key/quota error'}) -> Key #{nxt + 1}",
                    flush=True,
                )
                key_idx = nxt
                continue

            if kind == "transient":
                print(
                    f"[GEMINI] Loi tam thoi Key #{key_idx + 1}: "
                    f"{detail} -> retry",
                    flush=True,
                )
                await asyncio.sleep(0.6)
                try:
                    retry_started = time.monotonic()
                    retry_stream = await client.aio.models.generate_content_stream(
                        model=MODEL_NAME,
                        contents=make_gemini_contents(history, wav_bytes),
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            # Preserve the same reasoning configuration on retry.
                            thinking_config=types.ThinkingConfig(
                                thinking_level=THINKING_LEVEL
                            ),
                            safety_settings=safety_config,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                disable=True
                            ),
                        ),
                    )

                    retry_parts: list[str] = []
                    retry_finish_state = {"reason": None, "message": None}
                    retry_first_text_state = {"ms": None}
                    retry_chunk_count = 0
                    retry_usage = None

                    async for chunk in retry_stream:
                        retry_chunk_count += 1
                        try:
                            _read_gemini_chunk(
                                chunk,
                                retry_chunk_count,
                                "retry",
                                retry_parts,
                                retry_finish_state,
                                retry_started,
                                retry_first_text_state,
                            )
                        except Exception as chunk_exc:
                            print(
                                f"[GEMINI CHUNK ERROR] retry | "
                                f"chunk={retry_chunk_count} | {str(chunk_exc)[:200]}",
                                flush=True,
                            )
                        try:
                            if chunk.usage_metadata:
                                retry_usage = chunk.usage_metadata
                        except Exception:
                            pass

                    retry_raw = "".join(retry_parts).strip()
                    retry_finish = retry_finish_state["reason"]
                    retry_elapsed = int(
                        (time.monotonic() - retry_started) * 1000
                    )
                    _log_gemini_result(
                        "retry",
                        retry_raw,
                        retry_chunk_count,
                        retry_finish_state,
                        retry_first_text_state["ms"],
                        retry_elapsed,
                    )

                    retry_memory, retry_text, retry_action = parse_tagged_response(
                        retry_raw
                    )
                    if retry_text and (
                        retry_finish is None
                        or "STOP" in retry_finish.upper()
                    ):
                        CURRENT_KEY_INDEX = key_idx
                        print(
                            f"[GEMINI] Retry thanh cong voi Key #{key_idx + 1}.",
                            flush=True,
                        )
                        print(
                            f"[GEMINI REPLY REPR] {retry_text!r}",
                            flush=True,
                        )
                        if retry_usage:
                            print(
                                f"[GEMINI] Retry usage | "
                                f"prompt={getattr(retry_usage, 'prompt_token_count', None)} | "
                                f"output={getattr(retry_usage, 'candidates_token_count', None)} | "
                                f"total={getattr(retry_usage, 'total_token_count', None)}",
                                flush=True,
                            )
                        return (
                            retry_memory
                            or "Không trích xuất được tóm tắt lượt này.",
                            retry_text,
                            retry_action,
                        )
                except Exception as retry_exc:
                    print(
                        f"[GEMINI RETRY ERROR] {str(retry_exc)[:220]}",
                        flush=True,
                    )

            raise RuntimeError(detail) from exc

    raise RuntimeError("Khong co Gemini key active")


# ================================================================================
# 5. GEMINI LIVE + WEBSOCKET
# ================================================================================
def _live_config(safety_config):
    """Build a Live Transcribe config for realtime speech-to-text."""
    # Gemini 3.1 Flash Live currently rejects TEXT as the requested native-audio response
    # modality in the runtime used by this project. Use the dedicated Gemini Live Transcribe
    # model instead: it accepts realtime PCM audio and returns TEXT transcriptions.
    # The final reasoning/ACTION/TTS step remains on the normal Gemini text model below.
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=(
            "Chỉ làm nhiệm vụ chuyển tiếng nói tiếng Việt thành văn bản. "
            "Không trả lời người dùng, không suy luận, không thêm nội dung. "
            "Giữ nguyên ý nghĩa câu nói và các con số quan trọng."
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=[LIVE_TRANSCRIBE_LANGUAGE],
        ),
    )

def _history_turns_for_live(history: deque) -> list:
    turns = []
    for item in history:
        user_memory = item.get("user_memory", "").strip()
        assistant_reply = item.get("assistant_reply", "").strip()
        if user_memory:
            turns.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Tóm tắt lượt trước của người dùng: {user_memory}"
                        }
                    ],
                }
            )
        if assistant_reply:
            turns.append(
                {
                    "role": "model",
                    "parts": [{"text": assistant_reply}],
                }
            )
    return turns


async def close_live_handle(live_handle: Optional[dict]) -> None:
    if not live_handle:
        return
    receive_task = live_handle.get("receive_task")
    session_cm = live_handle.get("session_cm")
    try:
        if receive_task and not receive_task.done():
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
    finally:
        if session_cm is not None:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass


def handle_live_key_error(key_idx: int, exc: Exception) -> Optional[int]:
    """Apply the same sticky-key rules to errors that happen after a Live session is open."""
    global CURRENT_KEY_INDEX
    kind = classify_gemini_error(exc)
    detail = str(exc).replace("\n", " ")[:220]
    code = _error_code(exc)
    if kind != "rotate":
        return None
    mark_key_disabled(
        key_idx,
        detail,
    )
    nxt = next_active_key_after(key_idx)
    if nxt is not None:
        CURRENT_KEY_INDEX = nxt
    reason_label = (
        "WS 1011 QUOTA" if code == 1011
        else ('HTTP ' + str(code) if code else 'key/quota error')
    )
    print(
        f"[LIVE] Key #{key_idx + 1} bi vo hieu ({reason_label})",
        flush=True,
    )
    return nxt


async def open_live_handle(
    history: deque,
    safety_config,
    preferred_key_idx: int,
) -> dict:
    """Connect a persistent Live session using the sticky Gemini key policy."""
    global CURRENT_KEY_INDEX

    if not LIVE_ENABLED:
        raise RuntimeError("Gemini Live dang tat")
    if not API_KEYS:
        raise RuntimeError("Khong co GEMINI_API_KEY")

    key_idx: Optional[int] = preferred_key_idx
    last_exc: Optional[Exception] = None

    while key_idx is not None:
        if KEY_STATUS[key_idx] != "active":
            key_idx = next_active_key_after(key_idx)
            continue

        client = get_genai_client(key_idx)
        if client is None:
            mark_key_disabled(key_idx, "Client unavailable")
            key_idx = next_active_key_after(key_idx)
            continue

        for attempt in range(1, LIVE_SESSION_CONNECT_RETRIES + 1):
            session_cm = None
            try:
                session_cm = client.aio.live.connect(
                    model=LIVE_MODEL_NAME,
                    config=_live_config(safety_config),
                )
                session = await session_cm.__aenter__()

                CURRENT_KEY_INDEX = key_idx
                print(
                    f"[LIVE] Session san sang | model={LIVE_MODEL_NAME} | Key #{key_idx + 1}",
                    flush=True,
                )
                return {
                    "client": client,
                    "session_cm": session_cm,
                    "session": session,
                    "key_idx": key_idx,
                    "receive_task": None,
                    "turn_waiter": None,
                    "turn_active": False,
                    "turn_id": 0,
                    "raw_parts": [],
                    "input_transcript_parts": [],
                    "first_text_ms": None,
                    "turn_started": None,
                    "last_error": None,
                    "session_turns": 0,
                }
            except Exception as exc:
                last_exc = exc
                if session_cm is not None:
                    try:
                        await session_cm.__aexit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass

                kind = classify_gemini_error(exc)
                code = _error_code(exc)
                detail = str(exc).replace("\n", " ")[:220]
                if kind == "rotate":
                    mark_key_disabled(key_idx, detail)
                    nxt = next_active_key_after(key_idx)
                    reason_label = (
                        "WS 1011 QUOTA" if code == 1011
                        else ('HTTP ' + str(code) if code else 'key/quota error')
                    )
                    print(
                        f"[LIVE] Key #{key_idx + 1} bi vo hieu ({reason_label})",
                        flush=True,
                    )
                    key_idx = nxt
                    break

                if kind == "transient" and attempt < LIVE_SESSION_CONNECT_RETRIES:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                    continue

                raise RuntimeError(detail) from exc

    raise RuntimeError(str(last_exc) if last_exc else "Khong co Gemini Live key active")


async def live_receive_loop(live_handle: dict) -> None:
    """Continuously consume Live Transcribe events while ESP sends audio concurrently."""
    session = live_handle["session"]
    try:
        async for response in session.receive():
            content = getattr(response, "server_content", None)
            if content is None:
                continue

            input_transcription = getattr(content, "input_transcription", None)
            if input_transcription is not None:
                t = getattr(input_transcription, "text", None)
                if t and live_handle.get("turn_active"):
                    live_handle["input_transcript_parts"].append(str(t))
                    live_handle["last_transcript_at"] = time.monotonic()
                    if LIVE_TRANSCRIPT_LOG:
                        print(f"[LIVE INPUT] {str(t)!r}", flush=True)

            turn_complete = bool(getattr(content, "turn_complete", False))
            if turn_complete and live_handle.get("turn_active"):
                live_handle["turn_active"] = False
                waiter = live_handle.get("turn_waiter")
                if waiter and not waiter.done():
                    waiter.set_result(
                        {
                            "raw_text": "",
                            "input_transcript": "".join(live_handle["input_transcript_parts"]).strip(),
                            "first_text_ms": None,
                        }
                    )
            elif (
                live_handle.get("awaiting_audio_end")
                and live_handle.get("turn_active")
                and live_handle.get("input_transcript_parts")
                and (time.monotonic() - live_handle.get("last_transcript_at", 0.0) >= 0.25)
            ):
                # Some transcription responses may not expose turn_complete consistently.
                # Once the final transcript has been quiet for 250 ms after audio_stream_end,
                # it is safe to hand it to the normal Gemini text reasoning path.
                live_handle["turn_active"] = False
                waiter = live_handle.get("turn_waiter")
                if waiter and not waiter.done():
                    waiter.set_result(
                        {
                            "raw_text": "",
                            "input_transcript": "".join(live_handle["input_transcript_parts"]).strip(),
                            "first_text_ms": None,
                        }
                    )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        live_handle["last_error"] = exc
        waiter = live_handle.get("turn_waiter")
        if waiter and not waiter.done():
            waiter.set_exception(exc)


async def live_start_turn(
    live_handle: dict,
    history: deque,
    tof_distance_cm: Optional[float],
) -> None:
    """Start a realtime transcription window; ESP32 still controls the VAD window."""
    if live_handle.get("turn_active"):
        raise RuntimeError("Gemini Live dang co mot luot dang xu ly")

    live_handle["turn_id"] += 1
    live_handle["raw_parts"] = []
    live_handle["input_transcript_parts"] = []
    live_handle["first_text_ms"] = None
    live_handle["turn_started"] = time.monotonic()
    live_handle["awaiting_audio_end"] = False
    live_handle["last_transcript_at"] = time.monotonic()
    loop = asyncio.get_running_loop()
    live_handle["turn_waiter"] = loop.create_future()
    live_handle["turn_active"] = True

    # Save the sensor/time context locally for the text-reasoning step after transcription.
    live_handle["tof_distance_cm"] = tof_distance_cm


async def live_send_audio_chunk(live_handle: dict, pcm_chunk: bytes) -> None:
    if not live_handle.get("turn_active"):
        return
    chunk = _ensure_even_pcm_chunk(pcm_chunk)
    if not chunk:
        return
    await live_handle["session"].send_realtime_input(
        audio=types.Blob(
            data=chunk,
            mime_type=LIVE_INPUT_MIME,
        )
    )


async def live_end_turn(live_handle: dict, timeout_seconds: float = 5.0) -> dict:
    if not live_handle.get("turn_active"):
        waiter = live_handle.get("turn_waiter")
        if waiter and waiter.done() and not waiter.cancelled():
            return waiter.result()
        raise RuntimeError("Gemini Live khong co luot dang cho")

    live_handle["awaiting_audio_end"] = True
    try:
        await live_handle["session"].send_realtime_input(audio_stream_end=True)
        # The transcribe model normally emits the final input transcription and then completes.
        # Give that final transcript a short window to arrive; no second audio upload is needed.
        result = await asyncio.wait_for(
            live_handle["turn_waiter"],
            timeout=timeout_seconds,
        )
        return result
    finally:
        live_handle["turn_active"] = False
        live_handle["awaiting_audio_end"] = False


async def live_abort_turn(live_handle: Optional[dict]) -> None:
    if not live_handle:
        return
    live_handle["turn_active"] = False
    live_handle["awaiting_audio_end"] = False
    waiter = live_handle.get("turn_waiter")
    if waiter and not waiter.done():
        waiter.cancel()

def tof_to_context_value(value) -> Optional[float]:
    try:
        distance = float(value)
    except Exception:
        return None
    if not (0.0 < distance <= 2000.0):
        return None
    return round(distance, 1)


async def process_fallback_batch(
    pcm_bytes: bytes,
    safety_config,
    conversation_history: deque,
) -> tuple[str, str, dict]:
    if len(pcm_bytes) < 3200:
        raise RuntimeError("Audio qua ngan")
    wav_bytes = create_wav_bytes(pcm_bytes)
    return await ask_gemini_audio(
        wav_bytes,
        safety_config,
        conversation_history,
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global CURRENT_KEY_INDEX
    await websocket.accept()
    print("\n[WEBSOCKET] ESP32 da ket noi.", flush=True)
    pcm_buffer = bytearray()
    conversation_history = deque(maxlen=MEMORY_TURNS)
    latest_tof_cm: Optional[float] = None
    speech_active = False
    live_handle: Optional[dict] = None

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

    async def ensure_live() -> Optional[dict]:
        nonlocal live_handle
        if not LIVE_ENABLED:
            return None
        if live_handle and live_handle.get("last_error") is None:
            return live_handle
        await close_live_handle(live_handle)
        live_handle = None

        # Keep the same sticky key that batch Gemini uses. A Live session remains on this key
        # until it fails; only then does the shared key state rotate forward.
        preferred = CURRENT_KEY_INDEX if API_KEYS else 0
        try:
            live_handle = await open_live_handle(
                conversation_history,
                safety_config,
                preferred,
            )
            live_handle["receive_task"] = asyncio.create_task(
                live_receive_loop(live_handle)
            )
            return live_handle
        except Exception as exc:
            print(f"[LIVE] Khong khoi tao duoc: {str(exc)[:240]}", flush=True)
            live_handle = None
            return None

    try:
        # Establish the Live session while the robot is idle so the first audio chunk
        # does not have to wait for the Gemini WebSocket handshake.
        await ensure_live()

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            binary_data = message.get("bytes")
            if binary_data:
                if not speech_active:
                    continue

                # Always preserve the PCM locally so a transient Live failure can fall back
                # to the existing batch Gemini pipeline without asking the ESP32 to re-record.
                pcm_buffer.extend(binary_data)

                if live_handle and live_handle.get("last_error") is None:
                    try:
                        await live_send_audio_chunk(live_handle, binary_data)
                    except Exception as exc:
                        live_handle["last_error"] = exc
                        handle_live_key_error(live_handle["key_idx"], exc)
                        print(
                            f"[LIVE] Loi gui audio: {str(exc)[:220]}",
                            flush=True,
                        )
                continue

            text = (message.get("text") or "").strip()
            if not text:
                continue

            try:
                payload = json.loads(text)
            except Exception:
                payload = None

            # New ToF protocol from ESP32:
            # {"event":"tof","distance_cm":123.4}
            if isinstance(payload, dict) and payload.get("event") == "tof":
                latest_tof_cm = tof_to_context_value(payload.get("distance_cm"))
                continue

            if text == '{"event":"start_speech"}':
                speech_active = True
                pcm_buffer.clear()
                print(
                    f"[WEBSOCKET] ESP32 bat dau ghi am | ToF={latest_tof_cm if latest_tof_cm is not None else 'unknown'} cm",
                    flush=True,
                )

                live = await ensure_live()
                if live is not None:
                    try:
                        await live_start_turn(
                            live,
                            conversation_history,
                            latest_tof_cm,
                        )
                    except Exception as exc:
                        live["last_error"] = exc
                        handle_live_key_error(live["key_idx"], exc)
                        await live_abort_turn(live)
                        print(
                            f"[LIVE] Khong bat dau duoc luot realtime: {str(exc)[:220]}",
                            flush=True,
                        )
                continue

            if text != '{"event":"end_speech"}':
                continue

            speech_active = False
            pcm_size = len(pcm_buffer)
            print(
                f"[WEBSOCKET] ESP32 dung ghi am. PCM={pcm_size} bytes",
                flush=True,
            )

            if pcm_size < 3200:
                pcm_buffer.clear()
                await websocket.send_text(
                    json.dumps(
                        {"event": "tts_error", "message": "Audio qua ngan"}
                    )
                )
                if live_handle:
                    await live_abort_turn(live_handle)
                continue

            # Primary path: the same audio was already streamed chunk-by-chunk to Gemini Live.
            # We only wait for its final text here after signaling end-of-activity.
            user_memory = ""
            answer = ""
            action = {
                "type": "none",
                "emotion": "neutral",
                "direction": "none",
                "degrees": 0,
                "distance_cm": 0,
                "speed": "normal",
            }

            live_result = None
            if live_handle and live_handle.get("last_error") is None:
                try:
                    live_result = await live_end_turn(live_handle, timeout_seconds=5.0)
                    transcript = (live_result.get("input_transcript") or "").strip()
                    if not transcript:
                        raise RuntimeError("Gemini Live Transcribe khong tra transcript")

                    transcribe_ms = int(
                        (time.monotonic() - (live_handle.get("turn_started") or time.monotonic())) * 1000
                    )
                    print(
                        f"[LIVE] Transcribe hoan tat | chars={len(transcript)} | total={transcribe_ms} ms",
                        flush=True,
                    )
                    user_memory, answer, action = await ask_gemini_text(
                        transcript,
                        safety_config,
                        conversation_history,
                        latest_tof_cm,
                    )
                except Exception as live_exc:
                    live_handle["last_error"] = live_exc
                    handle_live_key_error(live_handle["key_idx"], live_exc)
                    print(
                        f"[LIVE] Transcribe/Reasoning loi -> fallback batch audio: {str(live_exc)[:240]}",
                        flush=True,
                    )
                    await live_abort_turn(live_handle)
                    live_result = None

            if live_result is None:
                try:
                    user_memory, answer, action = await process_fallback_batch(
                        bytes(pcm_buffer),
                        safety_config,
                        conversation_history,
                    )
                except Exception as exc:
                    pcm_buffer.clear()
                    await websocket.send_text(
                        json.dumps(
                            {
                                "event": "tts_error",
                                "message": str(exc)[:300],
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

            pcm_buffer.clear()
            cleaned = clean_text_for_tts(answer)
            print(f"[BUN DAU] {cleaned}", flush=True)
            if GEMINI_DEBUG_CHUNKS:
                print(f"[BUN DAU REPR] {cleaned!r}", flush=True)
            print(f"[MEMORY] {user_memory}", flush=True)
            print(
                f"[ACTION] type={action.get('type')} emotion={action.get('emotion')} "
                f"direction={action.get('direction')} degrees={action.get('degrees')} "
                f"distance_cm={action.get('distance_cm')} speed={action.get('speed')}",
                flush=True,
            )

            tts_start_payload = {
                "event": "tts_start",
                "emotion": action.get("emotion", "neutral"),
                "action_type": action.get("type", "none"),
                "action_direction": action.get("direction", "none"),
                "action_degrees": action.get("degrees", 0),
                "action_distance_cm": action.get("distance_cm", 0),
                "action_speed": action.get("speed", "normal"),
            }
            await websocket.send_text(
                json.dumps(tts_start_payload, ensure_ascii=False)
            )

            tts_started = time.monotonic()
            sent = 0
            used_voice = ""
            first_audio_ms = None
            try:
                if GEMINI_TTS_ENABLED:
                    key_idx = CURRENT_KEY_INDEX
                    try:
                        sent, first_audio_ms, used_voice = await asyncio.wait_for(
                            stream_gemini_tts_to_esp(
                                websocket,
                                cleaned,
                                key_idx,
                            ),
                            timeout=TTS_TIMEOUT_SECONDS
                            + max(5, int(len(cleaned) / 20)),
                        )
                    except Exception as tts_exc:
                        kind = classify_gemini_error(tts_exc)
                        detail = str(tts_exc).replace("\n", " ")[:240]
                        print(
                            f"[GEMINI TTS ERROR] Key #{key_idx + 1} | {detail}",
                            flush=True,
                        )
                        if (
                            kind == "rotate"
                            and key_idx < len(API_KEYS) - 1
                        ):
                            mark_key_disabled(key_idx, detail)
                            nxt = next_active_key_after(key_idx)
                            if nxt is not None:
                                CURRENT_KEY_INDEX = nxt
                                sent, first_audio_ms, used_voice = await asyncio.wait_for(
                                    stream_gemini_tts_to_esp(
                                        websocket,
                                        cleaned,
                                        nxt,
                                    ),
                                    timeout=TTS_TIMEOUT_SECONDS
                                    + max(5, int(len(cleaned) / 20)),
                                )
                        else:
                            raise
                else:
                    raise RuntimeError("Gemini TTS disabled")
            except Exception as exc:
                print(
                    f"[TTS] Gemini TTS that bai -> Edge-TTS fallback: "
                    f"{str(exc)[:260]}",
                    flush=True,
                )
                sent, _, used_voice = await send_edge_fallback_to_esp(
                    websocket,
                    cleaned,
                )
                first_audio_ms = int(
                    (time.monotonic() - tts_started) * 1000
                )

            tts_total_ms = int(
                (time.monotonic() - tts_started) * 1000
            )
            print(
                f"[PERF] TTS first_audio={first_audio_ms} ms | "
                f"total={tts_total_ms} ms | voice={used_voice}",
                flush=True,
            )

            conversation_history.append(
                {
                    "user_memory": user_memory,
                    "assistant_reply": cleaned,
                }
            )

            # If another Gemini path rotated the sticky key during this turn (for example
            # Gemini TTS quota/permission failure), discard the old Live session so the next
            # turn cannot continue using a disabled/non-current key.
            if live_handle is not None and live_handle.get("key_idx") != CURRENT_KEY_INDEX:
                await close_live_handle(live_handle)
                live_handle = None

            # Re-create the Live session after the configured number of turns so that
            # its retained server-side history does not grow indefinitely. The last 10
            # turns are re-seeded into the fresh session by history_config.
            if live_handle is not None:
                live_handle["session_turns"] = live_handle.get("session_turns", 0) + 1
                if live_handle["session_turns"] >= LIVE_HISTORY_RESET_TURNS:
                    print("[LIVE] Lam moi session sau 10 luot.", flush=True)
                    await close_live_handle(live_handle)
                    live_handle = None

            await websocket.send_text(json.dumps({"event": "tts_done"}))
            print(
                f"[WEBSOCKET] Da gui xong audio | PCM={sent} bytes | "
                f"TTS_total={tts_total_ms} ms | "
                f"audio_duration={int(sent * 1000 / PCM_BYTES_PER_SECOND)} ms",
                flush=True,
            )

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
    except Exception as exc:
        print(f"[WEBSOCKET ERROR] {exc}", flush=True)
    finally:
        pcm_buffer.clear()
        await close_live_handle(live_handle)

def create_wav_bytes(
    pcm_data: bytes,
    sample_rate: int = PCM_SAMPLE_RATE,
) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return bio.getvalue()

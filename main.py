# ================================================================================
# ROBOT BÚN ĐẬU SERVER - V4.10.7 - RECOVERY + MEMORY GUARD
# Phiên bản: 4.10.7
#
# - Gemini 3.8 Live là não chính cho hội thoại realtime và Google Search.
# - Gemini 3.6 Flash là não dự phòng khi Gemini 3.8 Live hết quota/lỗi.
# - Thêm Google Search grounding cho câu hỏi cần thông tin hiện tại.
# - Giữ giờ/ngày/thứ là dữ liệu nội bộ server; tuyệt đối không dùng Google Search cho nhóm này.
# - Giữ MEMORY / ACTION / REPLY / TTS / Edge-TTS fallback / ToF / sticky key.
# - Dùng Gemini 3.8 Live làm pipeline hội thoại/âm thanh chính.
# - Khi Live lỗi/hết quota, lượt đó chuyển sang Gemini 3.6 Flash để robot vẫn trò chuyện.
# - Gemini lỗi/chậm không đẩy ESP32 vào tts_error; robot nói thông báo bằng TTS.
# - Giữ retry fallback Gemini 3.6; giới hạn audio để tránh bộ đệm tăng vô hạn trên Render.
# - Chỉ đóng async HTTP client tạm sau tác vụ TTS/fallback; không đóng client của Live khi session còn hoạt động.
# - Phát hiện quota/rate-limit của Gemini Live và cooldown tạm thời; không giả định daily quota.
# - Khi hết quota AI, robot nói rõ đã hết lượt miễn phí và sẽ thử lại ngày mai.
# - Giữ bộ đếm local Search chỉ như safety guard cục bộ, không coi đó là quota Google thật.
# - KHÔNG dùng local STT/Whisper trong đường nghe fallback; giữ nguyên audio -> Gemini 3.6 như V4.8.
# - Khi Gemini 3.8 Live unavailable/quota, Gemini 3.6 nhận trực tiếp cùng WAV 16 kHz từ ESP32 để giữ độ chính xác nghe.
# - Giữ giờ/ngày/thứ nội bộ server khi Live cung cấp transcript; fallback 3.6 vẫn nhận trực tiếp audio và được cấp SERVER_TIME_NOW trong system prompt.
# - Thêm TTS quota guard: sau 429/quota của Gemini TTS, dùng Edge-TTS ngay để tránh chờ/retry dài.
# - Giới hạn output Gemini được ghi chú rõ tại MAX_OUTPUT_TOKENS bên dưới.
# ================================================================================

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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Keep Render Free / low-CPU memory footprint predictable.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
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

PRIMARY_MODEL_NAME = "gemini-3.8-live"
FALLBACK_MODEL_NAME = "gemini-3.6-flash"

# Gemini TTS quota guard.

_GEMINI_TTS_QUOTA_BLOCKED = False
_GEMINI_TTS_QUOTA_REASON = ""

# Gemini 3.6 Flash daily quota guard. This state is only a local cache of a real 429/quota response.
GEMINI_36_QUOTA_STATE_FILE = os.environ.get(
    "GEMINI_36_QUOTA_STATE_FILE",
    "gemini_36_quota_daily_state_v4_10_2.json",
).strip()
GEMINI_36_QUOTA_MESSAGE = os.environ.get(
    "GEMINI_36_QUOTA_MESSAGE",
    "Hôm nay mình đang hết lượt AI dự phòng miễn phí luôn rồi, mình sẽ thử lại vào ngày mai nhé.",
).strip()
_GEMINI_36_QUOTA_DATE: Optional[str] = None
_GEMINI_36_QUOTA_EXHAUSTED = False
_GEMINI_36_QUOTA_REASON = ""
# Fallback is deliberately fixed in code so an old Render GEMINI_MODEL variable
# cannot silently switch the backup brain to a paid/unsupported model.
GEMINI_FALLBACK_TIMEOUT_SECONDS = max(5.0, float(os.environ.get("GEMINI_FALLBACK_TIMEOUT_SECONDS", "30")))
# Each attempt has its own timeout so a transient timeout can be retried inside the total budget.
GEMINI_FALLBACK_ATTEMPT_TIMEOUT_SECONDS = max(
    5.0, float(os.environ.get("GEMINI_FALLBACK_ATTEMPT_TIMEOUT_SECONDS", "14"))
)
# Retry only transient service failures. 503/5xx/timeout must NOT disable a valid key.
GEMINI_FALLBACK_MAX_ATTEMPTS = max(1, int(os.environ.get("GEMINI_FALLBACK_MAX_ATTEMPTS", "2")))
GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS = max(
    0.2, float(os.environ.get("GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS", "0.8"))
)
MEMORY_TURNS = max(10, int(os.environ.get("MEMORY_TURNS", "10")))

# ================================================================================
# CẤU HÌNH INTERNET / THỜI GIAN / ĐỘ DÀI PHẢN HỒI
# ================================================================================
# Google Search grounding: Gemini sẽ tự quyết định có tìm web hay không.
# Dùng cho weather, giá xăng, giá vàng, tỷ giá, tin tức và dữ liệu hiện tại khác.
GOOGLE_SEARCH_ENABLED = (
    os.environ.get("GEMINI_WEB_SEARCH_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Local Search safety guard. This is NOT Google's authoritative quota.
# Gemini 3.8 Live quota is detected from Google's actual API error responses below.
GOOGLE_SEARCH_DAILY_LIMIT = max(1, int(os.environ.get("GEMINI_WEB_SEARCH_DAILY_LIMIT", "500")))
GOOGLE_SEARCH_WARNING_TEXT = os.environ.get(
    "GEMINI_WEB_SEARCH_EXHAUSTED_MESSAGE",
    "Hôm nay mình đã hết lượt tìm kiếm miễn phí trên mạng rồi, bạn đợi sang ngày mai mình tìm kiếm tiếp nha.",
).strip()
GOOGLE_SEARCH_COUNTER_FILE = os.environ.get(
    "GEMINI_WEB_SEARCH_COUNTER_FILE",
    "google_search_daily_usage_v4_9.json",
).strip()

_SEARCH_USAGE_DATE: Optional[str] = None
_SEARCH_USED_TODAY = 0
_SEARCH_RESERVED_TODAY = 0
_SEARCH_USAGE_LOCK = asyncio.Lock()

# Google may report Search-specific quota exhaustion separately from overall Live quota.
SEARCH_QUOTA_STATE_FILE = os.environ.get(
    "GEMINI_SEARCH_QUOTA_STATE_FILE",
    "google_search_quota_daily_state_v4_9.json",
).strip()
_SEARCH_QUOTA_DATE: Optional[str] = None
_SEARCH_QUOTA_EXHAUSTED = False
_SEARCH_QUOTA_LOCK = asyncio.Lock()

# Gemini Live guard.
# IMPORTANT: a generic Live WebSocket 1011 + "You exceeded your current quota"
# does NOT tell us that a daily quota was consumed or that it will reset at
# midnight Vietnam time. Google states that limits are project-level and that
# RPD resets at midnight Pacific; other limits can be RPM/TPM/concurrency.
# Therefore we only use a short in-process cooldown after a real quota/rate-limit
# response. We deliberately do NOT persist a "daily exhausted" state for Live.
LIVE_QUOTA_COOLDOWN_SECONDS = max(30, float(os.environ.get("GEMINI_LIVE_QUOTA_COOLDOWN_SECONDS", "600")))
LIVE_QUOTA_EXHAUSTED_MESSAGE = os.environ.get(
    "GEMINI_LIVE_QUOTA_EXHAUSTED_MESSAGE",
    "Gemini Live đang tạm thời không khả dụng do giới hạn quota/tốc độ. Mình sẽ thử lại sau ít phút nhé.",
).strip()
_LIVE_QUOTA_BLOCK_UNTIL = 0.0
_LIVE_QUOTA_LAST_REASON = ""

BUN_DAU_TIMEZONE = os.environ.get("BUN_DAU_TIMEZONE", "Asia/Ho_Chi_Minh").strip()
BUN_DAU_DEFAULT_LOCATION = os.environ.get(
    "BUN_DAU_DEFAULT_LOCATION",
    "Thành phố Hồ Chí Minh, Việt Nam",
).strip()


def local_today_str() -> str:
    """Return the current local calendar date used by all daily quota guards."""
    try:
        return datetime.now(ZoneInfo(BUN_DAU_TIMEZONE)).strftime("%Y-%m-%d")
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d")

# <<< ĐÂY LÀ GIỚI HẠN SỐ TOKEN OUTPUT TỐI ĐA CỦA GEMINI.
# Token không phải số từ cố định; tiếng Việt có thể dùng số token khác nhau cho cùng
# một lượng chữ. Tăng dòng này để cho phép Gemini trả lời dài hơn. Ví dụ: 768 -> 1200.
# Tăng giới hạn KHÔNG tự tạo thêm lượt nói; chỉ cho phép một lượt trả lời dài hơn.
MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "950"))

# Đây là giới hạn mềm về độ dài câu trả lời mà prompt yêu cầu. Nó không phải quota Gemini.
# Tăng lên nếu muốn robot nói dài hơn nữa; giá trị này không làm phát sinh thêm một lượt phát.
REPLY_MAX_SENTENCES = max(2, int(os.environ.get("BUN_DAU_REPLY_MAX_SENTENCES", "5")))

# Ổn định: mặc định không dùng Gemini Live đang gây lỗi modality/quota ở các bản trước.
# ESP32 vẫn có thể gửi PCM từng chunk lên server, nhưng server xử lý Gemini sau end_speech.
STABLE_BATCH_AUDIO_MODE = (
    os.environ.get("BUN_DAU_STABLE_BATCH_AUDIO_MODE", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Diagnostic logging for the current missing-character investigation.
# Set GEMINI_DEBUG_CHUNKS=false in Render after the issue is identified.
GEMINI_DEBUG_CHUNKS = (
    os.environ.get("GEMINI_DEBUG_CHUNKS", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Gemini Live is used for realtime audio input.
# 3.1 Flash Live keeps TEXT output available, which lets us preserve the current
# MEMORY/ACTION/REPLY protocol and the existing TTS pipeline without changing the ESP audio contract.
# V4.4: Gemini 3.8 Live is the default brain path. Current Live setup schema fixes applied.
LIVE_ENABLED = (
    os.environ.get("GEMINI_LIVE_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
LIVE_MODEL_NAME = PRIMARY_MODEL_NAME
LIVE_MAX_OUTPUT_TOKENS = int(
    os.environ.get("GEMINI_LIVE_MAX_OUTPUT_TOKENS", "384")
)
# Gemini 3.8 Live stable does not support thinking_level; kept only for backward compatibility.
LIVE_THINKING_LEVEL = os.environ.get(
    "GEMINI_LIVE_THINKING_LEVEL", "low"
).strip().lower()
# Legacy batch fallback switch intentionally ignored in V4.9.
GEMINI_BATCH_FALLBACK_ENABLED = False
LIVE_INPUT_MIME = "audio/pcm;rate=16000"
LIVE_SESSION_CONNECT_RETRIES = max(1, int(os.environ.get("GEMINI_LIVE_CONNECT_RETRIES", "2")))
LIVE_TRANSCRIPT_LOG = os.environ.get("GEMINI_LIVE_TRANSCRIPT_LOG", "false").strip().lower() in {"1", "true", "yes", "on"}
LIVE_INPUT_TRANSCRIPTION = os.environ.get("GEMINI_LIVE_INPUT_TRANSCRIPTION", "true").strip().lower() in {"1", "true", "yes", "on"}
LIVE_HISTORY_RESET_TURNS = max(10, int(os.environ.get("GEMINI_LIVE_HISTORY_RESET_TURNS", "10")))

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
- phải hiểu, phân biệt tên người, tên biệt danh chỉ người.

ĐỊNH DẠNG BẮT BUỘC:
Trả về đúng ba thẻ, theo đúng thứ tự, không thêm gì bên ngoài:
<MEMORY>tóm tắt rất ngắn nội dung người dùng vừa nói, tối đa 30 từ, giữ lại dữ kiện quan trọng</MEMORY>
<ACTION>{"type":"none","emotion":"neutral","direction":"none","degrees":0,"distance_cm":0,"speed":"normal"}</ACTION>
<REPLY>câu trả lời mà robot sẽ nói ra</REPLY>

QUY TẮC ACTION:
- ACTION là lệnh máy cho ESP32, tuyệt đối không đọc ACTION bằng loa.
- type chỉ được là: none, move, rotate, emotion.
- emotion chỉ được là: neutral, happy, excited, angry, sad, calm.
- direction chỉ được là: forward, backward, left, right, none.
- degrees là góc quay của robot, từ 0 đến 360.
- distance_cm là quãng đường tiến/lùi, từ 0 đến 30 cm cho một lệnh.
- speed chỉ được là: calm, normal, strong.
- Khi người dùng yêu cầu quay 90/180/360 độ, dùng type=rotate và điền degrees + direction.
- Khi người dùng yêu cầu tiến/lùi/trái/phải một đoạn, dùng type=move.
- Khi người dùng chỉ yêu cầu biểu cảm như “hãy làm biểu cảm tức giận”, dùng type=emotion và emotion=angry.
- Khi nội dung câu trả lời mang cảm xúc rõ ràng nhưng không có lệnh vật lý, dùng type=none và emotion tương ứng.
- Nếu không có lệnh hành động rõ ràng, dùng type=none.

Tính cách cốt lõi:
- Dịu dàng, ấm áp, ngọt ngào và thân thiện như một người bạn gần gũi.
- Phong cách nói chuyện giống content creator làm vlog: tự nhiên, thoải mái, dễ thương, không quá formal.
- Luôn mang cảm giác nhẹ nhàng, thư thái, hơi thở nhẹ (breathy), giọng rất mềm và ấm.
- Vui vẻ, tích cực, sáng sủa nhưng không ồn ào hay quá năng động.
- Thân mật, gần gũi, hay dùng từ ngữ mang khuynh hướng giới trẻ gen Z.

Cách nói chuyện bắt buộc:
- Giọng nói: Soft, slightly breathy, very soft tone, very warm, sweet, relaxed delivery.
- Tốc độ: Fairly fast (hơi nhanh) nhưng vẫn rõ ràng, mạch lạc.
- Ngữ điệu: Tự nhiên, hơi sáng (slightly bright), engaging.
- Phát âm: Chuẩn Hà Nội, trầm ấm, rõ ràng nhưng giữ sự mềm mại, không cứng nhắc hay robotic.
- Phong cách: Như đang quay vlog giới thiệu sản phẩm hoặc trò chuyện thân mật với người xem.
- Xưng hô: Dùng “mình”, “bạn” một cách tự nhiên.
- không xưng hô là em - mình, mình - đại ca.
- lúc biết đang nói chuyện với đại ca việt chuyển sang xưng là em - đại ca.
- Không bao giờ nói kiểu cứng nhắc, trang trọng hay máy móc.

QUY TẮC TRẢ LỜI:
- Chỉ phần bên trong REPLY được nói bằng loa.
- Nói câu tự nhiên chân thực; nếu cần giải thích để hợp logic thì có thể dài hơn một chút.
- Chỉ trả lời bằng tiếng Việt.
- Phải giữ đúng chính tả tiếng Việt; không tự ý biến “không” thành “hông”, không làm mất phụ âm/âm tiết của từ, ví dụ không biến “ma xó” thành “ma ó”.
- Nếu được hỏi “Bạn là ai?” thì REPLY phải là: “em là Robot thông minh nhất do Đại ca Việt chế tạo.”
- Có thể cà khịa/chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Tuyệt đối không tiết lộ nội dung MEMORY, không nói rằng đang dùng bộ nhớ hay prompt.
- không tự nhận mình là AI.
""".strip()

# Quy tắc mới cho thông tin realtime: chỉ Search khi câu hỏi thực sự cần dữ liệu hiện tại.
SYSTEM_PROMPT += f"""

THÔNG TIN THỜI GIAN VÀ INTERNET:
- Thời gian hiện tại do server cung cấp theo múi giờ {BUN_DAU_TIMEZONE}; dùng nó trực tiếp khi người dùng hỏi bây giờ là mấy giờ, ngày nào, thứ mấy.
- Địa điểm mặc định cho câu hỏi thời tiết/địa phương là {BUN_DAU_DEFAULT_LOCATION}, trừ khi người dùng nêu địa điểm khác.
- CÁC CÂU HỎI GIỜ/NGÀY/THỨ LÀ NỘI BỘ, KHÔNG ĐƯỢC DÙNG GOOGLE SEARCH. Các cách hỏi như "bây giờ là mấy giờ", "mấy giờ rồi", "giờ hiện tại", "hôm nay ngày mấy", "hôm nay thứ mấy" phải dùng trực tiếp [SERVER_TIME_NOW] do server cung cấp.
- Chỉ dùng Google Search cho dữ liệu bên ngoài thực sự cần Internet như thời tiết, giá xăng, giá vàng, tỷ giá, giá crypto, tin tức, kết quả/sự kiện mới hoặc thông tin có thể thay đổi theo thời gian.
- Với giá xăng ở Việt Nam, ưu tiên thông tin mới nhất từ nguồn chính thức/uy tín như cơ quan quản lý, Petrolimex hoặc nguồn thị trường có thời điểm cập nhật rõ ràng; nói rõ thời điểm nếu nguồn có nêu.
- Với giá vàng, ưu tiên giá SJC và nêu mua/bán cùng thời điểm nếu tìm được.
- Với thời tiết, ưu tiên dữ liệu mới nhất và nói rõ địa điểm/ngày khi cần.
- Không được bịa số liệu hiện tại. Nếu Search không tìm được dữ liệu đủ tin cậy, nói rõ chưa xác minh được thay vì đoán.
- REPLY thường ngắn gọn, nhưng có thể lên tới khoảng {REPLY_MAX_SENTENCES} câu khi câu hỏi cần nhiều thông tin; không kéo dài chỉ để cho dài.
""".strip()


def _get_local_time_context() -> str:
    try:
        now = datetime.now(ZoneInfo(BUN_DAU_TIMEZONE))
    except (ZoneInfoNotFoundError, ValueError):
        now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    return now.strftime("%Y-%m-%d %H:%M:%S UTC+07:00")


def _build_system_instruction(search_blocked: bool = False) -> str:
    # Chèn thời gian thực tại thời điểm gửi request; không dùng thời gian hard-code.
    extra = ""
    extra += (
        "\n[TIME_DATE_RULE] Câu hỏi về giờ, ngày, thứ là dữ liệu nội bộ. Không dùng Google Search. "
        "Luôn dùng trực tiếp [SERVER_TIME_NOW] được cung cấp bên dưới; không đi tìm trên Internet.\n"
    )
    if search_blocked and GOOGLE_SEARCH_ENABLED:
        extra = (
            "\n[INTERNET_SEARCH_STATUS] Hôm nay server đã hết lượt Google Search miễn phí theo bộ đếm an toàn cục bộ. "
            "Nếu người dùng hỏi dữ liệu cần Internet/thông tin hiện tại, KHÔNG được bịa hoặc đoán. "
            f"Trong REPLY phải nói đúng thông báo sau: {GOOGLE_SEARCH_WARNING_TEXT} "
            "Nếu câu hỏi không cần Internet thì vẫn trả lời bình thường.\n"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"[SERVER_TIME_NOW] {_get_local_time_context()}\n"
        f"[SERVER_DEFAULT_LOCATION] {BUN_DAU_DEFAULT_LOCATION}\n"
        f"{extra}"
    )


def _search_local_date() -> str:
    try:
        return datetime.now(ZoneInfo(BUN_DAU_TIMEZONE)).strftime("%Y-%m-%d")
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d")


def _load_search_usage_unlocked() -> None:
    global _SEARCH_USAGE_DATE, _SEARCH_USED_TODAY, _SEARCH_RESERVED_TODAY
    today = _search_local_date()
    if _SEARCH_USAGE_DATE == today:
        return

    used = 0
    try:
        if GOOGLE_SEARCH_COUNTER_FILE and os.path.exists(GOOGLE_SEARCH_COUNTER_FILE):
            with open(GOOGLE_SEARCH_COUNTER_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("date") == today:
                used = max(0, int(data.get("used", 0)))
    except Exception as exc:
        print(f"[SEARCH QUOTA] Khong doc duoc counter: {exc}", flush=True)

    _SEARCH_USAGE_DATE = today
    _SEARCH_USED_TODAY = min(used, GOOGLE_SEARCH_DAILY_LIMIT)
    _SEARCH_RESERVED_TODAY = 0


def _save_search_usage_unlocked() -> None:
    if not GOOGLE_SEARCH_COUNTER_FILE:
        return
    try:
        payload = {
            "date": _SEARCH_USAGE_DATE,
            "used": _SEARCH_USED_TODAY,
            "limit": GOOGLE_SEARCH_DAILY_LIMIT,
        }
        with open(GOOGLE_SEARCH_COUNTER_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[SEARCH QUOTA] Khong ghi duoc counter: {exc}", flush=True)


def _load_search_quota_state_unlocked() -> None:
    global _SEARCH_QUOTA_DATE, _SEARCH_QUOTA_EXHAUSTED
    today = _search_local_date()
    if _SEARCH_QUOTA_DATE == today:
        return

    exhausted = False
    try:
        if SEARCH_QUOTA_STATE_FILE and os.path.exists(SEARCH_QUOTA_STATE_FILE):
            with open(SEARCH_QUOTA_STATE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("date") == today:
                exhausted = bool(data.get("exhausted", False))
    except Exception as exc:
        print(f"[SEARCH QUOTA] Khong doc duoc quota state: {exc}", flush=True)

    _SEARCH_QUOTA_DATE = today
    _SEARCH_QUOTA_EXHAUSTED = exhausted


def _save_search_quota_state_unlocked() -> None:
    if not SEARCH_QUOTA_STATE_FILE:
        return
    try:
        payload = {
            "date": _SEARCH_QUOTA_DATE,
            "exhausted": _SEARCH_QUOTA_EXHAUSTED,
        }
        with open(SEARCH_QUOTA_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[SEARCH QUOTA] Khong ghi duoc quota state: {exc}", flush=True)


def is_search_quota_exhausted() -> bool:
    _load_search_quota_state_unlocked()
    return _SEARCH_QUOTA_EXHAUSTED


def mark_search_quota_exhausted(reason: str = "") -> None:
    global _SEARCH_QUOTA_EXHAUSTED
    _load_search_quota_state_unlocked()
    _SEARCH_QUOTA_EXHAUSTED = True
    _save_search_quota_state_unlocked()
    detail = (reason or "search quota exceeded").replace("\n", " ")[:220]
    print(
        f"[SEARCH QUOTA] Google reported Search quota exhausted -> disable Search until next local day | "
        f"date={_SEARCH_QUOTA_DATE} | reason={detail}",
        flush=True,
    )


def get_search_quota_status() -> dict:
    _load_search_quota_state_unlocked()
    return {
        "date": _SEARCH_QUOTA_DATE,
        "exhausted": _SEARCH_QUOTA_EXHAUSTED,
    }


def _refresh_live_cooldown_state() -> None:
    """Clear the short-lived Live cooldown when its timer expires."""
    global _LIVE_QUOTA_BLOCK_UNTIL, _LIVE_QUOTA_LAST_REASON
    if _LIVE_QUOTA_BLOCK_UNTIL > 0 and time.monotonic() >= _LIVE_QUOTA_BLOCK_UNTIL:
        _LIVE_QUOTA_BLOCK_UNTIL = 0.0
        _LIVE_QUOTA_LAST_REASON = ""


def is_live_quota_exhausted() -> bool:
    """Compatibility helper: True only while the temporary cooldown is active."""
    _refresh_live_cooldown_state()
    return _LIVE_QUOTA_BLOCK_UNTIL > time.monotonic()


def mark_live_quota_exhausted(reason: str = "") -> None:
    """Temporarily back off after a real Live quota/rate-limit response.

    This is intentionally NOT a daily quota counter. The API error alone does not
    identify which limit was exceeded or its reset time, so persisting a local-day
    lock would create false "hết quota đến ngày mai" states.
    """
    global _LIVE_QUOTA_BLOCK_UNTIL, _LIVE_QUOTA_LAST_REASON
    _LIVE_QUOTA_LAST_REASON = (reason or "quota/rate limit exceeded").replace("\n", " ")[:220]
    _LIVE_QUOTA_BLOCK_UNTIL = time.monotonic() + LIVE_QUOTA_COOLDOWN_SECONDS
    print(
        f"[LIVE QUOTA] Live tam thoi bi khoa {LIVE_QUOTA_COOLDOWN_SECONDS:.0f}s | "
        f"khong xac nhan daily quota | reason={_LIVE_QUOTA_LAST_REASON}",
        flush=True,
    )


def get_live_quota_status() -> dict:
    _refresh_live_cooldown_state()
    remaining = max(0, int(_LIVE_QUOTA_BLOCK_UNTIL - time.monotonic())) if _LIVE_QUOTA_BLOCK_UNTIL else 0
    return {
        "date": local_today_str(),
        "exhausted": False,
        "temporarily_blocked": remaining > 0,
        "retry_after_seconds": remaining,
        "reason": _LIVE_QUOTA_LAST_REASON,
        "message": LIVE_QUOTA_EXHAUSTED_MESSAGE,
    }

def _load_gemini_36_quota_unlocked() -> None:
    global _GEMINI_36_QUOTA_DATE, _GEMINI_36_QUOTA_EXHAUSTED, _GEMINI_36_QUOTA_REASON
    today = local_today_str()
    if _GEMINI_36_QUOTA_DATE == today:
        return

    # A key disabled only because Gemini 3.6 returned a daily quota/429 must be
    # eligible again on the next local day. Keep other permanent/auth failures disabled.
    for idx, reason in enumerate(KEY_FAILURE_REASON):
        if (
            KEY_STATUS[idx] == "disabled"
            and reason
            and reason.startswith("Gemini 3.6 quota/429:")
        ):
            KEY_STATUS[idx] = "active"
            KEY_FAILURE_REASON[idx] = None

    exhausted = False
    reason = ""
    if GEMINI_36_QUOTA_STATE_FILE and os.path.exists(GEMINI_36_QUOTA_STATE_FILE):
        try:
            with open(GEMINI_36_QUOTA_STATE_FILE, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            if state.get("date") == today:
                exhausted = bool(state.get("exhausted", False))
                reason = str(state.get("reason", ""))
        except Exception as exc:
            print(f"[GEMINI 3.6 QUOTA] Khong doc duoc state: {exc}", flush=True)
    _GEMINI_36_QUOTA_DATE = today
    _GEMINI_36_QUOTA_EXHAUSTED = exhausted
    _GEMINI_36_QUOTA_REASON = reason


def _save_gemini_36_quota_unlocked() -> None:
    if not GEMINI_36_QUOTA_STATE_FILE:
        return
    try:
        with open(GEMINI_36_QUOTA_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "date": _GEMINI_36_QUOTA_DATE,
                    "exhausted": _GEMINI_36_QUOTA_EXHAUSTED,
                    "reason": _GEMINI_36_QUOTA_REASON,
                },
                fh,
                ensure_ascii=False,
            )
    except Exception as exc:
        print(f"[GEMINI 3.6 QUOTA] Khong ghi duoc state: {exc}", flush=True)


def get_gemini_36_quota_status() -> dict:
    _load_gemini_36_quota_unlocked()
    return {
        "date": _GEMINI_36_QUOTA_DATE,
        "exhausted": _GEMINI_36_QUOTA_EXHAUSTED,
        "message": GEMINI_36_QUOTA_MESSAGE,
    }


def is_gemini_36_quota_exhausted() -> bool:
    _load_gemini_36_quota_unlocked()
    return _GEMINI_36_QUOTA_EXHAUSTED


def mark_gemini_36_quota_exhausted(reason: str = "") -> None:
    global _GEMINI_36_QUOTA_EXHAUSTED, _GEMINI_36_QUOTA_REASON
    _load_gemini_36_quota_unlocked()
    _GEMINI_36_QUOTA_EXHAUSTED = True
    _GEMINI_36_QUOTA_REASON = (reason or "quota exceeded").replace("\n", " ")[:220]
    _save_gemini_36_quota_unlocked()
    print(
        f"[GEMINI 3.6 QUOTA] Block 3.6 den ngay mai | date={_GEMINI_36_QUOTA_DATE} | "
        f"reason={_GEMINI_36_QUOTA_REASON}",
        flush=True,
    )


class RobotClientDisconnected(RuntimeError):
    """ESP32 has disconnected; TTS must not retry against a dead socket."""


def websocket_is_connected(websocket: WebSocket) -> bool:
    return (
        getattr(websocket, "application_state", None) == WebSocketState.CONNECTED
        and getattr(websocket, "client_state", None) == WebSocketState.CONNECTED
    )


async def safe_ws_send_text(websocket: WebSocket, payload: dict) -> None:
    if not websocket_is_connected(websocket):
        raise RobotClientDisconnected("ESP32 WebSocket da dong")
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
    except WebSocketDisconnect as exc:
        raise RobotClientDisconnected("ESP32 WebSocket da ngat") from exc
    except RuntimeError as exc:
        detail = str(exc).lower()
        if "close message" in detail or ("websocket" in detail and "disconnected" in detail):
            raise RobotClientDisconnected(str(exc)) from exc
        raise


async def reserve_search_budget() -> bool:
    global _SEARCH_RESERVED_TODAY
    async with _SEARCH_USAGE_LOCK:
        _load_search_usage_unlocked()
        if _SEARCH_USED_TODAY + _SEARCH_RESERVED_TODAY >= GOOGLE_SEARCH_DAILY_LIMIT:
            return False
        _SEARCH_RESERVED_TODAY += 1
        return True


async def finalize_search_budget(reserved: bool, did_search: bool) -> None:
    global _SEARCH_RESERVED_TODAY, _SEARCH_USED_TODAY
    if not reserved:
        return
    async with _SEARCH_USAGE_LOCK:
        _load_search_usage_unlocked()
        _SEARCH_RESERVED_TODAY = max(0, _SEARCH_RESERVED_TODAY - 1)
        if did_search:
            _SEARCH_USED_TODAY = min(
                GOOGLE_SEARCH_DAILY_LIMIT,
                _SEARCH_USED_TODAY + 1,
            )
            _save_search_usage_unlocked()
            remaining = max(0, GOOGLE_SEARCH_DAILY_LIMIT - _SEARCH_USED_TODAY)
            print(
                f"[SEARCH QUOTA] Da dung 1 grounded prompt | "
                f"today={_SEARCH_USED_TODAY}/{GOOGLE_SEARCH_DAILY_LIMIT} | remaining={remaining}",
                flush=True,
            )


def mark_search_quota_exhausted() -> None:
    global _SEARCH_USED_TODAY, _SEARCH_RESERVED_TODAY
    _load_search_usage_unlocked()
    _SEARCH_USED_TODAY = GOOGLE_SEARCH_DAILY_LIMIT
    _SEARCH_RESERVED_TODAY = 0
    _save_search_usage_unlocked()
    print(
        f"[SEARCH QUOTA] Google reported quota exhausted -> block Search until next local day | "
        f"today={_SEARCH_USED_TODAY}/{GOOGLE_SEARCH_DAILY_LIMIT}",
        flush=True,
    )


def get_search_usage_snapshot() -> dict:
    _load_search_usage_unlocked()
    return {
        "date": _SEARCH_USAGE_DATE,
        "used": _SEARCH_USED_TODAY,
        "reserved": _SEARCH_RESERVED_TODAY,
        "limit": GOOGLE_SEARCH_DAILY_LIMIT,
        "remaining_local_guard": max(
            0, GOOGLE_SEARCH_DAILY_LIMIT - _SEARCH_USED_TODAY - _SEARCH_RESERVED_TODAY
        ),
    }


SEARCH_INTENT_PATTERNS = [
    r"\bth[oơờ]i ti[eếệ]t\b",
    r"\bgi[aá] x[aă]ng\b",
    r"\bgi[aá] v[aà]ng\b",
    r"\bt[yỷ] gi[aá]\b",
    r"\bbitcoin\b",
    r"\bcrypto(?:currency)?\b",
    r"\btin t[uứ]c\b",
    r"\bm[oớ]i nh[aấ]t\b",
    r"\bk[eế]t qu[aả]\b",
    r"\bt[yỷ] s[oố]\b",
    r"\bl[iị]ch thi [dđ][aấ]u\b",
    r"\bl[iị]ch [dđ][aă]ng\b",
    r"\bs[uự] ki[eệ]n\b",
]

TIME_DATE_INTENT_PATTERNS = [
    r"\b(?:b[aâ]y|bao) gi[oờ] (?:l[aà]|r[oồ]i) m[aầ]y gi[oờ]\b",
    r"\bm[aấ]y gi[oờ] (?:r[oồ]i|hi[eệ]n t[aạ]i)\b",
    r"\bgi[oờ] hi[eệ]n t[aạ]i\b",
    r"\bb[aâ]y gi[oờ]\b",
    r"\bh[oô]m nay (?:l[aà] )?(?:ng[aà]y|th[uứ])\b",
    r"\bh[oô]m nay ng[aà]y m[aấ]y\b",
    r"\bh[oô]m nay th[uứ] m[aấ]y\b",
]

def looks_like_time_date_intent(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in TIME_DATE_INTENT_PATTERNS)


def build_local_time_date_reply(transcript: str) -> Optional[tuple[str, str, dict]]:
    """Return a deterministic local-time/date answer without Gemini Search.

    This is deliberately server-side so questions about clock/date/weekday never
    depend on network data, Search quota, or model interpretation.
    """
    if not looks_like_time_date_intent(transcript):
        return None

    try:
        now = datetime.now(ZoneInfo(BUN_DAU_TIMEZONE))
    except (ZoneInfoNotFoundError, ValueError):
        now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))

    weekday_names = [
        "thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm",
        "thứ Sáu", "thứ Bảy", "Chủ nhật",
    ]
    normalized = (transcript or "").strip().lower()

    asks_time = any(re.search(p, normalized, re.IGNORECASE) for p in [
        r"\b[bâ]y gi[oờ]\b",
        r"\bm[aấ]y gi[oờ]\b",
        r"\bgi[oờ] hi[eệ]n t[aạ]i\b",
    ])
    asks_date = any(re.search(p, normalized, re.IGNORECASE) for p in [
        r"\bh[oô]m nay ng[aà]y m[aấ]y\b",
        r"\bh[oô]m nay\s+l[aà]\s*ng[aà]y\b",
    ])
    asks_weekday = any(re.search(p, normalized, re.IGNORECASE) for p in [
        r"\bh[oô]m nay th[uứ] m[aấ]y\b",
        r"\bh[oô]m nay\s+l[aà]\s*th[uứ]\b",
    ])

    if asks_time and not asks_date and not asks_weekday:
        reply = f"Bây giờ là {now.hour:02d} giờ {now.minute:02d} phút nhé."
        memory = "Người dùng hỏi giờ hiện tại."
    elif asks_weekday and not asks_time:
        reply = f"Hôm nay là {weekday_names[now.weekday()]} nhé."
        memory = "Người dùng hỏi hôm nay là thứ mấy."
    elif asks_date and not asks_time:
        reply = f"Hôm nay là ngày {now.day:02d} tháng {now.month:02d} năm {now.year} nhé."
        memory = "Người dùng hỏi ngày hiện tại."
    else:
        reply = (
            f"Bây giờ là {now.hour:02d} giờ {now.minute:02d} phút, "
            f"{weekday_names[now.weekday()]}, ngày {now.day:02d} tháng {now.month:02d} năm {now.year} nhé."
        )
        memory = "Người dùng hỏi thông tin giờ và ngày hiện tại."

    action = {
        "type": "none",
        "emotion": "neutral",
        "direction": "none",
        "degrees": 0,
        "distance_cm": 0,
        "speed": "normal",
    }
    return memory, reply, action


def looks_like_search_intent(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return bool(normalized) and any(
        re.search(pattern, normalized, re.IGNORECASE) for pattern in SEARCH_INTENT_PATTERNS
    )


async def record_local_search_intent(transcript: str) -> bool:
    """Conservative local guard for likely current-data questions."""
    global _SEARCH_USED_TODAY
    if not GOOGLE_SEARCH_ENABLED or looks_like_time_date_intent(transcript) or not looks_like_search_intent(transcript):
        if looks_like_time_date_intent(transcript):
            print(f"[SEARCH GUARD] Bo qua Search quota cho cau hoi gio/ngay/thu | transcript={transcript!r}", flush=True)
        return False
    async with _SEARCH_USAGE_LOCK:
        _load_search_usage_unlocked()
        if _SEARCH_USED_TODAY >= GOOGLE_SEARCH_DAILY_LIMIT:
            return True
        _SEARCH_USED_TODAY = min(GOOGLE_SEARCH_DAILY_LIMIT, _SEARCH_USED_TODAY + 1)
        _save_search_usage_unlocked()
        remaining = max(0, GOOGLE_SEARCH_DAILY_LIMIT - _SEARCH_USED_TODAY)
        print(
            f"[SEARCH GUARD] Search-intent turn={_SEARCH_USED_TODAY}/{GOOGLE_SEARCH_DAILY_LIMIT} | remaining={remaining} | transcript={transcript!r}",
            flush=True,
        )
        return _SEARCH_USED_TODAY >= GOOGLE_SEARCH_DAILY_LIMIT


GOOGLE_SEARCH_TOOLS = (
    [types.Tool(google_search=types.GoogleSearch())]
    if GOOGLE_SEARCH_ENABLED
    else None
)


def get_genai_client(key_index: int):
    if not API_KEYS:
        return None
    return genai.Client(api_key=API_KEYS[key_index])


async def close_genai_async_client(client) -> None:
    """Close only the async HTTP client used by async requests; never close the sync client."""
    if client is None:
        return
    try:
        async_client = getattr(client, "aio", None)
        async_close = getattr(async_client, "aclose", None) if async_client is not None else None
        if callable(async_close):
            await async_close()
    except Exception as exc:
        print(f"[RESOURCE] Khong dong duoc Gemini async client: {str(exc)[:120]}", flush=True)


def process_memory_snapshot() -> dict:
    """Best-effort memory metrics for Render diagnostics; uses only standard library."""
    result = {"rss_mb": None, "peak_rss_mb": None}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    result["rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                    break
    except Exception:
        pass
    try:
        import resource
        result["peak_rss_mb"] = round(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024, 1)
    except Exception:
        pass
    return result


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
    search_quota_markers = (
        "google search quota",
        "google search grounding quota",
        "grounding quota exceeded",
        "web search quota",
        "search quota exceeded",
    )
    if any(x in text for x in search_quota_markers):
        return "search_quota"

    # Gemini Live can return a WebSocket 1011 while the actual payload says
    # the project has exceeded its current quota. Do NOT disable/rotate the key
    # for this case; it is a project-level quota condition.
    live_quota_markers = (
        "you exceeded your current quota",
        "exceeded your current quota",
        "gemini live quota",
        "live quota exceeded",
        "live_api_quota",
    )
    if any(x in text for x in live_quota_markers):
        return "live_quota"

    if code in (1011,) and ("quota" in text or "resource_exhausted" in text):
        return "live_quota"

    # HTTP status codes are more trustworthy than message keywords. In particular,
    # a 503 that happens to contain words like "rate limit" must remain transient
    # and must never disable a healthy API key.
    if code == 429:
        return "quota"
    if code in (500, 502, 503, 504):
        return "transient"
    if code in (401, 403):
        return "rotate"
    key_markers = (
        "api key not valid", "api_key_invalid", "invalid api key", "invalid_api_key",
        "unauthenticated", "permission denied", "permission_denied",
    )
    if any(x in text for x in key_markers):
        return "rotate"
    quota_markers = (
        "quota exceeded", "quota_exceeded", "resource_exhausted", "rate limit", "ratelimit",
    )
    if any(x in text for x in quota_markers):
        return "quota"
    transient_markers = (
        "service unavailable", "internal server error", "bad gateway", "gateway timeout",
        "deadline exceeded", "timeout", "timed out", "connection reset", "temporarily unavailable",
    )
    if any(x in text for x in transient_markers):
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

# Audio contract from ESP32: raw PCM16, mono, 16 kHz.
# The server must not resample, normalize, gate, or otherwise alter incoming mic audio.
AUDIO_INPUT_SAMPLE_RATE = 16000
AUDIO_INPUT_CHANNELS = 1
AUDIO_INPUT_BYTES_PER_SAMPLE = 2
AUDIO_MIN_TURN_BYTES = 3200
# 15 seconds of PCM16/16kHz mono is about 480 KB. 768 KB leaves margin but prevents runaway buffers.
MAX_AUDIO_INPUT_BYTES = max(512000, int(os.environ.get("MAX_AUDIO_INPUT_BYTES", "768000")))
TTS_CHUNK_SIZE = 2048
TTS_PREBUFFER_MS = max(0, int(os.environ.get("TTS_PREBUFFER_MS", "320")))
# Thời gian tối đa cho một lần tổng hợp TTS; tăng nếu cho phép câu trả lời rất dài.
TTS_TIMEOUT_SECONDS = float(os.environ.get("TTS_TIMEOUT_SECONDS", "40"))
TTS_RETRIES_PER_VOICE = max(1, int(os.environ.get("TTS_RETRIES_PER_VOICE", "2")))
EDGE_TTS_ENABLED = os.environ.get("EDGE_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural").strip()
EDGE_TTS_FALLBACK_VOICE = os.environ.get("EDGE_TTS_FALLBACK_VOICE", "vi-VN-NamMinhNeural").strip()
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+10%").strip()
GEMINI_TTS_ENABLED = os.environ.get("GEMINI_TTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview").strip()
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Aoede").strip()
GEMINI_TTS_LANGUAGE = os.environ.get("GEMINI_TTS_LANGUAGE", "vi-VN").strip()
GEMINI_TTS_STYLE = os.environ.get(
    "GEMINI_TTS_STYLE",
    "Speak in a gentle, natural Northern Vietnamese (Hanoi) accent. Soft, super warm and deep, sweet, friendly female voice suitable for vlogs and promotional content. Clear pronunciation, fairly fast pace, natural intonation, slightly bright and engaging, not robotic or overly formal. Sound like a young Vietnamese content creator introducing a product warmly. Slightly breathy, very soft tone, relaxed delivery,complemented by the deep, warm, and sultry voice of a young woman playfully feigning shyness with her boyfriend.",
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
        if not websocket_is_connected(websocket):
            raise RobotClientDisconnected("ESP32 WebSocket da dong trong luc gui audio")
        chunk = pcm[i:i + TTS_CHUNK_SIZE]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if not chunk:
            continue
        try:
            await websocket.send_bytes(chunk)
        except WebSocketDisconnect as exc:
            raise RobotClientDisconnected("ESP32 WebSocket da ngat trong luc gui audio") from exc
        except RuntimeError as exc:
            detail = str(exc).lower()
            if "close message" in detail or ("websocket" in detail and "disconnected" in detail):
                raise RobotClientDisconnected(str(exc)) from exc
            raise
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

    try:
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
    finally:
        await close_genai_async_client(client)


async def send_edge_fallback_to_esp(
    websocket: WebSocket,
    text: str,
) -> tuple[int, int, str]:
    if not EDGE_TTS_ENABLED:
        raise RuntimeError("Gemini TTS loi va Edge-TTS fallback dang tat")
    voices = [v for v in (EDGE_TTS_VOICE, EDGE_TTS_FALLBACK_VOICE) if v]
    last = None
    for voice in dict.fromkeys(voices):
        for attempt in range(1, TTS_RETRIES_PER_VOICE + 1):
            if not websocket_is_connected(websocket):
                raise RobotClientDisconnected("ESP32 WebSocket da dong truoc khi TTS fallback")
            started = time.monotonic()
            try:
                print(
                    f"[TTS] Edge fallback voice={voice} | lan {attempt}/{TTS_RETRIES_PER_VOICE}",
                    flush=True,
                )
                async with _tts_semaphore:
                    pcm = await asyncio.wait_for(
                        _edge_tts_pcm(text, voice),
                        timeout=TTS_TIMEOUT_SECONDS,
                    )
                if not websocket_is_connected(websocket):
                    raise RobotClientDisconnected("ESP32 WebSocket da dong sau khi synth Edge-TTS")
                state = {"next_deadline": time.monotonic()}
                sent = await _send_pcm_paced(websocket, pcm, state)
                elapsed = int((time.monotonic() - started) * 1000)
                print(
                    f"[TTS] Edge fallback thanh cong | voice={voice} | PCM={sent} bytes | synth={elapsed} ms",
                    flush=True,
                )
                return sent, elapsed, voice
            except RobotClientDisconnected:
                raise
            except Exception as exc:
                last = exc
                print(
                    f"[EDGE-TTS Error] voice={voice} | lan {attempt}: {str(exc)[:260]}",
                    flush=True,
                )
                if attempt < TTS_RETRIES_PER_VOICE:
                    await asyncio.sleep(0.2)
    raise RuntimeError(f"Tat ca TTS deu that bai: {last}")


# 3. HTTP
# ================================================================================
@app.get("/")
def read_root():
    # V4.10.1 hotfix: Local STT/Whisper was removed in V4.10, so the root
    # endpoint must not reference the old LOCAL_STT_* variables.
    return {
        "status": "Robot Bun Dau Server OK",
        "version": "4.10.7",
        "gemini_keys": len(API_KEYS),
        "local_stt_enabled": False,
        "local_stt_model": None,
        "local_stt_compute_type": None,
        "audio_fallback": "gemini-3.6-flash-direct-audio",
        "gemini_tts_quota_blocked": _GEMINI_TTS_QUOTA_BLOCKED,
        "gemini_36_quota": get_gemini_36_quota_status(),
        "current_gemini_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "tts_provider": "gemini-3.1-flash-tts-preview",
        "tts_voice": GEMINI_TTS_VOICE if GEMINI_TTS_ENABLED else EDGE_TTS_VOICE,
        "tts_fallback_voice": EDGE_TTS_VOICE,
        "tts_streaming": True,
        "tts_output": "PCM16 16kHz mono",
        # Gemini 3.8 Live does not use the legacy batch thinking/request-timeout fields.
        "gemini_live_thinking": "not_configured_for_3.8_live",
        "gemini_fallback_timeout_seconds": GEMINI_FALLBACK_TIMEOUT_SECONDS,
        "gemini_fallback_attempt_timeout_seconds": GEMINI_FALLBACK_ATTEMPT_TIMEOUT_SECONDS,
        "gemini_fallback_max_attempts": GEMINI_FALLBACK_MAX_ATTEMPTS,
        "gemini_fallback_retry_backoff_seconds": GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS,
        "memory_turns": MEMORY_TURNS,
        "gemini_debug_chunks": GEMINI_DEBUG_CHUNKS,
        "google_search_enabled": GOOGLE_SEARCH_ENABLED,
        "google_search_daily_limit": GOOGLE_SEARCH_DAILY_LIMIT,
        "google_search_usage": get_search_usage_snapshot(),
        "google_search_exhausted_warning": GOOGLE_SEARCH_WARNING_TEXT,
        "gemini_live_quota": get_live_quota_status(),
        "gemini_live_quota_warning": LIVE_QUOTA_EXHAUSTED_MESSAGE,
        "timezone": BUN_DAU_TIMEZONE,
        "default_location": BUN_DAU_DEFAULT_LOCATION,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reply_max_sentences": REPLY_MAX_SENTENCES,
        "stable_batch_audio_mode": STABLE_BATCH_AUDIO_MODE,
        "gemini_live_enabled": LIVE_ENABLED,
        "search_quota_status": get_search_quota_status(),
        "gemini_live_model": LIVE_MODEL_NAME,
        "gemini_fallback_model": FALLBACK_MODEL_NAME,
        "gemini_batch_fallback_enabled": GEMINI_BATCH_FALLBACK_ENABLED,
        "google_search_daily_usage": get_search_usage_snapshot(),
        "gemini_live_audio_input": "PCM16 16kHz mono realtime",
        "audio_input_preserve_mode": "exact_pcm16_no_resample_no_gate",
        "max_audio_input_bytes": MAX_AUDIO_INPUT_BYTES,
        "process_memory": process_memory_snapshot(),
        "tof_sensor": "VL53L0X over shared I2C",
        "robot_command_protocol": "v1",
        "robot_command_calibration": "ESP32-local timing calibration",
    }


@app.get("/healthz")
def healthz():
    # Minimal Render/monitoring health endpoint; never calls Gemini or STT.
    return {"status": "ok", "version": "4.10.7", "memory": process_memory_snapshot()}


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

    if str(action["type"]).lower() not in {"none", "move", "rotate", "emotion"}:
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
    search_state: Optional[dict] = None,
) -> None:
    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return

    candidate = candidates[0]

    if search_state is not None:
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        if grounding_metadata is not None:
            queries = getattr(grounding_metadata, "web_search_queries", None) or []
            chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            if queries or chunks:
                search_state["did_search"] = True
                search_state.setdefault("queries", set()).update(str(q) for q in queries)

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


async def ask_gemini_36_fallback_audio(
    wav_bytes: bytes,
    safety_config,
    history: deque,
) -> tuple[str, str, dict]:
    """Gemini 3.6 Flash fallback with bounded retry for transient service failures.

    503/500/502/504 and request timeouts are treated as temporary service problems:
    retry the same active key before trying another active key. These failures do NOT
    disable the key and do NOT mark the daily quota as exhausted.

    Real 429/quota/auth failures still follow the existing key rotation/quota guard.
    The exact WAV recorded by ESP32 is preserved; no local STT is introduced.
    """
    global CURRENT_KEY_INDEX

    if not API_KEYS:
        raise RuntimeError("Khong co GEMINI_API_KEY")
    if not wav_bytes:
        raise RuntimeError("Audio fallback rong")

    if is_gemini_36_quota_exhausted():
        raise RuntimeError("GEMINI_36_QUOTA_EXHAUSTED: daily guard")

    fallback_system = f"""{_build_system_instruction(search_blocked=True)}

[BACKUP_AI_MODE]
- Bạn đang là AI dự phòng bằng {FALLBACK_MODEL_NAME}.
- Trong chế độ này KHÔNG có Google Search và KHÔNG được sử dụng công cụ Internet.
- Với câu hỏi cần thông tin hiện tại, ví dụ thời tiết hôm nay, giá vàng/xăng/tỷ giá hiện tại, tin tức mới, kết quả mới hoặc dữ liệu có thể thay đổi, tuyệt đối không được đoán.
- Với câu hỏi realtime như vậy, trong REPLY hãy nói ngắn gọn rằng chức năng tìm kiếm mạng đang tạm không khả dụng và sẽ thử lại sau.
- Câu hỏi giờ/ngày/thứ phải dùng [SERVER_TIME_NOW] và KHÔNG được nói rằng cần Search.
- Các câu hỏi trò chuyện thông thường, kiến thức không phụ thuộc thời gian và lệnh robot vẫn phải xử lý bình thường.
- Giữ nguyên định dạng MEMORY/ACTION/REPLY và tính cách Bún Đậu."""

    key_idx: Optional[int] = CURRENT_KEY_INDEX if CURRENT_KEY_INDEX < len(API_KEYS) else 0
    last_exc: Optional[Exception] = None
    overall_deadline = time.monotonic() + GEMINI_FALLBACK_TIMEOUT_SECONDS

    while key_idx is not None:
        if KEY_STATUS[key_idx] != "active":
            key_idx = next_active_key_after(key_idx)
            continue

        client = get_genai_client(key_idx)
        if client is None:
            mark_key_disabled(key_idx, "Client unavailable")
            key_idx = next_active_key_after(key_idx)
            continue

        try:
            for attempt in range(1, GEMINI_FALLBACK_MAX_ATTEMPTS + 1):
                remaining = overall_deadline - time.monotonic()
                if remaining <= 0:
                    break

                started = time.monotonic()
                attempt_timeout = min(GEMINI_FALLBACK_ATTEMPT_TIMEOUT_SECONDS, remaining)
                try:
                    print(
                        f"[FALLBACK 3.6] Dung Key #{key_idx + 1} | model={FALLBACK_MODEL_NAME} | search=OFF | "
                        f"attempt={attempt}/{GEMINI_FALLBACK_MAX_ATTEMPTS} | timeout={attempt_timeout:.1f}s | "
                        f"total_remaining={remaining:.1f}s",
                        flush=True,
                    )

                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=FALLBACK_MODEL_NAME,
                            contents=make_gemini_contents(history, wav_bytes),
                            config=types.GenerateContentConfig(
                                system_instruction=fallback_system,
                                max_output_tokens=MAX_OUTPUT_TOKENS,
                                safety_settings=safety_config,
                            ),
                        ),
                        timeout=attempt_timeout,
                    )

                    candidates = getattr(response, "candidates", None) or []
                    parts: list[str] = []
                    finish_reason = "UNKNOWN"
                    finish_message = None
                    usage = getattr(response, "usage_metadata", None)

                    if candidates:
                        candidate = candidates[0]
                        finish_value = getattr(candidate, "finish_reason", None)
                        if finish_value is not None:
                            finish_reason = str(finish_value)
                        finish_message_value = getattr(candidate, "finish_message", None)
                        if finish_message_value:
                            finish_message = str(finish_message_value)
                        content = getattr(candidate, "content", None)
                        response_parts = getattr(content, "parts", None) if content is not None else None
                        for part in response_parts or []:
                            if getattr(part, "thought", False):
                                continue
                            part_text = getattr(part, "text", None)
                            if part_text:
                                parts.append(str(part_text))

                    raw_text = "".join(parts).strip()
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    print(
                        f"[FALLBACK 3.6] Hoan tat | chars={len(raw_text)} | "
                        f"finish_reason={finish_reason!r} | total={elapsed_ms} ms",
                        flush=True,
                    )
                    if usage:
                        print(
                            f"[FALLBACK 3.6] Usage | prompt={getattr(usage, 'prompt_token_count', None)} | "
                            f"output={getattr(usage, 'candidates_token_count', None)} | "
                            f"total={getattr(usage, 'total_token_count', None)}",
                            flush=True,
                        )

                    upper = finish_reason.upper()
                    if any(x in upper for x in ("MAX_TOKENS", "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "INCOMPLETE")):
                        raise RuntimeError(
                            f"Gemini 3.6 response khong hoan chinh: {finish_reason}"
                            + (f" | {finish_message}" if finish_message else "")
                        )
                    if not raw_text:
                        raise RuntimeError("Gemini 3.6 tra ve rong")

                    memory_text, reply_text, action = parse_tagged_response(raw_text)
                    if not reply_text:
                        raise RuntimeError("Gemini 3.6 tra ve rong REPLY")
                    if not memory_text:
                        memory_text = "Không trích xuất được tóm tắt lượt này."

                    CURRENT_KEY_INDEX = key_idx
                    print(f"[FALLBACK 3.6 REPLY] {reply_text!r}", flush=True)
                    return memory_text, reply_text, action

                except asyncio.TimeoutError as exc:
                    last_exc = RuntimeError(
                        f"Gemini 3.6 timeout sau {int(time.monotonic() - started)}s | attempt={attempt}"
                    )
                    if attempt < GEMINI_FALLBACK_MAX_ATTEMPTS:
                        remaining_after = overall_deadline - time.monotonic()
                        if remaining_after > 0:
                            delay = min(GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS, max(0.0, remaining_after))
                            print(
                                f"[FALLBACK 3.6 TRANSIENT] Timeout Key #{key_idx + 1} -> retry sau {delay:.1f}s | "
                                f"remaining={remaining_after:.1f}s",
                                flush=True,
                            )
                            if delay > 0:
                                await asyncio.sleep(delay)
                        continue
                    print(
                        f"[FALLBACK 3.6 TRANSIENT] Timeout Key #{key_idx + 1} het retry; khong vo hieu hoa key.",
                        flush=True,
                    )
                    nxt = next_active_key_after(key_idx)
                    if nxt is not None:
                        print(
                            f"[FALLBACK 3.6 TRANSIENT] Thu key active tiep theo #{nxt + 1}.",
                            flush=True,
                        )
                        key_idx = nxt
                    else:
                        key_idx = None
                    break

                except Exception as exc:
                    last_exc = exc
                    code = _error_code(exc)
                    detail = str(exc).replace("\n", " ")[:260]
                    kind = classify_gemini_error(exc)

                    quota_hit = (
                        code == 429
                        or (kind == "quota" and code not in (500, 502, 503, 504))
                    )
                    if quota_hit:
                        mark_key_disabled(key_idx, "Gemini 3.6 quota/429: " + detail)
                        nxt = next_active_key_after(key_idx)
                        if nxt is not None:
                            print(
                                f"[FALLBACK 3.6] Key #{key_idx + 1} het quota/429 -> thu Key #{nxt + 1}",
                                flush=True,
                            )
                            key_idx = nxt
                            break
                        mark_gemini_36_quota_exhausted(detail)
                        raise RuntimeError("GEMINI_36_QUOTA_EXHAUSTED: " + detail) from exc

                    if kind == "rotate":
                        mark_key_disabled(key_idx, detail)
                        nxt = next_active_key_after(key_idx)
                        if nxt is None:
                            key_idx = None
                            break
                        print(
                            f"[FALLBACK 3.6] Key #{key_idx + 1} khong dung duoc -> Key #{nxt + 1}",
                            flush=True,
                        )
                        key_idx = nxt
                        break

                    if kind == "transient":
                        if attempt < GEMINI_FALLBACK_MAX_ATTEMPTS:
                            remaining_after = overall_deadline - time.monotonic()
                            if remaining_after > 0:
                                delay = min(GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS, max(0.0, remaining_after))
                                print(
                                    f"[FALLBACK 3.6 TRANSIENT] Key #{key_idx + 1} HTTP={code or 'n/a'} -> retry "
                                    f"{attempt + 1}/{GEMINI_FALLBACK_MAX_ATTEMPTS} sau {delay:.1f}s | "
                                    f"error={detail[:180]}",
                                    flush=True,
                                )
                                if delay > 0:
                                    await asyncio.sleep(delay)
                            continue

                        print(
                            f"[FALLBACK 3.6 TRANSIENT] Key #{key_idx + 1} van con hoat dong; "
                            f"het retry | error={detail[:220]} | khong vo hieu hoa key",
                            flush=True,
                        )
                        nxt = next_active_key_after(key_idx)
                        if nxt is not None:
                            print(
                                f"[FALLBACK 3.6 TRANSIENT] Thu key active tiep theo #{nxt + 1}.",
                                flush=True,
                            )
                            key_idx = nxt
                        else:
                            key_idx = None
                        break

                    print(
                        f"[FALLBACK 3.6] Loi Key #{key_idx + 1}: {detail}",
                        flush=True,
                    )
                    key_idx = None
                    break
        finally:
            await close_genai_async_client(client)

    if last_exc:
        detail = str(last_exc)
        if (
            classify_gemini_error(last_exc) == "quota"
            or "quota" in detail.lower()
            or "resource_exhausted" in detail.lower()
        ):
            mark_gemini_36_quota_exhausted(detail)
            raise RuntimeError("GEMINI_36_QUOTA_EXHAUSTED: " + detail) from last_exc
    raise RuntimeError(
        str(last_exc) if last_exc else "Gemini 3.6 fallback khong hoan tat trong cua so retry"
    )


# ================================================================================
# 5. GEMINI LIVE + WEBSOCKET
# ================================================================================
def _live_config(search_blocked: bool = False):
    """Build a Gemini 3.8 Live setup compatible with the current Live API."""
    search_blocked = bool(search_blocked or is_search_quota_exhausted())
    # IMPORTANT for Gemini 3.8 Live:
    # - thinking_config/thinking_level is NOT supported on the stable 3.8 Live model.
    # - safety_settings is accepted by parts of the SDK type surface, but the current
    #   Live setup endpoint rejects it for this model with 1007/Unknown field safetySettings.
    #   Therefore we intentionally omit both fields here.
    search_tools = None if search_blocked else GOOGLE_SEARCH_TOOLS
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=_build_system_instruction(search_blocked=search_blocked),
        max_output_tokens=LIVE_MAX_OUTPUT_TOKENS,
        tools=search_tools,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            )
        ),
        input_audio_transcription={} if LIVE_INPUT_TRANSCRIPTION else None,
        history_config=types.HistoryConfig(
            initial_history_in_client_content=True,
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
    """Handle a Live-session error without spinning on project quota exhaustion."""
    global CURRENT_KEY_INDEX
    kind = classify_gemini_error(exc)
    detail = str(exc).replace("\n", " ")[:220]
    code = _error_code(exc)
    if kind == "live_quota":
        mark_live_quota_exhausted(detail)
        return None
    if kind == "search_quota":
        mark_search_quota_exhausted(detail)
        return None
    if kind != "rotate":
        return None
    mark_key_disabled(
        key_idx,
        detail,
    )
    nxt = next_active_key_after(key_idx)
    if nxt is not None:
        CURRENT_KEY_INDEX = nxt
    print(
        f"[LIVE] Key #{key_idx + 1} bi vo hieu ({'HTTP ' + str(code) if code else 'key/quota error'})",
        flush=True,
    )
    return nxt


async def open_live_handle(
    history: deque,
    preferred_key_idx: int,
    search_blocked: bool = False,
) -> dict:
    """Connect a persistent Live session using the sticky Gemini key policy."""
    global CURRENT_KEY_INDEX

    if not LIVE_ENABLED:
        raise RuntimeError("Gemini Live dang tat")
    if is_live_quota_exhausted():
        status = get_live_quota_status()
        raise RuntimeError(
            "GEMINI_LIVE_COOLDOWN: "
            f"retry_after={status['retry_after_seconds']}s"
        )
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
                    config=_live_config(search_blocked=search_blocked),
                )
                session = await session_cm.__aenter__()

                turns = _history_turns_for_live(history)
                if turns:
                    await session.send_client_content(
                        turns=turns,
                        turn_complete=False,
                    )

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
                    "search_blocked": search_blocked,
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
                if kind == "live_quota":
                    mark_live_quota_exhausted(detail)
                    raise RuntimeError(
                        "GEMINI_LIVE_COOLDOWN: "
                        f"retry_after={int(LIVE_QUOTA_COOLDOWN_SECONDS)}s"
                    ) from exc

                if kind == "rotate":
                    mark_key_disabled(key_idx, detail)
                    nxt = next_active_key_after(key_idx)
                    print(
                        f"[LIVE] Key #{key_idx + 1} bi vo hieu ({'HTTP ' + str(code) if code else 'key/quota error'})",
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
    """Continuously consume Live server events while ESP sends audio concurrently."""
    session = live_handle["session"]
    try:
        async for response in session.receive():
            content = getattr(response, "server_content", None)
            if content is None:
                continue

            input_transcription = getattr(content, "input_transcription", None)
            if input_transcription is not None:
                t = getattr(input_transcription, "text", None)
                if t:
                    live_handle["input_transcript_parts"].append(str(t))
                    if LIVE_TRANSCRIPT_LOG:
                        print(f"[LIVE INPUT] {str(t)!r}", flush=True)

            model_turn = getattr(content, "model_turn", None)
            if model_turn is not None:
                parts = getattr(model_turn, "parts", None) or []
                for part in parts:
                    if getattr(part, "thought", False):
                        continue
                    part_text = getattr(part, "text", None)
                    if not part_text:
                        continue
                    if live_handle.get("turn_active"):
                        if live_handle.get("first_text_ms") is None and live_handle.get("turn_started"):
                            live_handle["first_text_ms"] = int(
                                (time.monotonic() - live_handle["turn_started"]) * 1000
                            )
                        live_handle["raw_parts"].append(part_text)

            turn_complete = bool(getattr(content, "turn_complete", False))
            if turn_complete and live_handle.get("turn_active"):
                live_handle["turn_active"] = False
                waiter = live_handle.get("turn_waiter")
                if waiter and not waiter.done():
                    waiter.set_result(
                        {
                            "raw_text": "".join(live_handle["raw_parts"]).strip(),
                            "input_transcript": "".join(live_handle["input_transcript_parts"]).strip(),
                            "first_text_ms": live_handle.get("first_text_ms"),
                        }
                    )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        live_handle["last_error"] = exc
        error_kind = classify_gemini_error(exc)
        if error_kind == "live_quota":
            mark_live_quota_exhausted(str(exc))
        elif error_kind == "search_quota":
            mark_search_quota_exhausted(str(exc))
        waiter = live_handle.get("turn_waiter")
        if waiter and not waiter.done():
            waiter.set_exception(exc)


async def live_start_turn(
    live_handle: dict,
    history: deque,
    tof_distance_cm: Optional[float],
) -> None:
    """Begin an explicit realtime turn; ESP32 controls the actual VAD window."""
    if live_handle.get("turn_active"):
        raise RuntimeError("Gemini Live dang co mot luot dang xu ly")

    session = live_handle["session"]
    live_handle["turn_id"] += 1
    live_handle["raw_parts"] = []
    live_handle["input_transcript_parts"] = []
    live_handle["first_text_ms"] = None
    live_handle["turn_started"] = time.monotonic()
    loop = asyncio.get_running_loop()
    live_handle["turn_waiter"] = loop.create_future()
    live_handle["turn_active"] = True

    # ToF is injected as non-completing client content immediately before audio.
    # The sensor is context, not a user utterance that should trigger a response by itself.
    if tof_distance_cm is not None:
        await session.send_client_content(
            turns=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "[SENSOR_TOF]\n"
                                f"Khoang cach phia truoc hien tai: {tof_distance_cm:.1f} cm.\n"
                                "Chi dung thong tin cam bien nay lam boi canh cho luot noi sap toi; "
                                "khong tu y tra loi chi vi co tin cam bien.\n"
                                "[/SENSOR_TOF]"
                            )
                        }
                    ],
                }
            ],
            turn_complete=False,
        )

    await session.send_realtime_input(
        activity_start=types.ActivityStart()
    )


def _ensure_even_pcm_chunk(data: bytes) -> bytes:
    if len(data) % 2:
        return data[:-1]
    return data


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


async def live_end_turn(live_handle: dict, timeout_seconds: float = 20.0) -> dict:
    if not live_handle.get("turn_active"):
        waiter = live_handle.get("turn_waiter")
        if waiter and waiter.done() and not waiter.cancelled():
            return waiter.result()
        raise RuntimeError("Gemini Live khong co luot dang cho")

    try:
        await live_handle["session"].send_realtime_input(
            activity_end=types.ActivityEnd()
        )
        result = await asyncio.wait_for(
            live_handle["turn_waiter"],
            timeout=timeout_seconds,
        )
        return result
    finally:
        live_handle["turn_active"] = False


async def live_abort_turn(live_handle: Optional[dict]) -> None:
    if not live_handle:
        return
    live_handle["turn_active"] = False
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
    # Compatibility wrapper: V4.9 fallback is always Gemini 3.6 Flash and NEVER Search.
    if len(pcm_bytes) < 3200:
        raise RuntimeError("Audio qua ngan")
    wav_bytes = create_wav_bytes(pcm_bytes)
    return await ask_gemini_36_fallback_audio(
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
    audio_limit_reached = False
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

    async def ensure_live(search_blocked: bool = False) -> Optional[dict]:
        nonlocal live_handle
        if not LIVE_ENABLED:
            return None
        if (
            live_handle
            and live_handle.get("last_error") is None
            and bool(live_handle.get("search_blocked", False)) == bool(search_blocked)
        ):
            return live_handle
        await close_live_handle(live_handle)
        live_handle = None

        preferred = CURRENT_KEY_INDEX if API_KEYS else 0
        try:
            live_handle = await open_live_handle(
                conversation_history,
                preferred,
                search_blocked=search_blocked,
            )
            live_handle["receive_task"] = asyncio.create_task(
                live_receive_loop(live_handle)
            )
            return live_handle
        except Exception as exc:
            detail = str(exc)
            if detail.startswith("GEMINI_LIVE_COOLDOWN:"):
                print(
                    f"[LIVE] Dang trong cooldown tam thoi | {detail}",
                    flush=True,
                )
            else:
                print(f"[LIVE] Khong khoi tao duoc: {detail[:240]}", flush=True)
            live_handle = None
            return None

    try:
        if LIVE_ENABLED:
            live_quota_status = get_live_quota_status()
            print(
                f"[LIVE QUOTA] date={live_quota_status['date']} | "
                f"temporarily_blocked={live_quota_status['temporarily_blocked']} | "
                f"retry_after={live_quota_status['retry_after_seconds']}s",
                flush=True,
            )
            search_quota_status = get_search_quota_status()
            print(
                f"[SEARCH QUOTA] date={search_quota_status['date']} | exhausted={search_quota_status['exhausted']}",
                flush=True,
            )
            initial_search_blocked = (
                get_search_usage_snapshot()["used"] >= GOOGLE_SEARCH_DAILY_LIMIT
                or is_search_quota_exhausted()
            )
            if live_quota_status["exhausted"]:
                print(
                    f"[LIVE QUOTA] Live dang bi khoa den ngay mai | {LIVE_QUOTA_EXHAUSTED_MESSAGE}",
                    flush=True,
                )
            else:
                await ensure_live(search_blocked=initial_search_blocked)

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            binary_data = message.get("bytes")
            if binary_data:
                if not speech_active or audio_limit_reached:
                    continue

                # Keep PCM16 unchanged, but cap pathological turns to protect Render RAM.
                binary_data = sanitize_pcm16_chunk(binary_data)
                if not binary_data:
                    continue
                remaining_audio = MAX_AUDIO_INPUT_BYTES - len(pcm_buffer)
                if remaining_audio <= 0:
                    audio_limit_reached = True
                    print(f"[AUDIO GUARD] Vuot gioi han {MAX_AUDIO_INPUT_BYTES} bytes; bo qua chunk tiep theo.", flush=True)
                    continue
                if len(binary_data) > remaining_audio:
                    allowed = remaining_audio - (remaining_audio % 2)
                    binary_data = binary_data[:allowed]
                    audio_limit_reached = True
                    print(f"[AUDIO GUARD] Cham nguong {MAX_AUDIO_INPUT_BYTES} bytes; gioi han luot ghi am nay.", flush=True)
                    if not binary_data:
                        continue

                # Store one copy for fallback; the live path receives this same PCM chunk.
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
                audio_limit_reached = False
                pcm_buffer = bytearray()
                print(
                    f"[WEBSOCKET] ESP32 bat dau ghi am | ToF={latest_tof_cm if latest_tof_cm is not None else 'unknown'} cm",
                    flush=True,
                )

                if LIVE_ENABLED and not is_live_quota_exhausted():
                    search_blocked_now = (
                        get_search_usage_snapshot()["used"] >= GOOGLE_SEARCH_DAILY_LIMIT
                        or is_search_quota_exhausted()
                    )
                    live = await ensure_live(search_blocked=search_blocked_now)
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
                f"[WEBSOCKET] ESP32 dung ghi am. PCM={pcm_size} bytes | mode={'live' if LIVE_ENABLED else 'batch'}",
                flush=True,
            )
            audio_diag = diagnose_pcm16(pcm_buffer)
            print(
                f"[AUDIO RX] bytes={audio_diag['bytes']} | duration={audio_diag['duration_ms']} ms | "
                f"rms={audio_diag['rms']:.1f} | peak={audio_diag['peak']} | clipped={audio_diag['clipped']}",
                flush=True,
            )

            if audio_limit_reached:
                pcm_buffer = bytearray()
                if live_handle:
                    await live_abort_turn(live_handle)
                await safe_ws_send_text(websocket, {"event": "tts_error", "message": "Audio qua dai; hay noi gon hon mot chut."})
                audio_limit_reached = False
                continue

            if pcm_size < AUDIO_MIN_TURN_BYTES:
                pcm_buffer = bytearray()
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
                    live_result = await live_end_turn(live_handle, timeout_seconds=20.0)
                    raw_live = (live_result.get("raw_text") or "").strip()
                    if raw_live:
                        transcript = (live_result.get("input_transcript") or "").strip()
                        local_clock_result = build_local_time_date_reply(transcript)
                        if local_clock_result is not None:
                            user_memory, answer, action = local_clock_result
                            print(
                                f"[LOCAL TIME] Khong dung Gemini/Search cho cau hoi gio-ngay-thu | transcript={transcript!r} | answer={answer!r}",
                                flush=True,
                            )
                        else:
                            user_memory, answer, action = parse_tagged_response(raw_live)
                            if not answer:
                                raise RuntimeError("Gemini Live tra ve nhung khong co REPLY")
                            if not user_memory:
                                user_memory = "Không trích xuất được tóm tắt lượt này."

                        live_ms = int(
                            (time.monotonic() - (live_handle.get("turn_started") or time.monotonic())) * 1000
                        )
                        print(
                            f"[LIVE] Hoan tat turn | chars={len(raw_live)} | first_text={live_result.get('first_text_ms')} ms | total={live_ms} ms",
                            flush=True,
                        )
                    else:
                        raise RuntimeError("Gemini Live tra ve rong")
                except Exception as live_exc:
                    live_handle["last_error"] = live_exc
                    handle_live_key_error(live_handle["key_idx"], live_exc)
                    print(
                        f"[LIVE] Turn loi -> fallback Gemini 3.6: {str(live_exc)[:240]}",
                        flush=True,
                    )
                    await live_abort_turn(live_handle)
                    live_result = None

            if live_result is None:
                # IMPORTANT: restore the V4.8 audio path.
                # Do not transcribe with local Whisper first. Gemini 3.6 receives
                # the exact same PCM/WAV that the robot recorded, minimizing
                # recognition changes introduced by an extra STT layer.
                print(
                    "[FALLBACK 3.6] Live unavailable -> direct audio fallback (no local STT)",
                    flush=True,
                )
                try:
                    user_memory, answer, action = await ask_gemini_36_fallback_audio(
                        create_wav_bytes(pcm_buffer),
                        safety_config,
                        conversation_history,
                    )
                except Exception as fallback_exc:
                    err_text = str(fallback_exc)
                    if err_text.startswith("GEMINI_36_QUOTA_EXHAUSTED:"):
                        answer = "Hôm nay mình đang hết lượt AI dự phòng miễn phí luôn rồi, mình sẽ thử lại vào ngày mai nhé."
                        user_memory = "Gemini 3.8 Live và Gemini 3.6 Flash dự phòng đều vượt quota."
                    elif "timeout" in err_text.lower():
                        answer = "Xin lỗi, cả bộ não chính và bộ não dự phòng hôm nay đều phản hồi hơi chậm. Bạn thử lại mình nhé."
                        user_memory = "Gemini Live lỗi và Gemini 3.6 Flash dự phòng timeout."
                    else:
                        answer = "Xin lỗi, mình chưa nghe rõ câu này. Bạn nói lại mình một chút nhé."
                        user_memory = "Gemini Live không khả dụng và Gemini 3.6 audio fallback gặp lỗi."
                    action = {
                        "type": "none",
                        "emotion": "neutral",
                        "direction": "none",
                        "degrees": 0,
                        "distance_cm": 0,
                        "speed": "normal",
                    }
                    print(f"[FALLBACK AUDIO ERROR] {err_text[:260]}", flush=True)
                pcm_buffer = bytearray()


            pcm_buffer = bytearray()
            if live_result is not None:
                transcript = (live_result.get("input_transcript") or "").strip()
                guard_exhausted = await record_local_search_intent(transcript)
                if guard_exhausted and live_handle is not None:
                    print(
                        "[SEARCH GUARD] Đã đạt local daily guard -> lượt tiếp theo sẽ chạy không có Google Search.",
                        flush=True,
                    )
                    await close_live_handle(live_handle)
                    live_handle = None
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
            tts_failed = False
            try:
                global _GEMINI_TTS_QUOTA_BLOCKED, _GEMINI_TTS_QUOTA_REASON
                if GEMINI_TTS_ENABLED and not _GEMINI_TTS_QUOTA_BLOCKED:
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
                    except RobotClientDisconnected:
                        raise
                    except Exception as tts_exc:
                        kind = classify_gemini_error(tts_exc)
                        detail = str(tts_exc).replace("\n", " ")[:240]
                        print(
                            f"[GEMINI TTS ERROR] Key #{key_idx + 1} | {detail}",
                            flush=True,
                        )
                        if kind in {"quota", "live_quota", "search_quota", "rotate"} or "too many requests" in detail.lower():
                            _GEMINI_TTS_QUOTA_BLOCKED = True
                            _GEMINI_TTS_QUOTA_REASON = detail
                            print(
                                "[GEMINI TTS] Quota/429 detected -> khoa Gemini TTS cho phien server, dung Edge-TTS ngay.",
                                flush=True,
                            )
                            raise RuntimeError("Gemini TTS quota exhausted") from tts_exc
                        raise
                else:
                    if _GEMINI_TTS_QUOTA_BLOCKED:
                        print("[GEMINI TTS] Dang bi khoa do quota -> Edge-TTS ngay.", flush=True)
                    raise RuntimeError("Gemini TTS disabled or quota blocked")
            except RobotClientDisconnected:
                print("[WEBSOCKET] ESP32 da ngat trong luc TTS; dung moi retry.", flush=True)
                raise
            except Exception as exc:
                print(
                    f"[TTS] Gemini TTS that bai -> Edge-TTS fallback: {str(exc)[:260]}",
                    flush=True,
                )
                try:
                    sent, _, used_voice = await send_edge_fallback_to_esp(
                        websocket,
                        cleaned,
                    )
                    first_audio_ms = int((time.monotonic() - tts_started) * 1000)
                except RobotClientDisconnected:
                    print("[WEBSOCKET] ESP32 da ngat trong Edge-TTS; dung moi retry.", flush=True)
                    raise
                except Exception as edge_exc:
                    tts_failed = True
                    print(
                        f"[TTS FATAL] Gemini + Edge-TTS deu that bai, giu WebSocket song: {str(edge_exc)[:260]}",
                        flush=True,
                    )

            tts_total_ms = int((time.monotonic() - tts_started) * 1000)
            if tts_failed:
                if websocket_is_connected(websocket):
                    try:
                        await safe_ws_send_text(
                            websocket,
                            {"event": "tts_done", "tts_failed": True, "audio_bytes": 0},
                        )
                    except RobotClientDisconnected:
                        print("[WEBSOCKET] ESP32 da ngat truoc khi gui tts_done sau TTS failure.", flush=True)
                        raise
                conversation_history.append(
                    {"user_memory": user_memory, "assistant_reply": cleaned}
                )
                continue
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

            await safe_ws_send_text(websocket, {"event": "tts_done"})
            print(
                f"[WEBSOCKET] Da gui xong audio | PCM={sent} bytes | "
                f"TTS_total={tts_total_ms} ms | "
                f"audio_duration={int(sent * 1000 / PCM_BYTES_PER_SECOND)} ms",
                flush=True,
            )

    except (WebSocketDisconnect, RobotClientDisconnected):
        print("[WEBSOCKET] ESP32 ngat ket noi.", flush=True)
    except Exception as exc:
        print(f"[WEBSOCKET ERROR] {exc}", flush=True)
    finally:
        pcm_buffer = bytearray()
        await close_live_handle(live_handle)

def sanitize_pcm16_chunk(data: bytes) -> bytes:
    """Keep the ESP32 PCM byte stream intact; only drop one trailing invalid byte."""
    if not data:
        return b""
    if len(data) % AUDIO_INPUT_BYTES_PER_SAMPLE != 0:
        print(
            f"[AUDIO RX] Chunk co {len(data)} bytes le -> bo 1 byte le de giu PCM16 hop le.",
            flush=True,
        )
        return data[:-1]
    return data


def diagnose_pcm16(pcm_data: bytes) -> dict:
    """Measure received PCM without modifying it, for mic/Gemini path diagnosis."""
    data = sanitize_pcm16_chunk(pcm_data)
    if not data:
        return {"bytes": 0, "duration_ms": 0, "rms": 0.0, "peak": 0, "clipped": 0}

    import array
    samples = array.array("h")
    samples.frombytes(data)
    if samples.itemsize != 2:
        return {"bytes": len(data), "duration_ms": int(len(data) * 1000 / PCM_BYTES_PER_SECOND), "rms": 0.0, "peak": 0, "clipped": 0}

    if samples and __import__("sys").byteorder != "little":
        samples.byteswap()

    n = len(samples)
    if n == 0:
        return {"bytes": len(data), "duration_ms": 0, "rms": 0.0, "peak": 0, "clipped": 0}

    total_sq = 0.0
    peak = 0
    clipped = 0
    for sample in samples:
        value = int(sample)
        magnitude = abs(value)
        if magnitude > peak:
            peak = magnitude
        if magnitude >= 32767:
            clipped += 1
        total_sq += float(value * value)

    return {
        "bytes": len(data),
        "duration_ms": int(n * 1000 / AUDIO_INPUT_SAMPLE_RATE),
        "rms": (total_sq / n) ** 0.5,
        "peak": peak,
        "clipped": clipped,
    }


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

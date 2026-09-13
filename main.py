import asyncio
import io
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
# 1. CẤU HÌNH API KEY VÀ MODEL GEMINI
# ============================================================================== 
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [
    k.strip(' "\'\t\r\n')
    for k in RAW_KEYS.split(",")
    if k.strip(' "\'\t\r\n')
]

CURRENT_KEY_INDEX = 0

# Trạng thái key trong suốt vòng đời server.
#
# active: có thể dùng
# disabled: key đã xác định là không thể dùng (401/403/429 hoặc key không hợp lệ)
#
# Quan trọng: failover chỉ đi về phía trước, KHÔNG quay vòng về key cũ.
# Ví dụ: key 1 chết -> key 2 thành công -> các request sau bắt đầu từ key 2.
# Nếu key 2 chết -> chuyển key 3, không quay lại key 1.
KEY_STATUS = ["active"] * len(API_KEYS)
KEY_FAILURE_REASON = [None] * len(API_KEYS)

# Giữ nguyên model hiện tại của dự án.
MODEL_NAME = "gemini-3.6-flash"

TTS_VOICE = "vi-VN-HoaiMyNeural"
TTS_RATE = "+10%"

# PCM output mà ESP32 đang dùng: 16 kHz / 16-bit / mono.
PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 2
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_BYTES_PER_SAMPLE

# Mỗi frame 2048 bytes = 64 ms audio.
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
    selected_key = API_KEYS[key_index % len(API_KEYS)]
    return genai.Client(api_key=selected_key)


def _error_code(exc) -> Optional[int]:
    """Cố gắng lấy HTTP/status code từ exception của google-genai."""
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None and str(value).isdigit():
                return int(value)
        except Exception:
            pass
    return None


def classify_gemini_error(exc) -> str:
    """
    Phân loại lỗi để quyết định có bỏ qua key hay không.

    rotate: lỗi gắn với key/quota -> chuyển key tiếp theo.
    transient: lỗi máy chủ/mạng -> không đánh dấu key hỏng.
    fatal: lỗi cấu hình/request -> không xoay key vì xoay cũng không giải quyết được.
    """
    code = _error_code(exc)
    text = str(exc).lower()

    if code in (401, 403):
        return "rotate"

    if code == 429:
        return "rotate"

    key_error_markers = (
        "api key not valid",
        "api_key_invalid",
        "invalid api key",
        "invalid_api_key",
        "unauthenticated",
        "permission denied",
        "permission_denied",
        "authentication",
        "quota exceeded",
        "quota_exceeded",
        "resource_exhausted",
        "rate limit",
        "ratelimit",
    )
    if any(marker in text for marker in key_error_markers):
        return "rotate"

    transient_markers = (
        "503",
        "service unavailable",
        "unavailable",
        "500",
        "internal server error",
        "502",
        "504",
        "deadline exceeded",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
    )
    if code in (500, 502, 503, 504) or any(marker in text for marker in transient_markers):
        return "transient"

    return "fatal"


def next_active_key_after(start_index: int) -> Optional[int]:
    """
    Tìm key hoạt động tiếp theo theo chiều tăng index.
    Không bao giờ quay vòng về key đứng trước start_index.
    """
    for idx in range(start_index + 1, len(API_KEYS)):
        if KEY_STATUS[idx] == "active":
            return idx
    return None


def disable_key(key_idx: int, reason: str) -> None:
    KEY_STATUS[key_idx] = "disabled"
    KEY_FAILURE_REASON[key_idx] = reason[:240]
    print(
        f"[GEMINI KEY] Key #{key_idx + 1} bị vô hiệu trong phiên server: {reason[:180]}",
        flush=True,
    )


def key_status_summary() -> str:
    parts = []
    for idx, status in enumerate(KEY_STATUS, start=1):
        if status == "active":
            parts.append(f"#{idx}=ACTIVE")
        else:
            reason = KEY_FAILURE_REASON[idx - 1] or "disabled"
            parts.append(f"#{idx}=DISABLED({reason[:45]})")
    return ", ".join(parts)


# ============================================================================== 
# 2. AUDIO / TEXT HELPERS
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
    if message is None:
        payload = {"event": event}
    else:
        payload = {"event": event, "message": message}

    # Không thêm dependency JSON; WebSocket/FastAPI sẽ encode dict? Không.
    # Dùng module json để đảm bảo payload hợp lệ.
    import json
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


async def synthesize_sentence_to_pcm(sentence_text: str, target_sample_rate: int = PCM_SAMPLE_RATE) -> Optional[bytes]:
    """Edge-TTS cho một câu và decode thành PCM16 mono."""
    clean_txt = clean_text_for_tts(sentence_text)

    if not clean_txt or not re.search(r"\w", clean_txt):
        return b""

    communicate = edge_tts.Communicate(
        clean_txt,
        voice=TTS_VOICE,
        rate=TTS_RATE,
    )

    mp3_buffer = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.extend(chunk["data"])

    if not mp3_buffer:
        return None

    def decode_mp3() -> bytes:
        decoded = miniaudio.decode(
            bytes(mp3_buffer),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=PCM_CHANNELS,
            sample_rate=target_sample_rate,
        )
        return decoded.samples.tobytes()

    pcm_bytes = await asyncio.to_thread(decode_mp3)
    return pcm_bytes or None


async def stream_pcm_realtime(websocket: WebSocket, pcm_bytes: bytes) -> None:
    """Gửi PCM ở đúng tốc độ audio thay vì 1024 bytes/25 ms gây lệch tốc độ."""
    if not pcm_bytes:
        return

    for offset in range(0, len(pcm_bytes), TTS_CHUNK_SIZE):
        chunk = pcm_bytes[offset:offset + TTS_CHUNK_SIZE]
        start = time.monotonic()

        await websocket.send_bytes(chunk)

        # Thời lượng thật của chunk. Sleep phần thời gian còn lại để tổng chu kỳ
        # gần đúng realtime, không đẩy nhanh hơn 28% như bản cũ.
        chunk_duration = len(chunk) / PCM_BYTES_PER_SECOND
        elapsed = time.monotonic() - start
        remaining = chunk_duration - elapsed

        if remaining > 0:
            await asyncio.sleep(remaining)


async def text_to_pcm_sentence_with_retry(
    sentence_text: str,
    websocket: WebSocket,
    sentence_index: int,
    total_sentences: int,
) -> bool:
    """TTS tối đa 2 lần cho một câu. Không giả tts_done khi thất bại."""
    for attempt in range(1, 3):
        try:
            clean_txt = clean_text_for_tts(sentence_text)
            print(
                f"[TTS] Cau {sentence_index}/{total_sentences}, lan thu {attempt}: {clean_txt!r}",
                flush=True,
            )

            pcm_bytes = await synthesize_sentence_to_pcm(clean_txt)

            if pcm_bytes is None:
                raise RuntimeError("Edge-TTS khong tra ve audio")

            if not pcm_bytes:
                raise RuntimeError("PCM rong sau khi decode")

            print(
                f"[TTS] Cau {sentence_index}/{total_sentences}: {len(pcm_bytes)} bytes PCM",
                flush=True,
            )

            await stream_pcm_realtime(websocket, pcm_bytes)
            return True

        except Exception as exc:
            print(
                f"[EDGE-TTS Error] Cau {sentence_index}, lan {attempt}: {exc}",
                flush=True,
            )
            if attempt < 2:
                await asyncio.sleep(0.15)

    return False


# ============================================================================== 
# 3. HTTP
# ============================================================================== 
@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu WebSocket Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key_index": CURRENT_KEY_INDEX,
        "current_active_key": CURRENT_KEY_INDEX + 1 if API_KEYS else None,
        "key_status": key_status_summary(),
        "pcm_format": "PCM16 16kHz mono",
    }


# ============================================================================== 
# 4. WEBSOCKET
# ============================================================================== 
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global CURRENT_KEY_INDEX

    await websocket.accept()
    print("\n[WEBSOCKET] ESP32 đã kết nối thành công!", flush=True)

    pcm_buffer = bytearray()

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError:
                print("[WEBSOCKET] ESP32 đã ngắt kết nối (Socket Closed).", flush=True)
                break

            if message.get("type") == "websocket.disconnect":
                print("[WEBSOCKET] ESP32 đã gửi tín hiệu ngắt kết nối.", flush=True)
                break

            # ------------------------------------------------------------------
            # A. PCM MICROPHONE
            # ------------------------------------------------------------------
            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])
                continue

            # ------------------------------------------------------------------
            # B. CONTROL TEXT
            # ------------------------------------------------------------------
            if "text" not in message or not message["text"]:
                continue

            msg_text = message["text"].strip()

            if msg_text == '{"event":"start_speech"}':
                pcm_buffer.clear()
                print("[WEBSOCKET] -> ESP32 bắt đầu ghi âm...", flush=True)
                continue

            if msg_text != '{"event":"end_speech"}':
                print(f"[WEBSOCKET] Text khong xac dinh: {msg_text}", flush=True)
                continue

            # ------------------------------------------------------------------
            # C. END SPEECH -> GEMINI -> TTS
            # ------------------------------------------------------------------
            pcm_size = len(pcm_buffer)
            print(
                f"[WEBSOCKET] -> ESP32 dừng ghi âm. Dung lượng PCM: {pcm_size} bytes",
                flush=True,
            )

            if pcm_size < 3200:
                print("[WEBSOCKET] Âm thanh quá ngắn, bỏ qua.", flush=True)
                pcm_buffer.clear()
                await send_event(
                    websocket,
                    "tts_error",
                    "Audio quá ngắn hoặc không nghe thấy tiếng nói.",
                )
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

            # ------------------------------------------------------------------
            # D. GEMINI KEY FAILOVER MỘT CHIỀU + NHẬN ĐỦ RESPONSE
            # ------------------------------------------------------------------
            total_keys = len(API_KEYS)

            if total_keys == 0:
                print("[GEMINI ERROR] Không tìm thấy GEMINI_API_KEY nào!", flush=True)
                await send_event(websocket, "tts_error", "No API Keys")
                continue

            # Nếu CURRENT_KEY_INDEX đã bị disable bởi một lượt trước, tìm key active
            # gần nhất về phía trước thay vì quay lại key cũ.
            if CURRENT_KEY_INDEX >= total_keys or KEY_STATUS[CURRENT_KEY_INDEX] != "active":
                # Chỉ tìm về phía trước, không quay lại key cũ đã bỏ qua.
                replacement = next_active_key_after(CURRENT_KEY_INDEX)
                if replacement is not None:
                    CURRENT_KEY_INDEX = replacement

            key_idx = CURRENT_KEY_INDEX
            full_response_text = ""
            finish_reason = None
            finish_message = None
            usage_summary = None
            gemini_ok = False

            while key_idx is not None and key_idx < total_keys:
                if KEY_STATUS[key_idx] != "active":
                    key_idx = next_active_key_after(key_idx)
                    continue

                print(
                    f"[GEMINI] Đang gọi Key #{key_idx + 1} (Async Stream) | {key_status_summary()}",
                    flush=True,
                )

                client = get_genai_client(key_idx)
                if client is None:
                    disable_key(key_idx, "Không tạo được Gemini client")
                    key_idx = next_active_key_after(key_idx)
                    continue

                try:
                    gemini_stream = await client.aio.models.generate_content_stream(
                        model=MODEL_NAME,
                        contents=[
                            genai.types.Part.from_bytes(
                                data=wav_bytes,
                                mime_type="audio/wav",
                            ),
                        ],
                        config=genai.types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            # 300 trước đây tính cả token suy luận.
                            # Gemini 3.6 mặc định thinking=medium nên có thể cắt output dù câu ngắn.
                            max_output_tokens=1024,
                            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                            temperature=0.7,
                            safety_settings=safety_config,
                        ),
                    )

                    print(
                        f"[GEMINI] Thành công khởi tạo request với Key #{key_idx + 1}",
                        flush=True,
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

                    full_response_text = "".join(response_text_parts)
                    finish_reason = last_finish_reason
                    finish_message = last_finish_message
                    usage_summary = last_usage

                    print(
                        f"[GEMINI] Stream hoàn tất | Key #{key_idx + 1} | finish_reason={finish_reason!r} | finish_message={finish_message!r}",
                        flush=True,
                    )

                    if usage_summary is not None:
                        try:
                            print(
                                "[GEMINI] Usage | "
                                f"prompt={getattr(usage_summary, 'prompt_token_count', None)} | "
                                f"output={getattr(usage_summary, 'candidates_token_count', None)} | "
                                f"total={getattr(usage_summary, 'total_token_count', None)}",
                                flush=True,
                            )
                        except Exception:
                            pass

                    # Nếu model bị cắt do giới hạn token, không coi đó là response hoàn chỉnh.
                    # Với cấu hình minimal + 1024 trường hợp này phải rất hiếm, nhưng vẫn kiểm tra.
                    finish_upper = (finish_reason or "").upper()
                    incomplete = any(marker in finish_upper for marker in (
                        "MAX_TOKENS",
                        "INCOMPLETE",
                    ))

                    if incomplete:
                        print(
                            "[GEMINI] Response bị cắt theo finish_reason. "
                            "Không gửi TTS để tránh đọc câu cụt.",
                            flush=True,
                        )
                        await send_event(
                            websocket,
                            "tts_error",
                            "Gemini response incomplete",
                        )
                        # Không retry key khác: đây là vấn đề giới hạn generation,
                        # không phải lỗi API key. Tránh vòng lặp gọi lại vô hạn cùng key.
                        key_idx = None
                        break

                    # Nếu có text hoàn chỉnh, giữ key này làm key hoạt động hiện tại.
                    CURRENT_KEY_INDEX = key_idx
                    gemini_ok = True
                    print(
                        f"[GEMINI] Key #{key_idx + 1} được ghi nhớ là key đang hoạt động.",
                        flush=True,
                    )
                    break

                except Exception as api_err:
                    error_class = classify_gemini_error(api_err)
                    error_text = str(api_err)
                    print(
                        f"[GEMINI] Key #{key_idx + 1} gặp lỗi ({error_class}): {error_text[:240]}",
                        flush=True,
                    )

                    if error_class == "rotate":
                        disable_key(key_idx, error_text)
                        next_idx = next_active_key_after(key_idx)
                        if next_idx is None:
                            print(
                                "[GEMINI] Không còn key hoạt động phía sau key hiện tại.",
                                flush=True,
                            )
                        key_idx = next_idx
                        continue

                    if error_class == "transient":
                        # Không đánh dấu key hỏng. SDK có retry nội bộ cho lỗi tạm thời;
                        # nếu request vẫn thất bại thì báo lỗi thay vì làm key tốt bị vô hiệu.
                        print(
                            f"[GEMINI] Lỗi tạm thời với Key #{key_idx + 1}; không vô hiệu key.",
                            flush=True,
                        )

                    # 400/model/config/content lỗi không giải quyết bằng đổi key.
                    key_idx = None
                    break

            if not gemini_ok:
                print(
                    f"[GEMINI] Không lấy được response hoàn chỉnh. Trạng thái key: {key_status_summary()}",
                    flush=True,
                )
                await send_event(websocket, "tts_error", "Gemini request failed")
                continue

            # ------------------------------------------------------------------
            # E. CHỈ TTS KHI GEMINI ĐÃ TRẢ RESPONSE HOÀN CHỈNH
            # ------------------------------------------------------------------
            try:
                cleaned_text = clean_text_for_tts(full_response_text)

                if not cleaned_text:
                    print("[GEMINI] Response rỗng, không có gì để đọc.", flush=True)
                    await send_event(websocket, "tts_error", "Gemini returned empty response")
                    continue

                print(f"[BÚN ĐẬU RESPONSE]: {cleaned_text}", flush=True)

                # Tách câu nhưng không tạo list rỗng.
                sentences = [
                    clean_text_for_tts(s)
                    for s in re.split(r"(?<=[.?!;\n])\s+", cleaned_text)
                ]
                sentences = [s for s in sentences if s and re.search(r"\w", s)]

                if not sentences:
                    await send_event(websocket, "tts_error", "No readable sentence")
                    continue

                # Báo bắt đầu TTS. ESP vẫn ở THINKING cho tới khi có đủ prebuffer.
                await send_event(websocket, "tts_start")

                all_tts_ok = True

                for sentence_index, sentence in enumerate(sentences, start=1):
                    ok = await text_to_pcm_sentence_with_retry(
                        sentence,
                        websocket,
                        sentence_index,
                        len(sentences),
                    )

                    if not ok:
                        all_tts_ok = False
                        print(
                            f"[TTS] Không tạo được câu {sentence_index}. Dừng lượt này để tránh trả tts_done giả.",
                            flush=True,
                        )
                        break

                if not all_tts_ok:
                    await send_event(
                        websocket,
                        "tts_error",
                        "TTS failed before the answer was completely spoken",
                    )
                    continue

                # CHỈ khi toàn bộ câu đã được stream thành công mới báo done.
                await send_event(websocket, "tts_done")
                print(
                    "[WEBSOCKET] -> Hoàn tất gửi toàn bộ dữ liệu âm thanh tới ESP32.\n",
                    flush=True,
                )

            except WebSocketDisconnect:
                print("[WEBSOCKET] ESP32 ngắt kết nối khi đang xử lý TTS.", flush=True)
                break
            except Exception as stream_err:
                print(
                    f"[STREAM ERROR] Lỗi trong quá trình Gemini/TTS: {stream_err}",
                    flush=True,
                )
                try:
                    await send_event(websocket, "tts_error", "Stream failed")
                except Exception:
                    pass

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 đã ngắt kết nối chủ động.", flush=True)
    except Exception as e:
        print(f"[WEBSOCKET ERROR]: {e}", flush=True)
    finally:
        pcm_buffer.clear()

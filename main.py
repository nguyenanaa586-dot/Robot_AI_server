import asyncio
import io
import os
import re
import wave
from typing import Optional, Tuple

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
    k.strip(' "\'\t\r\n') for k in RAW_KEYS.split(",") if k.strip(' "\'\t\r\n')
]

CURRENT_KEY_INDEX = 0
MODEL_NAME = "gemini-3.6-flash"

# Edge-TTS
TTS_VOICE = "vi-VN-HoaiMyNeural"
TTS_RATE = "+10%"

# ==============================================================================
# 2. PROMPT BÚN ĐẬU
# ==============================================================================
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
- Mỗi câu trả lời phải kết thúc bằng dấu câu phù hợp như . ? hoặc ! nếu câu phù hợp cần dấu câu.
- Không kết thúc giữa một từ, giữa một cụm từ, hoặc bằng một chữ cái/thành phần rõ ràng là chưa hoàn chỉnh.
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.
- Nếu nhận được các câu tự động đăng ký kênh Youtube -> chỉ được hỏi lại nhẹ nhàng không rõ.
- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Không spam.
- Không tự nhận là AI.

## QUAN TRỌNG VỀ KẾT THÚC CÂU
- Trước khi kết thúc lượt trả lời, phải tự kiểm tra rằng ý đã trọn vẹn.
- Không được dừng giữa câu chỉ vì đang stream.
- Nếu còn đang nói một câu, phải tiếp tục cho đến khi hoàn chỉnh rồi mới kết thúc.
"""

# ==============================================================================
# 3. HÀM TIỆN ÍCH
# ==============================================================================
def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
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


def safe_get_finish_info(chunk) -> Tuple[Optional[str], Optional[str]]:
    """Đọc finish_reason/finish_message an toàn từ chunk cuối của Gemini."""
    try:
        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            return None, None
        candidate = candidates[0]
        reason = getattr(candidate, "finish_reason", None)
        message = getattr(candidate, "finish_message", None)

        if reason is not None:
            name = getattr(reason, "name", None)
            value = getattr(reason, "value", None)
            reason_text = name or value or str(reason)
        else:
            reason_text = None

        return reason_text, message
    except Exception:
        return None, None


def looks_like_incomplete_response(text: str) -> bool:
    """
    Heuristic bảo thủ: chỉ đánh dấu là nghi bị cắt khi không có dấu kết thúc
    VÀ phần cuối trông rõ ràng là chưa hoàn chỉnh. Không tự cắt/sửa câu hợp lệ.
    """
    text = text.strip()
    if not text:
        return False

    # Có dấu kết thúc câu hợp lệ -> bình thường.
    if re.search(r"[.!?…]$", text):
        return False

    # Nếu kết thúc bằng dấu phẩy/chấm phẩy/hai chấm -> thường là câu đang dang dở.
    if re.search(r"[,;:]$", text):
        return True

    words = re.findall(r"\S+", text)
    if not words:
        return False

    last = words[-1].strip(" \t\r\n,;:")

    # Một chữ cái đơn ở cuối (trường hợp log '...không r') là dấu hiệu rất mạnh.
    if re.fullmatch(r"[A-Za-zÀ-ỹĐđ]", last):
        return True

    # Một số từ/cụm từ cuối thường là thành phần mở đầu cho ý chưa hoàn chỉnh.
    dangling_endings = {
        "mà", "nên", "nhưng", "vì", "với", "để", "nếu", "khi", "thì",
        "còn", "rồi", "sẽ", "đang", "bị", "được", "là", "của", "cho",
        "từ", "trong", "ngoài", "vào", "ra", "lên", "xuống", "theo",
        "đến", "về", "như", "hay", "hoặc", "và", "một", "những", "cái",
    }
    if last.lower() in dangling_endings:
        return True

    # Nếu đoạn kết thúc bằng rất ít ký tự chữ và trước đó có dấu phẩy, thường là bị cụt.
    if "," in text and len(last) <= 2:
        return True

    return False


def extract_candidate_finish_reason(chunk) -> Optional[str]:
    reason, _ = safe_get_finish_info(chunk)
    return reason


async def generate_gemini_response(
    client,
    wav_bytes: bytes,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Stream Gemini, gom text và giữ lại finish reason/message."""
    gemini_stream = await client.aio.models.generate_content_stream(
        model=MODEL_NAME,
        contents=[
            genai.types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        ],
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=300,
            temperature=0.7,
            safety_settings=[
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
            ],
        ),
    )

    full_response_text = ""
    finish_reason = None
    finish_message = None
    chunk_count = 0

    async for chunk in gemini_stream:
        chunk_count += 1
        txt = safe_get_chunk_text(chunk)
        if txt:
            full_response_text += txt

        reason, message = safe_get_finish_info(chunk)
        if reason is not None:
            finish_reason = reason
        if message:
            finish_message = message

    print(
        f"[GEMINI] Stream kết thúc | chunks={chunk_count} | finish_reason={finish_reason} | "
        f"finish_message={finish_message!r}",
        flush=True,
    )

    return clean_text_for_tts(full_response_text), finish_reason, finish_message


async def recover_incomplete_response(client, draft_text: str) -> str:
    """Yêu cầu Gemini hoàn thiện bản nháp chỉ khi bản nháp có dấu hiệu bị cắt."""
    recovery_prompt = f"""
Bản trả lời nháp sau đây đã bị cắt giữa chừng:

{draft_text}

Hãy viết lại TOÀN BỘ câu trả lời thành một câu trả lời hoàn chỉnh, tự nhiên, ngắn gọn,
đúng tính cách Bún Đậu và đúng ngôi mày-tao. Không giải thích, không nói rằng bản nháp bị lỗi,
không thêm thông tin ngoài ý cần thiết. Chỉ trả về câu trả lời cuối cùng hoàn chỉnh và phải kết
thúc bằng dấu câu phù hợp.
"""

    try:
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=recovery_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=300,
                temperature=0.3,
            ),
        )
        recovered = clean_text_for_tts(response.text or "")
        print(f"[GEMINI RECOVERY] Kết quả: {recovered}", flush=True)
        return recovered
    except Exception as exc:
        print(f"[GEMINI RECOVERY ERROR] {exc}", flush=True)
        return ""


async def text_to_pcm_chunks_edge(
    sentence_text: str,
    websocket: WebSocket,
    target_sample_rate: int = 16000,
) -> bool:
    """TTS -> PCM16/16k/mono -> gửi xuống ESP theo tốc độ thực của audio."""
    clean_txt = clean_text_for_tts(sentence_text)
    if not clean_txt or not re.search(r"\w", clean_txt):
        return False

    try:
        print(f"[TTS] Bắt đầu: '{clean_txt}'", flush=True)
        communicate = edge_tts.Communicate(
            clean_txt,
            voice=TTS_VOICE,
            rate=TTS_RATE,
        )
        mp3_parts = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_parts.append(chunk["data"])

        mp3_bytes = b"".join(mp3_parts)
        if not mp3_bytes:
            print("[TTS ERROR] Edge-TTS không trả audio.", flush=True)
            return False

        def decode_mp3() -> bytes:
            decoded = miniaudio.decode(
                mp3_bytes,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=target_sample_rate,
            )
            return decoded.samples.tobytes()

        pcm_bytes = await asyncio.to_thread(decode_mp3)
        if not pcm_bytes:
            print("[TTS ERROR] Decode MP3 -> PCM rỗng.", flush=True)
            return False

        print(
            f"[TTS] PCM {len(pcm_bytes)} bytes (~{len(pcm_bytes) / 32000:.2f}s audio)",
            flush=True,
        )

        # ESP phát 32000 byte/s ở 16kHz/16bit/mono.
        # Gửi theo 1024 byte và chờ đúng ~32ms để tránh đẩy nhanh hơn tốc độ tiêu thụ.
        chunk_size = 1024
        bytes_per_second = target_sample_rate * 2
        next_send_time = asyncio.get_running_loop().time()

        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            await websocket.send_bytes(chunk)

            next_send_time += len(chunk) / bytes_per_second
            delay = next_send_time - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)

        return True

    except Exception as exc:
        print(f"[TTS ERROR] {exc}", flush=True)
        return False


# ==============================================================================
# 4. ENDPOINT
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu WebSocket Server V2 OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key_index": CURRENT_KEY_INDEX,
        "model": MODEL_NAME,
    }


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
            # A. AUDIO BINARY
            # ------------------------------------------------------------------
            if "bytes" in message and message["bytes"]:
                pcm_buffer.extend(message["bytes"])
                continue

            # ------------------------------------------------------------------
            # B. EVENT TEXT
            # ------------------------------------------------------------------
            if "text" in message and message["text"]:
                msg_text = message["text"].strip()

                if msg_text == '{"event":"start_speech"}':
                    pcm_buffer.clear()
                    print("[WEBSOCKET] -> ESP32 Bắt đầu ghi âm...", flush=True)
                    continue

                if msg_text != '{"event":"end_speech"}':
                    print(f"[WEBSOCKET] Text không xác định: {msg_text[:200]}", flush=True)
                    continue

                print(
                    f"[WEBSOCKET] -> ESP32 Dừng ghi âm. Dung lượng PCM: {len(pcm_buffer)} bytes",
                    flush=True,
                )

                if len(pcm_buffer) < 3200:
                    print("[WEBSOCKET] Âm thanh quá ngắn, bỏ qua.", flush=True)
                    pcm_buffer.clear()
                    await websocket.send_text('{"event":"tts_done"}')
                    continue

                wav_bytes = create_wav_bytes(bytes(pcm_buffer), sample_rate=16000)
                pcm_buffer.clear()

                total_keys = len(API_KEYS)
                if total_keys == 0:
                    print("[GEMINI ERROR] Không tìm thấy GEMINI_API_KEY nào!", flush=True)
                    await websocket.send_text('{"event":"tts_error", "message":"No API Keys"}')
                    continue

                # ==============================================================
                # C. GEMINI STREAM + XOAY KEY KHI LỖI REQUEST/STREAM
                # ==============================================================
                final_response = ""
                final_finish_reason = None
                final_finish_message = None
                successful_client = None

                for step in range(total_keys):
                    key_idx = (CURRENT_KEY_INDEX + step) % total_keys
                    client = get_genai_client(key_idx)

                    try:
                        print(
                            f"[GEMINI] Đang gọi Key #{key_idx + 1} (Async Stream)...",
                            flush=True,
                        )
                        draft_text, finish_reason, finish_message = await generate_gemini_response(
                            client, wav_bytes
                        )
                        successful_client = client
                        CURRENT_KEY_INDEX = key_idx
                        final_response = draft_text
                        final_finish_reason = finish_reason
                        final_finish_message = finish_message
                        print(f"[GEMINI] Thành công với Key #{key_idx + 1}", flush=True)
                        break
                    except Exception as api_err:
                        print(
                            f"[GEMINI] Key #{key_idx + 1} lỗi trong request/stream: "
                            f"{str(api_err)[:200]}. Chuyển key tiếp theo.",
                            flush=True,
                        )
                        successful_client = None
                        continue

                if successful_client is None:
                    print("[GEMINI] Tất cả API Keys đều thất bại!", flush=True)
                    await websocket.send_text('{"event":"error", "message":"503 Service Unavailable"}')
                    continue

                # ==============================================================
                # D. CHẨN ĐOÁN + KHÔI PHỤC CÂU BỊ CẮT
                # ==============================================================
                print(
                    f"[GEMINI] finish_reason={final_finish_reason} | "
                    f"finish_message={final_finish_message!r}",
                    flush=True,
                )

                if not final_response:
                    print("[GEMINI ERROR] Không có text trả về.", flush=True)
                    await websocket.send_text('{"event":"tts_error", "message":"Empty AI response"}')
                    continue

                print(f"[BÚN ĐẬU RESPONSE RAW]: {final_response}", flush=True)

                # MAX_TOKENS / BLOCKLIST / OTHER: không tin bản nháp là hoàn chỉnh.
                unsafe_finish_reasons = {
                    "MAX_TOKENS",
                    "SAFETY",
                    "BLOCKLIST",
                    "PROHIBITED_CONTENT",
                    "SPII",
                    "RECITATION",
                    "OTHER",
                }

                needs_recovery = (
                    final_finish_reason in unsafe_finish_reasons
                    or looks_like_incomplete_response(final_response)
                )

                if needs_recovery:
                    print(
                        "[GEMINI] Phát hiện câu trả lời có khả năng chưa hoàn chỉnh. "
                        "Đang yêu cầu Gemini hoàn thiện...",
                        flush=True,
                    )
                    recovered = await recover_incomplete_response(
                        successful_client, final_response
                    )
                    if recovered:
                        final_response = recovered
                    else:
                        print(
                            "[GEMINI] Recovery thất bại. Không gửi tts_done giả.",
                            flush=True,
                        )
                        await websocket.send_text(
                            '{"event":"tts_error", "message":"Incomplete AI response"}'
                        )
                        continue

                cleaned_text = clean_text_for_tts(final_response)
                if not cleaned_text:
                    print("[GEMINI ERROR] Text sau khi làm sạch rỗng.", flush=True)
                    await websocket.send_text(
                        '{"event":"tts_error", "message":"Empty cleaned response"}'
                    )
                    continue

                print(f"[BÚN ĐẬU RESPONSE]: {cleaned_text}", flush=True)

                sentences = [
                    s for s in re.split(r"(?<=[.?!…])\s+", cleaned_text) if s.strip()
                ]

                all_tts_ok = True
                for idx, sent in enumerate(sentences, start=1):
                    clean_sent = clean_text_for_tts(sent)
                    if not clean_sent:
                        continue

                    print(
                        f"[TTS] Câu {idx}/{len(sentences)}: '{clean_sent}'",
                        flush=True,
                    )
                    ok = await text_to_pcm_chunks_edge(
                        clean_sent,
                        websocket,
                        target_sample_rate=16000,
                    )
                    if not ok:
                        all_tts_ok = False
                        break

                if all_tts_ok:
                    await websocket.send_text('{"event":"tts_done"}')
                    print(
                        "[WEBSOCKET] -> Hoàn tất gửi toàn bộ dữ liệu âm thanh tới ESP32.\n",
                        flush=True,
                    )
                else:
                    await websocket.send_text(
                        '{"event":"tts_error", "message":"TTS failed"}'
                    )
                    print("[WEBSOCKET] -> TTS thất bại, không gửi tts_done.", flush=True)

    except WebSocketDisconnect:
        print("[WEBSOCKET] ESP32 đã ngắt kết nối chủ động.", flush=True)
    except Exception as e:
        print(f"[WEBSOCKET ERROR]: {e}", flush=True)
    finally:
        pcm_buffer.clear()

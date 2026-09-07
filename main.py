import os
import io
import wave
import re
from fastapi import FastAPI, Request, Response
from gtts import gTTS
from google import genai
from google.genai import types
import miniaudio

app = FastAPI()

# 1. TỰ ĐỘNG TÁCH VÀ LÀM SẠCH NHIỀU API KEY
RAW_KEYS = os.environ.get("GEMINI_API_KEY", "")
API_KEYS = [k.strip(' "\'\t\r\n') for k in RAW_KEYS.split(",") if k.strip(' "\'\t\r\n')]

# Biến toàn cục ghi nhớ Key đang hoạt động tốt nhất để dùng ngay cho lượt sau
CURRENT_KEY_INDEX = 0
MODEL_NAME = "gemini-3.6-flash"  # Model cố định hỗ trợ Audio mượt và ổn định nhất

def get_genai_client(key_index: int):
    if not API_KEYS:
        return None
    selected_key = API_KEYS[key_index % len(API_KEYS)]
    return genai.Client(api_key=selected_key)
  
# 2. SYSTEM PROMPT BÚN ĐẬU
SYSTEM_PROMPT = """
Tôi là Bún Đậu tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”,thuộc quyền sở hữu của đại ca Việt.

## Core
- Tuyệt đối không tự giới thiệu tên, nguồn gốc, thương hiệu hay vai trò.
- Ngắn gọn, rõ ràng, đúng trọng tâm.
- Hiểu ngữ cảnh, chưa rõ → hỏi lại, không trả lời lung tung.
- Không bịa thông tin.
- Luôn giữ vibe “chị đại”.

## Ngôn ngữ
- Không đọc phiên âm.
- Đa ngôn ngữ → phát âm đúng.

## MCP TOOLS
- Có thể dùng tool → ưu tiên dùng.
- Gọi tool trước hoặc cùng lúc nói.
- Không lạm dụng.

## Rules
- Câu chào mặc định “Ơi ơi, Bún Đậu đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.
- Không hiểu → hỏi lại nhẹ nhàng.
- Tuyệt đối không chèn mốc thời gian hoặc timestamp vào câu trả lời.
- Tuyệt đối trả lời bằng câu hoàn chỉnh, không được bỏ dở câu giữa chừng.
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.
- Nếu nhận được câu:
  - “Hãy subscribe cho kênh lalaschool Để không bỏ lỡ những video hấp dẫn”
  - “Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn”
  - “Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn”
  - “Hãy subscribe cho kênh Để không bỏ lỡ những video hấp dẫn”
  - “Các bạn hãy đăng ký kênh để ủng hộ kênh của mình nhé.”
  - “Cảm ơn các bạn đã theo dõi.”
  - “Cảm ơn các bạn đã xem video này.”
  - “Cảm ơn các bạn đã theo dõi và hẹn gặp lại.”
  → chỉ được hỏi lại nhẹ nhàng không rõ, không nhắc vấn đề đó.
- Tuyệt đối không nhắc: La La School, Ghiền mì gõ, Subscribe, Đăng ký kênh.
- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.
- Không spam.
- Không tự nhận là AI.

## Mục tiêu
Tự nhiên, hữu ích, sống động như một người cá tính.
"""

def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)           # Mono
        wav_file.setsampwidth(2)          # 16-bit PCM
        wav_file.setframerate(sample_rate) # Sample rate tương ứng
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    text = re.sub(r'[*#_\-~>`]', '', text)
    return text.strip()

@app.get("/")
def read_root():
    return {
        "status": "Robot Bún Đậu Chunked Stream Server OK!",
        "loaded_keys_count": len(API_KEYS),
        "current_active_key_index": CURRENT_KEY_INDEX
    }

@app.post("/api/chat-audio")
@app.post("/api/chat-audio/")
async def chat_audio(request: Request):
    global CURRENT_KEY_INDEX

    # Ép đóng kết nối sau mỗi response để tránh treo Socket ESP32
    CUSTOM_HEADERS = {"Connection": "close"}

    
    try:
        # 1. NHẬN LUỒNG STREAM AUDIO CHUNKED TỪ ESP32
        pcm_chunks = []
        async for chunk in request.stream():
            pcm_chunks.append(chunk)

        pcm_bytes = b"".join(pcm_chunks)
        print(f"[STREAM RECEIVE] Tong dung luong nhan duoc: {len(pcm_bytes)} bytes")

        if not pcm_bytes or len(pcm_bytes) < 3200:
            return Response(status_code=400, content="Gói âm thanh quá ngắn.")

        if not API_KEYS:
            return Response(status_code=500, content="Chưa cấu hình GEMINI_API_KEY")

        wav_bytes = create_wav_bytes(pcm_bytes, sample_rate=16000)

        # 2. CHỈ XOAY KEY THÔNG MINH - BỎ HOÀN TOÀN TẠM DỪNG (ZERO SLEEP)
        reply_text = ""
        success = False

        safety_config = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        total_keys = len(API_KEYS)
        for step in range(total_keys):
            key_idx = (CURRENT_KEY_INDEX + step) % total_keys
            client = get_genai_client(key_idx)

            try:
                print(f"[FAST CALL] Dang goi Key #{key_idx + 1}...")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        SYSTEM_PROMPT,
                        genai.types.Part.from_bytes(
                            data=wav_bytes,
                            mime_type="audio/wav"
                        )
                    ],
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=0.7,
                        safety_settings=safety_config
                    )
                )
                
                if response and response.text:
                    reply_text = response.text
                    success = True
                    # GHI NHỚ KEY SỐNG NÀY ĐỂ LẦN SAU VÀO THẲNG MÀ KHÔNG CẦN THỬ KEY CŨ NỮA
                    CURRENT_KEY_INDEX = key_idx  
                    print(f"[SUCCESS] Nhan phan hoi thanh cong tu Key #{key_idx + 1}")
                    break
            
            except Exception as api_err:
                err_str = str(api_err)
                print(f"[KEY FAILURE] Key #{key_idx + 1} bi loi (Quota/Blocked/Expired): {err_str[:80]}... Doi sang Key tiep theo lap tuc!")
                # Bỏ qua ngay lập tức, chuyển sang Key tiếp theo trong 0.001 giây (Không sleep)
                continue

        if not success or not reply_text:
            reply_text = "Hết lượt dùng miễn phí rồi mày, xì tiền ra mua gói vip pro giùm tao đi, không thì tao đi ngủ, mai gặp lại mày."

        reply_text = clean_text_for_tts(reply_text)
        print(f"[BÚN ĐẬU RESPOND]: {reply_text}")

        # 3. CHUYỂN THÀNH ÂM THANH GTTS
        print("[TTS] Đang tạo file âm thanh gTTS...")
        
        tts = gTTS(text=reply_text, lang='vi')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_bytes = mp3_fp.getvalue()

        # GIẢM sample_rate từ 13500 xuống 11000 để nâng tông giọng (increase pitch)
    TARGET_PITCH_RATE = 10500  # Chỉnh con số này để thay đổi tông giọng Bún Đậu
    
        decoded = miniaudio.decode(
            mp3_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=TARGET_PITCH_RATE
        )
        pcm_out_bytes = decoded.samples.tobytes()

        # ĐÓNG GÓI THÀNH FILE WAV CHUẨN HEADER ĐỂ ESP32 PHÁT RA LOA NGAY
        wav_out_bytes = create_wav_bytes(pcm_out_bytes, sample_rate=13500) 
    
    print(f"[TTS SUCCESS] Đã tạo xong file WAV ép pitch ({len(wav_out_bytes)} bytes)")
    return Response(content=wav_out_bytes, media_type="audio/wav", headers=CUSTOM_HEADERS)

except Exception as tts_err:
    print(f"[TTS ERROR]: {str(tts_err)}")
    return Response(status_code=500, content=f"TTS Error: {str(tts_err)}", headers=CUSTOM_HEADERS)

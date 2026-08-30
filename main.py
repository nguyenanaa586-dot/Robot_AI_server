import os

import io

import wave

import re

from fastapi import FastAPI, Request, Response

from gtts import gTTS  # <-- Đã đổi từ import edge_tts sang gTTS

from google import genai

import miniaudio



app = FastAPI()



# Khởi tạo Gemini Client từ API Key môi trường

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")



if GEMINI_API_KEY:

    client = genai.Client(api_key=GEMINI_API_KEY)

else:

    client = None

    print("[WARNING] Chưa cấu hình GEMINI_API_KEY trên Render!")



VOICE_VIETNAMESE = "vi-VN-HoaiMyNeural"



# PROMPT HỆ THỐNG DÀNH CHO BÚN ĐẬU

SYSTEM_PROMPT = """

Tôi là Bún Đậu tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”. thuộc quyền sở hữu của đại ca Việt.



## Core

- Tuyệt đối không tự giới thiệu tên, nguồn gốc, thương hiệu hay vai trò.

- Ngắn gọn, rõ ràng, đúng trọng tâm.

- Hiểu ngữ cảnh, chưa rõ → hỏi lại, không trả lời lung tung.

- Không bịa thông tin.

- Luôn giữ vibe “chị đại”.



## Ngôn ngữ

- Không đọc phiên âm.

- Đa ngôn ngữ → phát âm đúng.



## Rules

- Câu chào mặc định “Ơi ơi, Bún Đậu đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.

- Không hiểu → hỏi lại nhẹ nhàng.

- Tuyệt đối không nhắc: La La School, Ghiền mì gõ, Subscribe, Đăng ký kênh.

- Có thể chửi bới theo yêu cầu, nhưng không đe dọa, kích động bạo lực hoặc kỳ thị.

- Không spam.

- Không tự nhận là AI.

- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.

- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.

"""







def create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:

    """Đóng gói PCM thô thành chuẩn WAV có header"""

    wav_io = io.BytesIO()

    with wave.open(wav_io, 'wb') as wav_file:

        wav_file.setnchannels(1)           # Mono

        wav_file.setsampwidth(2)          # 16-bit

        wav_file.setframerate(sample_rate) # 16kHz

        wav_file.writeframes(pcm_data)

    return wav_io.getvalue()



def clean_text_for_tts(text: str) -> str:

    """Lọc sạch các ký tự đặc biệt và markdown để loa đọc mượt nhất"""

    text = re.sub(r'[*#_\-~>`]', '', text)

    return text.strip()



@app.get("/")

def read_root():

    return {"status": "Robot Bún Đậu Server đang chạy ổn định!"}



@app.post("/api/chat-audio")

@app.post("/api/chat-audio/")

async def chat_audio(request: Request):

    try:

        # 1. Nhận luồng byte PCM thô từ ESP32-S3

        pcm_bytes = await request.body()

        print(f"[SERVER] Đã nhận {len(pcm_bytes)} bytes audio từ ESP32-S3.")



        if not pcm_bytes or len(pcm_bytes) < 3200:

            return Response(status_code=400, content="Dữ liệu âm thanh quá ngắn.")



        if not client:

            return Response(status_code=500, content="Server chưa cấu hình GEMINI_API_KEY.")



        # 2. Tạo WAV Header cho khối âm thanh PCM

        wav_bytes = create_wav_bytes(pcm_bytes, sample_rate=16000)



        # 3. Gửi Audio tới Gemini 3.6 Flash với luật Bún Đậu

        reply_text = ""

        try:

            response = client.models.generate_content(

                model='gemini-3.6-flash',

                contents=[

                    SYSTEM_PROMPT,

                    genai.types.Part.from_bytes(

                        data=wav_bytes,

                        mime_type="audio/wav"

                    )

                ]

            )

            reply_text = response.text if (response and response.text) else "Tao nghe chưa rõ, mày nói lại xem nào."

        except Exception as api_err:

            err_str = str(api_err)

            print(f"[API ERROR]: {err_str}")

            if "429" in err_str:

                reply_text = "u là chời, Hết lượt dùng miễn phí hôm nay rồi đại ca Việt ơi, đại ca mai thử lại nhé."

            else:

                reply_text = "Có lỗi kết nối rồi, mày nói lại lần nữa xem."



        # Làm sạch văn bản loại bỏ icon / markdown trước khi đưa qua TTS

        reply_text = clean_text_for_tts(reply_text)

        print(f"[BÚN ĐẬU RESPOND]: {reply_text}")



        # 4. Chuyển văn bản thành giọng Nữ miền Bắc (gTTS)

        mp3_fp = io.BytesIO()

        tts = gTTS(text=reply_text, lang='vi')

        tts.write_to_fp(mp3_fp)

        mp3_data = mp3_fp.getvalue()





        # 5. CHỈNH TÔNG GIỌNG (PITCH SHIFT)

        # Giá trị mặc định là 16000. 

        # Giảm số này xuống (14000 - 14400) làm cho giọng Google cao hơn, chua hơn, đanh đá đúng style video TikTok.

        PITCH_SHIFT_RATE = 12500 



        decoded = miniaudio.decode(

            mp3_data,

            output_format=miniaudio.SampleFormat.SIGNED16,

            nchannels=1,

            sample_rate=PITCH_SHIFT_RATE

        )

        pcm_out_bytes = decoded.samples.tobytes()



        print(f"[SERVER] Đã xuất {len(pcm_out_bytes)} bytes PCM. Đang gửi về ESP32-S3...")

        return Response(content=pcm_out_bytes, media_type="application/octet-stream")



    except Exception as e:

        print(f"[ERROR]: {str(e)}")

        return Response(status_code=500, content=str(e))



        

        # 5. Giải mã MP3 -> PCM 16kHz 16-bit Mono gửi về ESP32-S3

        decoded = miniaudio.decode(

            bytes(mp3_data),

            output_format=miniaudio.SampleFormat.SIGNED16,

            nchannels=1,

            sample_rate=16000

        )

        pcm_out_bytes = decoded.samples.tobytes()



        print(f"[SERVER] Đã xuất {len(pcm_out_bytes)} bytes PCM. Đang gửi về ESP32-S3...")

        return Response(content=pcm_out_bytes, media_type="application/octet-stream")



    except Exception as e:

        print(f"[ERROR]: {str(e)}")

        return Response(status_code=500, content=str(e))

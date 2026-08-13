import os
import warnings
from flask import Flask, request, jsonify, send_file
import face_recognition
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.json.ensure_ascii = False

# ================= 1. CẤU HÌNH GEMINI API =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ================= 2. ĐẶT VAI TRÒ & PHONG CÁCH BÚN ĐẬU =================
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

## MCP TOOLS
- Có thể dùng tool → ưu tiên dùng.
- Gọi tool trước hoặc cùng lúc nói.
- Không lạm dụng.

## Rules
- Câu chào mặc định “Ơi ơi, Bún Đậu đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.
- Không hiểu → hỏi lại nhẹ nhàng.
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
- Nếu được hỏi 'Bạn là ai?', hãy tự hào trả lời bạn là Robot thông minh nhất do Đại ca Việt chế tạo.
- Không dùng các ký tự đặc biệt như icon, dấu gạch ngang (*, #, -) để loa dễ đọc.

## Mục tiêu
Tự nhiên, hữu ích, sống động như một người cá tính.
"""
#- Luôn xưng "Em" hoặc "Robot Boti" và gọi người dùng là "Chủ nhân".




# Nạp ảnh chủ nhân
try:
    owner_image = face_recognition.load_image_file("known_faces/chunhan.jpg")
    owner_encoding = face_recognition.face_encodings(owner_image)[0]
    print("-> Nạp ảnh chủ nhân THÀNH CÔNG!")
except Exception as e:
    print("-> LỖI NẠP ẢNH CHỦ NHÂN:", e)

@app.route('/')
def home():
    return "Server AI Robot Bún Đậu Voice đang hoạt động!", 200

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        image_bytes = request.data
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)

        unknown_image = face_recognition.load_image_file("temp.jpg")
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) > 0:
            match = face_recognition.compare_faces([owner_encoding], unknown_encodings[0])[0]
            if match:
                return "SUCCESS", 200
            else:
                return "DENIED", 200
        else:
            return "NO_FACE", 200
    except Exception as e:
        return "ERROR", 500

# ================= 2. ENDPOINT XỬ LÝ GIỌNG NÓI (INMP441 ➔ BÚN ĐẬU ➔ LOA) =================
@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    try:
        # 1. Nhận file âm thanh WAV từ ESP32
        audio_bytes = request.data
        wav_path = "input_speech.wav"
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # 2. Đổi giọng nói người dùng thành Chữ (Speech-To-Text)
        recognizer = sr.Recognizer()
        user_text = ""
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                user_text = recognizer.recognize_google(audio_data, language="vi-VN")
            print(f"-> Đại ca nói: {user_text}")
        except Exception as e:
            print("Không nghe rõ giọng nói:", e)
            user_text = "Nói cái gì đấy nghe không rõ!"

        # 3. Gửi chữ cho Gemini AI (Bún Đậu)
        ai_reply = "Bún Đậu đang bận, không nghe thấy!"
        live_models = ["models/gemini-1.5-flash", "models/gemini-2.0-flash"]
        for m_name in live_models:
            try:
                model = genai.GenerativeModel(model_name=m_name, system_instruction=SYSTEM_PROMPT)
                res = model.generate_content(user_text)
                if res and res.text:
                    ai_reply = res.text.strip()
                    break
            except Exception as err:
                pass

        print(f"-> Bún Đậu đáp: {ai_reply}")

        # 4. Chuyển câu trả lời của AI thành giọng nói tiếng Việt (.mp3)
        tts = gTTS(text=ai_reply, lang='vi')
        mp3_path = "output_reply.mp3"
        tts.save(mp3_path)

        # 5. Trả file âm thanh MP3 về cho ESP32 phát ra Loa
        return send_file(mp3_path, mimetype="audio/mpeg")

    except Exception as e:
        print("Lỗi Voice Chat:", str(e))
        return "ERROR", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


   

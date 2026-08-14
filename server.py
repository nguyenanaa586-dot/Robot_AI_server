import os
import io
import re
import warnings
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
# --- SỬ DỤNG THƯ VIỆN GEMINI SDK MỚI CỦA GOOGLE ---
from google import genai
from google.genai import types

import speech_recognition as sr
from gtts import gTTS

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.json.ensure_ascii = False

# ================= 1. CẤU HÌNH GEMINI CLIENT (MỚI) =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ai_client = None

if GEMINI_API_KEY:
    # Khởi tạo Client theo chuẩn SDK google-genai mới
    ai_client = genai.Client(api_key=GEMINI_API_KEY)

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
#- Luôn xưng "Em" hoặc "Robot Boti" và gọi người dùng là "Chủ nhân".



# ================= 3. QUẢN LÝ NHẬN DIỆN KHUÔN MẶT BẰNG OPENCV =================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
owner_hist = None
OWNER_IMG_PATH = "known_faces/chunhan.jpg"

def init_owner_face():
    global owner_hist
    try:
        if os.path.exists(OWNER_IMG_PATH):
            img = cv2.imread(OWNER_IMG_PATH)
            if img is not None:
                img = cv2.resize(img, (320, 240))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    owner_hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
                    cv2.normalize(owner_hist, owner_hist, 0, 1, cv2.NORM_MINMAX)
                    print("-> Nạp ảnh chủ nhân THÀNH CÔNG bằng OpenCV!")
                else:
                    print("-> CẢNH BÁO: Không tìm thấy khuôn mặt trong chunhan.jpg")
        else:
            print(f"-> CẢNH BÁO: Không tìm thấy file {OWNER_IMG_PATH}")
    except Exception as e:
        print("-> LỖI NẠP ẢNH CHỦ NHÂN:", str(e))

init_owner_face()

def clean_text_for_tts(text):
    """Làm sạch văn bản, loại bỏ các ký tự Markdown để gTTS đọc chuẩn nhất"""
    text = re.sub(r'[\*\#\-\_\~\`\>\[\]\(\)]', '', text)
    return text.strip()

@app.route('/')
def home():
    return "Server AI Robot Bún Đậu Voice đang hoạt động mượt mà!", 200

# ================= 4. ENDPOINT XÁC THỰC MẶT (/VERIFY_FACE) =================
@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        image_bytes = request.data
        if not image_bytes or len(image_bytes) < 2000:
            return "NO_FACE", 200

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return "NO_FACE", 200

        img_small = cv2.resize(img, (320, 240))
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            return "NO_FACE", 200

        if owner_hist is None:
            return "SUCCESS", 200

        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        curr_hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

        similarity = cv2.compareHist(owner_hist, curr_hist, cv2.HISTCMP_CORREL)
        print(f"-> Độ tương đồng khuôn mặt: {similarity:.2f}")

        if similarity > 0.30:
            return "SUCCESS", 200
        else:
            return "DENIED", 200

    except Exception as e:
        print(f"[VERIFY_FACE ERROR]: {e}")
        return "NO_FACE", 200

# ================= 5. ENDPOINT XỬ LÝ GIỌNG NÓI (/VOICE_CHAT) =================
@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    try:
        audio_bytes = request.data
        if not audio_bytes or len(audio_bytes) < 1000:
            user_text = "Nói cái gì đấy nghe không rõ!"
        else:
            wav_path = "input_speech.wav"
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)

            # 1. Speech-To-Text
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

            if os.path.exists(wav_path):
                os.remove(wav_path)

        # 2. Gửi câu thoại cho Gemini AI qua SDK mới google-genai
        ai_reply = "Nói lại xem nào, Bún Đậu nghe chưa rõ!"
        if ai_client and user_text:
            try:
                # Gọi Model Gemini 2.5 Flash mới nhất
                response = ai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7,
                    ),
                )
                if response and response.text:
                    ai_reply = clean_text_for_tts(response.text)
            except Exception as err:
                print("Lỗi Gemini API (google-genai):", err)

        print(f"-> Bún Đậu đáp: {ai_reply}")

        # 3. Chuyển thành file MP3
        mp3_path = "output_reply.mp3"
        tts = gTTS(text=ai_reply, lang='vi')
        tts.save(mp3_path)

        return send_file(mp3_path, mimetype="audio/mpeg")

    except Exception as e:
        print("[VOICE_CHAT ERROR]:", str(e))
        fallback_mp3 = "error_reply.mp3"
        tts = gTTS(text="Bún Đậu bị lỗi kết nối rồi đại ca ơi!", lang='vi')
        tts.save(fallback_mp3)
        return send_file(fallback_mp3, mimetype="audio/mpeg")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

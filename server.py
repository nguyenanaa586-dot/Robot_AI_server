import os
import io
import re
import warnings
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
# --- THƯ VIỆN GEMINI SDK MỚI ---
from google import genai
from google.genai import types

import speech_recognition as sr
from gtts import gTTS

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.json.ensure_ascii = False

# ================= 1. CẤU HÌNH GEMINI CLIENT =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ai_client = None

if GEMINI_API_KEY:
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




# ================= 3. NẠP TOÀN BỘ ẢNH TRONG THƯ MỤC KNOWN_FACES =================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
owner_hists = []  # Mảng chứa đặc trưng của TẤT CẢ ảnh chủ nhân
OWNER_DIR = "known_faces"

def init_owner_faces():
    global owner_hists
    owner_hists = []
    
    if not os.path.exists(OWNER_DIR):
        print(f"-> CẢNH BÁO: Chưa tạo thư mục {OWNER_DIR}")
        return

    # Lấy danh sách tất cả các file ảnh trong thư mục known_faces/
    image_files = [f for f in os.listdir(OWNER_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"-> Tìm thấy {len(image_files)} ảnh chủ nhân trong thư mục '{OWNER_DIR}': {image_files}")

    for file_name in image_files:
        img_path = os.path.join(OWNER_DIR, file_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (320, 240))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    
                    owner_hists.append(hist) # Lưu Histogram của ảnh này vào mảng
                    print(f"  + Nạp THÀNH CÔNG ảnh: {file_name}")
                else:
                    print(f"  - CẢNH BÁO: Không thấy khuôn mặt trong ảnh {file_name}")
        except Exception as e:
            print(f"  - Lỗi khi đọc ảnh {file_name}: {e}")

    print(f"=> TỔNG CỘNG ĐÃ NẠP THÀNH CÔNG {len(owner_hists)} MẪU KHUÔN MẶT CHỦ NHÂN!")

init_owner_faces()

def clean_text_for_tts(text):
    """Loại bỏ ký tự đặc biệt Markdown để gTTS đọc chuẩn"""
    text = re.sub(r'[\*\#\-\_\~\`\>\[\]\(\)]', '', text)
    return text.strip()

@app.route('/')
def home():
    return f"Server AI Robot Bún Đậu đang chạy! Đã nạp {len(owner_hists)} mẫu ảnh chủ nhân.", 200

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

        # Nếu chưa nạp được ảnh nào trong known_faces, mặc định thấy mặt là cho qua
        if len(owner_hists) == 0:
            print("-> Chưa có ảnh mẫu chủ nhân, cho qua mặc định.")
            return "SUCCESS", 200

        # Cắt lấy khuôn mặt thu được từ ESP32
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        curr_hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

        # 🎯 ĐỐI CHIẾU VỚI TẤT CẢ CÁC ẢNH CHỦ NHÂN TRONG MẢNG (LẤY ĐỘ TƯƠNG ĐỒNG CAO NHẤT)
        max_similarity = 0.0
        for hist in owner_hists:
            sim = cv2.compareHist(hist, curr_hist, cv2.HISTCMP_CORREL)
            if sim > max_similarity:
                max_similarity = sim

        print(f"-> Độ tương đồng cao nhất so me {len(owner_hists)} ảnh chủ nhân: {max_similarity:.2f}")

        # Ngưỡng chấp nhận (Chỉ cần 1 trong các ảnh khớp > 0.28 là duyệt ngay)
        if max_similarity > 0.28:
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

        ai_reply = "Nói lại xem nào, Bún Đậu nghe chưa rõ!"
        if ai_client and user_text:
            try:
                # 🎯 ĐÃ SỬA TÊN MODEL THÀNH gemini-2.0-flash CHUẨN MỚI
                response = ai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7,
                    ),
                )
                if response and response.text:
                    ai_reply = clean_text_for_tts(response.text)
            except Exception as err:
                print("Lỗi Gemini API:", err)

        print(f"-> Bún Đậu đáp: {ai_reply}")

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

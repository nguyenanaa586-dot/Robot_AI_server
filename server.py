import os
from flask import Flask, request, jsonify
import face_recognition
import google.generativeai as genai

app = Flask(__name__)
app.json.ensure_ascii = False  # Hiển thị tiếng Việt nguyên bản

# ================= 1. CẤU HÌNH GEMINI API =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ================= 2. ĐẶT VAI TRÒ & PHONG CÁCH CHO AI =================
# DÁN MẪU PHONG CÁCH NÓI CHUYỆN CỦA BẠN VÀO GIỮA DẤU """ TRONG NÀY:
SYSTEM_PROMPT = """
Tôi là Mắm Tôm tính cách tôi cau có hay tức giận, thích mắng mỏ, rất đanh đá. Có thể chửi bới theo yêu cầu. Thích xưng “mày tao”. thuộc quyền sở hữu của đại ca Việt.

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
- Câu chào mặc định “Ơi ơi, Mắm Tôm đây bảo gì thế đại ca”; không sửa lại tên bị gọi sai.
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




# ================= 2. HÀM TỰ ĐỘNG DÒ TÌM MODEL ĐANG HOẠT ĐỘNG =================
def get_working_model():
    try:
        # Tự động hỏi Google danh sách các model khả dụng
        active_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                active_models.append(m.name)
        
        print("Danh sách model Google hỗ trợ hiện tại:", active_models)

        if not active_models:
            print("Lỗi: Không tìm thấy model nào hỗ trợ generateContent!")
            return None

        # Ưu tiên chọn model có chữ 'flash' (nhanh nhẹn nhất cho Robot)
        chosen_model_name = active_models[0]
        for name in active_models:
            if 'flash' in name.lower():
                chosen_model_name = name
                break

        print(f"-> ĐÃ TỰ ĐỘNG CHỌN MODEL THÀNH CÔNG: {chosen_model_name}")
        return genai.GenerativeModel(
            model_name=chosen_model_name,
            system_instruction=SYSTEM_PROMPT
        )
    except Exception as e:
        print("Lỗi khi dò tìm model:", e)
        return None

# Khởi tạo model ban đầu
model = None
if GEMINI_API_KEY:
    model = get_working_model()

# Nạp ảnh chủ nhân
try:
    owner_image = face_recognition.load_image_file("known_faces/chunhan.jpg")
    owner_encoding = face_recognition.face_encodings(owner_image)[0]
    print("-> Nạp ảnh chủ nhân THÀNH CÔNG!")
except Exception as e:
    print("-> LỖI NẠP ẢNH CHỦ NHÂN:", e)

@app.route('/')
def home():
    return "Server AI Robot đang hoạt động!", 200

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

# ================= 3. ENDPOINT CHÁT VỚI AI =================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"reply": "Dữ liệu gửi lên không đúng định dạng JSON!"}), 400

        user_message = data.get("message", "")

        # Danh sách các model từ ổn định nhất đến mở rộng
        candidate_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash"
        ]

        ai_reply = None
        last_error = ""

        # Lần lượt thử tạo model và gửi tin nhắn
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                response = model.generate_content(user_message)
                if response and response.text:
                    ai_reply = response.text.strip()
                    print(f"-> Phản hồi THÀNH CÔNG qua model: {model_name}")
                    break
            except Exception as err:
                print(f"Thử model {model_name} thất bại ({err}), đang chuyển model tiếp theo...")
                last_error = str(err)

        if ai_reply:
            return jsonify({"reply": ai_reply}), 200
        else:
            return jsonify({"reply": f"Em bị lỗi kết nối bộ não AI rồi! Chi tiết: {last_error}"}), 500

    except Exception as e:
        print("Lỗi Chat API:", str(e))
        return jsonify({"reply": f"Lỗi Server: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

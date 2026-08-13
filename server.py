from flask import Flask, request, jsonify
import face_recognition
import os

app = Flask(__name__)

# 1. NẠP ẢNH CHỦ NHÂN VÀO BỘ NHỚ AI
print("-> Đang nạp ảnh khuôn mặt chủ nhân...")
try:
    owner_image = face_recognition.load_image_file("known_faces/chunhan.jpg")
    owner_encoding = face_recognition.face_encodings(owner_image)[0]
    print("-> Nạp ảnh chủ nhân THÀNH CÔNG!")
except Exception as e:
    print("-> LỖI: Chưa có ảnh chunhan.jpg trong thư mục known_faces!")

# 2. ĐƯỜNG DẪN LẮNG NGHE YÊU CẦU QUÉT MẶT TỪ ESP32
@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        # Nhận dữ liệu ảnh JPEG từ ESP32-S3 gửi qua Wi-Fi
        image_bytes = request.data
        
        # Lưu tạm ảnh robot chụp ra file temp.jpg
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)

        # Mở ảnh vừa chụp và tìm khuôn mặt
        unknown_image = face_recognition.load_image_file("temp.jpg")
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) > 0:
            # So sánh khuôn mặt trong ảnh với khuôn mặt chủ nhân
            match = face_recognition.compare_faces([owner_encoding], unknown_encodings[0])[0]
            
            if match:
                print("==> PHÁT HIỆN: ĐÚNG LÀ CHỦ NHÂN!")
                return "SUCCESS", 200
            else:
                print("==> CẢNH BÁO: NGƯỜI LẠ!")
                return "DENIED", 200
        else:
            print("==> Không tìm thấy khuôn mặt nào trong ảnh!")
            return "NO_FACE", 200

    except Exception as e:
        print("Lỗi xử lý Server:", e)
        return "ERROR", 500

# 3. CHẠY SERVER AI TRÊN CỔNG 5000
if __name__ == '__main__':
    print("Server AI đang chạy và chờ kết nối từ Robot...")
    app.run(host='0.0.0.0', port=5000)
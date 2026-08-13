# 1. Sử dụng hệ điều hành Python 3.10 Linux siêu nhẹ
FROM python:3.10-slim

# 2. Cài đặt cmake và các thư viện biên dịch C++ cần thiết cho face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt-get/lists/*

# 3. Tạo thư mục làm việc
WORKDIR /app

# 4. Sao chép và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Sao chép toàn bộ code server vào
COPY . .

# 6. Chạy Server Gunicorn trên cổng do Render cấp
CMD gunicorn server:app --bind 0.0.0.0:$PORT

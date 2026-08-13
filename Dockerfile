# 1. Sử dụng hệ điều hành Python 3.10 Linux siêu nhẹ
FROM python:3.10-slim

# 2. Cài đặt các công cụ biên dịch C++
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt-get/lists/*

# 3. ÉP CHỈ DÙNG 1 LUỒNG BIÊN DỊCH (GIỮ RAM LUÔN < 1GB - QUAN TRỌNG NHẤT)
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV MAKEFLAGS="-j1"

# 4. Tạo thư mục làm việc
WORKDIR /app

# 5. Cài đặt dlib trước để lưu cache Docker
RUN pip install --no-cache-dir dlib

# 6. Cài đặt các thư viện còn lại trong requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Sao chép toàn bộ code
COPY . .

# 8. Chạy Server
CMD gunicorn server:app --bind 0.0.0.0:$PORT

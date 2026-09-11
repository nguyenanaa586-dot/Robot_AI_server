import asyncio
import edge_tts
import miniaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

# Hàm chuyển đổi Văn bản -> PCM 16kHz 16-bit Mono chuẩn cho I2S ESP32-S3
async def text_to_pcm(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice="vi-VN-HoaiMyNeural")
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
            
    if not mp3_bytes:
        return b""
        
    decoded = miniaudio.decode(
        mp3_bytes, 
        output_format=miniaudio.SampleFormat.SIGNED16, 
        nchannels=1, 
        sample_rate=16000
    )
    return decoded.samples.tobytes()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("-> ESP32-S3 đã kết nối WebSocket!")
    try:
        while True:
            # Nhận câu hỏi/dữ liệu từ ESP32
            message = await websocket.receive_text()
            print(f"ESP32 gửi: {message}")
            
            # (Sau này sẽ gọi Gemini API ở đây để lấy response_text)
            response_text = "Chào bạn! Tôi là Bún Đậu robot đây, hệ thống âm thanh đã sẵn sàng!"
            
            # Chuyển đổi thành PCM bytes
            pcm_bytes = await text_to_pcm(response_text)
            print(f"Đã giải mã {len(pcm_bytes)} bytes PCM. Đang truyền xuống ESP32...")
            
            # Gửi PCM theo từng gói nhỏ (2048 bytes) để không gây tràn Ring Buffer của I2S
            CHUNK_SIZE = 2048
            for i in range(0, len(pcm_bytes), CHUNK_SIZE):
                chunk = pcm_bytes[i:i + CHUNK_SIZE]
                await websocket.send_bytes(chunk)
                await asyncio.sleep(0.01) # Nhịp nghỉ 10ms giữ ổn định luồng
                
            # Gửi tín hiệu hoàn tất chuỗi âm thanh
            await websocket.send_text("END_AUDIO")
            print("-> Đã truyền xong toàn bộ luồng âm thanh.")
            
    except WebSocketDisconnect:
        print("<- ESP32-S3 đã ngắt kết nối.")
    except Exception as e:
        print(f"Lỗi WebSocket: {e}")

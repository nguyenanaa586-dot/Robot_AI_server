import asyncio
import edge_tts
import miniaudio

async def test():
    print("1. Đang gọi Edge-TTS...")
    communicate = edge_tts.Communicate("Alo alo 1 2 3 4", voice="vi-VN-HoaiMyNeural")
    mp3_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
            
    print(f"2. Tải MP3 thành công: {len(mp3_bytes)} bytes")
    
    print("3. Đang giải mã PCM...")
    decoded = miniaudio.decode(mp3_bytes, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=16000)
    pcm_bytes = decoded.samples.tobytes()
    print(f"4. Giải mã PCM thành công: {len(pcm_bytes)} bytes")

asyncio.run(test())

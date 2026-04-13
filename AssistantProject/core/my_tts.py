import asyncio
import websockets
import json
import ssl
import subprocess
import os
from typing import Optional

model = "speech-2.8-hd"
file_format = "mp3"


class StreamAudioPlayer:
    def __init__(self):
        self.mpv_process = None

    def start_mpv(self) -> bool:
        """Start MPV player process"""
        try:
            # 使用更简洁的命令行参数
            mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
            self.mpv_process = subprocess.Popen(
                mpv_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("MPV player started")
            return True
        except FileNotFoundError:
            print("Error: mpv not found. Please install mpv")
            return False
        except Exception as e:
            print(f"Failed to start mpv: {e}")
            return False

    def play_audio_chunk(self, hex_audio: str) -> bool:
        """Play audio chunk"""
        try:
            if self.mpv_process and self.mpv_process.stdin:
                audio_bytes = bytes.fromhex(hex_audio)
                self.mpv_process.stdin.write(audio_bytes)
                self.mpv_process.stdin.flush()
                return True
        except Exception as e:
            print(f"Play failed: {e}")
            return False
        return False

    def stop(self):
        """Stop player"""
        if self.mpv_process:
            if self.mpv_process.stdin and not self.mpv_process.stdin.closed:
                self.mpv_process.stdin.close()
            try:
                # 添加超时时间，防止程序挂起
                self.mpv_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("MPV process did not terminate gracefully, forcing termination")
                self.mpv_process.terminate()
                try:
                    self.mpv_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.mpv_process.kill()


async def establish_connection(api_key: str) -> Optional[websockets.WebSocketServerProtocol]:
    """Establish WebSocket connection"""
    url = "wss://api.minimaxi.com/ws/v1/t2a_v2"
    headers = {"Authorization": f"Bearer {api_key}"}

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        ws = await websockets.connect(url, additional_headers=headers, ssl=ssl_context)
        connected = json.loads(await ws.recv())
        if connected.get("event") == "connected_success":
            print("Connection successful")
            return ws
        return None
    except Exception as e:
        print(f"Connection failed: {e}")
        return None


async def start_task(websocket: websockets.WebSocketServerProtocol) -> bool:
    """Send task start request"""
    start_msg = {
        "event": "task_start",
        "model": model,
        "voice_setting": {
            "voice_id": "Chinese (Mandarin)_Mature_Woman",
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "english_normalization": False
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": file_format,
            "channel": 1
        }
    }
    await websocket.send(json.dumps(start_msg))
    response = json.loads(await websocket.recv())
    return response.get("event") == "task_started"


async def continue_task_with_stream_play(
        websocket: websockets.WebSocketServerProtocol,
        text: str,
        player: StreamAudioPlayer
) -> float:
    """Send continue request and stream play audio"""
    await websocket.send(json.dumps({
        "event": "task_continue",
        "text": text
    }))

    chunk_counter = 1
    total_audio_size = 0
    audio_data = b""

    while True:
        try:
            response = json.loads(await websocket.recv())

            # 检查是否有错误
            if "base_resp" in response and response["base_resp"]["status_code"] != 0:
                print(f"API Error: {response['base_resp']['status_msg']}")
                return 10

            if "data" in response and "audio" in response["data"]:
                audio = response["data"]["audio"]
                if audio:
                    print(f"Playing chunk #{chunk_counter}")
                    if player.play_audio_chunk(audio):
                        audio_bytes = bytes.fromhex(audio)
                        total_audio_size += len(audio_bytes)
                        audio_data += audio_bytes
                        chunk_counter += 1

            if response.get("is_final"):
                print(f"Audio synthesis completed: {chunk_counter - 1} chunks")
                if player.mpv_process and player.mpv_process.stdin:
                    player.mpv_process.stdin.close()

                estimated_duration = total_audio_size * 0.0625 / 1000
                wait_time = max(estimated_duration + 5, 10)
                return min(wait_time, 30)  # 限制最大等待时间为30秒

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed unexpectedly")
            break
        except Exception as e:
            print(f"Error during streaming: {e}")
            break

    return 10


async def close_connection(websocket: Optional[websockets.WebSocketServerProtocol]):
    """Close connection"""
    if websocket:
        try:
            await websocket.send(json.dumps({"event": "task_finish"}))
            await websocket.close()
        except Exception:
            pass


async def tts(text: str, api_key: Optional[str] = None):
    """
    文字转语音函数

    Args:
        text: 要转换的文本
        api_key: Minimax API密钥，如果未提供则从环境变量获取
    """
    # 优先使用传入的API key，否则从环境变量获取
    if not api_key:
        api_key = os.getenv("MINIMAX_TTS_KEY")

    if not api_key:
        print("⚠️ 未配置 MINIMAX_TTS_KEY，跳过语音播报。")
        return

    player = StreamAudioPlayer()

    try:
        if not player.start_mpv():
            return

        ws = await establish_connection(api_key)
        if not ws:
            return

        if not await start_task(ws):
            print("Task startup failed")
            return

        wait_time = await continue_task_with_stream_play(ws, text, player)
        await asyncio.sleep(wait_time)

    except Exception as e:
        print(f"Error in TTS process: {e}")
    finally:
        player.stop()
        if 'ws' in locals():
            await close_connection(ws)


def main():
    # 可以通过环境变量或直接在代码中设置API密钥
    api_key = os.getenv("MINIMAX_TTS_KEY")
    if not api_key:
        print("Please set MINIMAX_TTS_KEY environment variable")
        return

    text = "你好！有什么可以帮你的吗？"
    asyncio.run(tts(text, api_key))


if __name__ == "__main__":
    main()

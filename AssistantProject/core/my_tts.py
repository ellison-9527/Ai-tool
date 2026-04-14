# core/my_tts.py
import asyncio
import websockets
import json
import ssl
import subprocess
import os
from typing import Optional

model = "speech-2.8-hd"
file_format = "mp3"

ACTIVE_PLAYER = None
FORCE_STOP = False


class StreamAudioPlayer:
    def __init__(self):
        self.mpv_process = None

    def start_mpv(self) -> bool:
        try:
            mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
            kwargs = {}
            # 解决 Windows 下可能弹出黑窗口或后台杀不掉的问题
            if os.name == 'nt':
                kwargs['creationflags'] = 0x08000000  # subprocess.CREATE_NO_WINDOW

            self.mpv_process = subprocess.Popen(
                mpv_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs
            )
            return True
        except Exception as e:
            print(f"启动 mpv 失败: {e}，请确保系统已安装 mpv")
            return False

    def play_audio_chunk(self, hex_audio: str) -> bool:
        try:
            if self.mpv_process and self.mpv_process.poll() is None:
                audio_bytes = bytes.fromhex(hex_audio)
                self.mpv_process.stdin.write(audio_bytes)
                self.mpv_process.stdin.flush()
                return True
        except Exception:
            return False
        return False

    def stop(self):
        if self.mpv_process:
            try:
                if self.mpv_process.stdin:
                    self.mpv_process.stdin.close()
            except Exception:
                pass
            try:
                self.mpv_process.terminate()
                self.mpv_process.kill()
            except Exception:
                pass
            self.mpv_process = None


def stop_current_tts():
    global ACTIVE_PLAYER, FORCE_STOP
    FORCE_STOP = True
    if ACTIVE_PLAYER:
        ACTIVE_PLAYER.stop()
        ACTIVE_PLAYER = None


async def establish_connection(api_key: str) -> Optional[websockets.WebSocketServerProtocol]:
    url = "wss://api.minimaxi.com/ws/v1/t2a_v2"
    headers = {"Authorization": f"Bearer {api_key}"}
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        ws = await websockets.connect(url, additional_headers=headers, ssl=ssl_context)
        connected = json.loads(await ws.recv())
        if connected.get("event") == "connected_success":
            return ws
        return None
    except Exception:
        return None


async def start_task(websocket: websockets.WebSocketServerProtocol) -> bool:
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
    global FORCE_STOP
    await websocket.send(json.dumps({
        "event": "task_continue",
        "text": text
    }))

    total_audio_size = 0

    while True:
        if FORCE_STOP:
            break

        try:
            response_str = await asyncio.wait_for(websocket.recv(), timeout=0.5)
            response = json.loads(response_str)

            if "base_resp" in response and response["base_resp"]["status_code"] != 0:
                print(f"API 报错: {response['base_resp']['status_msg']}")
                break

            if "data" in response and "audio" in response["data"]:
                audio = response["data"]["audio"]
                if audio:
                    if not player.play_audio_chunk(audio):
                        break
                    total_audio_size += len(bytes.fromhex(audio))

            if response.get("is_final"):
                break

        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            # 捕获 UI 层传来的任务取消信号
            print("🛑 后台 TTS 任务被强制取消")
            break
        except Exception as e:
            print(f"网络流中断: {e}")
            break

    estimated_duration = total_audio_size * 0.0625 / 1000
    return min(max(estimated_duration, 2), 30)


async def tts(text: str, api_key: Optional[str] = None):
    global ACTIVE_PLAYER, FORCE_STOP

    if not api_key:
        api_key = os.getenv("MINIMAX_TTS_KEY")

    if not api_key:
        return

    stop_current_tts()
    FORCE_STOP = False

    player = StreamAudioPlayer()
    ACTIVE_PLAYER = player
    ws = None

    try:
        if not player.start_mpv():
            return
        ws = await establish_connection(api_key)
        if not ws:
            return
        if not await start_task(ws):
            return

        wait_time = await continue_task_with_stream_play(ws, text, player)

        # 自然等待期间，也能响应取消指令
        for _ in range(int(wait_time * 10)):
            if FORCE_STOP:
                break
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        pass  # 被取消是正常业务逻辑，不打印报错
    except Exception as e:
        print(f"TTS 运行异常: {e}")
    finally:
        player.stop()
        if ACTIVE_PLAYER == player:
            ACTIVE_PLAYER = None
        if ws:
            try:
                await ws.send(json.dumps({"event": "task_finish"}))
                await ws.close()
            except Exception:
                pass
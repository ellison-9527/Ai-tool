import re
import os
import urllib.parse
import base64
import mimetypes
from AssistantProject.core.agent import get_asr_text

def generate_chat_title(state_messages):
    """根据首条消息生成对话标题"""
    title = "新对话"
    if state_messages:
        first_msg = state_messages[0]["content"]
        if isinstance(first_msg, str):
            title = first_msg
        elif isinstance(first_msg, list):
            for item in first_msg:
                if item.get("type") == "text" and item.get("text").strip():
                    title = item.get("text")
                    break
    title = re.sub(r'[\\/*?:"<>|\n\r]', "", title).strip()
    return title[:15] if title else "图片对话"


def process_user_input(user_text, files, history, state_messages):
    """处理用户输入的多模态数据（文本、图片、音频），组装进历史记录"""
    if not user_text.strip() and not files:
        return False, history, state_messages, None

    if files:
        mutil_content = []
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                history.append({"role": "user", "content": {"path": file}})
                with open(file, "rb") as img_file:
                    img_base = base64.b64encode(img_file.read()).decode("utf-8")
                    mime_type, _ = mimetypes.guess_type(file)
                    mime_type = mime_type or "image/jpeg"
                    mutil_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base}"}})
            elif file.lower().endswith(('.wav', '.mp3', '.m4a', '.webm', '.ogg')):
                user_audio_text = get_asr_text(file)
                user_text = (user_text + " " + user_audio_text).strip()
        mutil_content.append({"type": "text", "text": user_text})
        state_messages.append({"role": "user", "content": mutil_content})
        user_input_for_agent = mutil_content
    else:
        state_messages.append({"role": "user", "content": user_text})
        user_input_for_agent = user_text

    history.append({'role': 'user', 'content': user_text})
    history.append({"role": "assistant", "content": "正在思考中...", "metadata": {"title": "🤔 思考中"}})
    
    return True, history, state_messages, user_input_for_agent


def extract_markdown_images(raw_text):
    """从大模型生成的 Markdown 文本中提取本地图片路径，并净化文本"""
    extracted_images = []

    def extract_and_remove(match):
        filepath = match.group(2)
        if filepath.startswith("/file="): filepath = filepath[6:]
        filepath = urllib.parse.unquote(filepath)

        if os.path.exists(filepath):
            extracted_images.append(filepath)
        return ""  # 从纯文本气泡中删除图片占位符

    clean_final_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', extract_and_remove, raw_text).strip()
    return clean_final_text, extracted_images

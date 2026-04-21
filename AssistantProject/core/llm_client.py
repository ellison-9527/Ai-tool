# core/llm_client.py
import os
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv
from AssistantProject.core.logger import logger

load_dotenv(override=True)

# 兼容两种环境变量命名方式，获取 API Key
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("BASE_URL")
)


def get_asr_text(file_path):
    """调用智谱 ASR 接口将语音转为文字"""
    logger.info(f"🎤 [多模态] 开始处理音频文件: {file_path}")
    url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    
    # ASR 强制使用专属的智谱 Key
    zhipu_key = os.getenv("ZHIPU_API_KEY") or api_key
    headers = {
        "Authorization": zhipu_key 
    }
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            # 必须带上 model 和 stream 参数
            data = {"model": "glm-asr-2512", "stream": "false"}
            response = requests.post(url, headers=headers, data=data, files=files)
            result = response.json()

            text = result.get("text", "")
            logger.info(f"✅ [多模态] 语音识别成功: {text}")
            return text
    except Exception as e:
        logger.error(f"❌ [多模态] 语音识别出错: {e}")
        return ""


def encode_image(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_multimodal_chat(message_dict, history, sys_prompt, max_token, temp):
    """处理包含文本、图片和语音的多模态输入"""
    text = message_dict.get("text", "")
    files = message_dict.get("files", [])

    if not text and not files:
        return {"text": "", "files": []}, history

    # 扩充音频后缀的白名单，防止网页录音被跳过
    audio_exts = ('.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac')
    image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp')

    voice_text = ""
    actual_images = []

    for f in files:
        f_lower = f.lower()
        if f_lower.endswith(audio_exts):
            recognized_text = get_asr_text(f)
            if recognized_text:
                voice_text += f" {recognized_text} "
        elif f_lower.endswith(image_exts):
            actual_images.append(f)

    combined_text = (voice_text + " " + text).strip()

    if not combined_text and not actual_images:
        return {"text": "", "files": []}, history

    # 更新前端页面展示 (历史记录)
    history.append({"role": "user", "content": combined_text if combined_text else "上传了图片"})
    for img_path in actual_images:
        history.append({"role": "user", "content": (img_path,)})

    messages = []
    if sys_prompt.strip():
        messages.append({"role": "system", "content": sys_prompt})

    for msg in history[:-1 - len(actual_images)]:
        if isinstance(msg["content"], str):
            messages.append(msg)

    # ==========================================
    # 核心：智能模型路由 (Dynamic Model Routing)
    # ==========================================
    if not actual_images:
        # 场景 A：没有图片，强制使用纯文本模型，发送字符串格式
        target_model = "glm-4-plus"
        messages.append({"role": "user", "content": combined_text})
        logger.info(f"🧠 [模型路由] 未检测到图片，使用文本模型: {target_model}")
    else:
        # 场景 B：有图片，使用视觉模型，发送多模态列表格式
        target_model = os.getenv("MODEL_NAME", "glm-4-plus")
        current_content = []
        if combined_text:
            current_content.append({"type": "text", "text": combined_text})

        for img_path in actual_images:
            base64_img = encode_image(img_path)
            current_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        messages.append({"role": "user", "content": current_content})
        logger.info(f"👁️ [模型路由] 检测到图片，使用视觉模型: {target_model}")

    try:
        response = client.chat.completions.create(
            model=target_model,  # 动态传入选中的模型
            messages=messages,
            max_tokens=int(max_token),
            temperature=float(temp)
        )
        ai_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": ai_reply})

    except Exception as e:
        error_msg = f"⚠️ 交互失败: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})

    return {"text": "", "files": []}, history
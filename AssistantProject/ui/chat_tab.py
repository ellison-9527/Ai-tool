# ui/chat_tab.py
import base64
import re
import uuid
import mimetypes
import gradio as gr

from AssistantProject.core.my_tts import tts
from AssistantProject.core.agent import simple_agent_chat, get_asr_text
from AssistantProject.core.rag_manager import get_kb_list
from AssistantProject.core.db_manager import get_all_sessions, load_session, save_session, delete_session

# ... (后面的代码全都不用动) ...


def generate_chat_title(state_messages):
    """根据第一条消息自动生成对话标题"""
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


def start_new_chat():
    """新建对话：清空屏幕和游标，刷新列表"""
    return [], {"text": "", "files": []}, [], None, gr.update(choices=get_all_sessions(), value=None)


async def play_latest_tts(history):
    """抓取最后一条 AI 回复并进行语音播报"""
    if not history:
        gr.Warning("暂无对话可播报！")
        return

    # 倒序查找最后一条 assistant 的回复
    for msg in reversed(history):
        if msg["role"] == "assistant":
            text = msg["content"]
            if text and "🤔 思考中" not in text:
                gr.Info("🔊 正在为您生成语音，请稍候...")
                await tts(text)
                return

    gr.Warning("没有找到可以播报的 AI 回复！")
def ui_delete_chat(display_name, current_chat_id):
    """UI删除交互：删除数据库中的记录并按需清屏"""
    if not display_name:
        gr.Warning("请先在列表中选择要删除的对话！")
        return gr.update(), gr.update(), gr.update(), gr.update(), current_chat_id

    delete_session(display_name)
    gr.Info("对话已成功删除")

    # 如果删除的恰好是当前正在看的对话，那就清空屏幕
    if current_chat_id and current_chat_id in display_name:
        return [], {"text": "", "files": []}, [], gr.update(choices=get_all_sessions(), value=None), None
    else:
        # 否则只刷新列表，屏幕保留
        return gr.update(), gr.update(), gr.update(), gr.update(choices=get_all_sessions(), value=None), current_chat_id


async def chat(message, history, state_messages, system_prompt, max_token, temperature, kb_name, target_model, current_chat_id):
    """聊天主函数，集成了多模型聚合与 SQLite 自动保存"""
    user = message.get("text", "")
    files = message.get("files", [])

    if not user.strip() and not files:
        gr.Warning("不能发送空消息哦！")
        yield history, {"text": "", "files": []}, state_messages, current_chat_id
        return

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
                user = (user + " " + user_audio_text).strip()

        mutil_content.append({"type": "text", "text": user})
        state_messages.append({"role": "user", "content": mutil_content})
    else:
        state_messages.append({"role": "user", "content": user})

    history.append({'role': 'user', 'content': user})
    history.append({"role": "assistant", "content": "正在思考中...", "metadata": {"title": "🤔 思考中"}})

    # 【核心：使用 8 位短 UUID 作为数据库主键】
    if not current_chat_id:
        current_chat_id = uuid.uuid4().hex[:8]

    yield history, {"text": "", "files": []}, state_messages, current_chat_id

    try:
        is_first_response = True
        async for event in simple_agent_chat(state_messages, system_prompt, max_token, temperature, target_model, kb_name):
            if is_first_response:
                if history and history[-1].get("metadata", {}).get("title") == "🤔 思考中":
                    history.pop()
                is_first_response = False

            if event["type"] == "tool_call":
                history.append({
                    "role": "assistant",
                    "content": "",
                    "metadata": {"title": f"调用工具: {event['name']}({event['args']})", "status": "pending"}
                })
                yield history, gr.update(), state_messages, current_chat_id
            elif event["type"] == "tools_result":
                if history[-1].get("metadata"):
                    history[-1]["content"] += f"\n\n--- 返回结果 ---\n{event['content']}"
                    history[-1]["metadata"]["status"] = "done"
                    yield history, gr.update(), state_messages, current_chat_id
            elif event["type"] == "token":
                if not history or history[-1]["role"] != "assistant" or history[-1].get("metadata") is not None:
                    history.append({"role": "assistant", "content": ""})
                history[-1]["content"] += event["content"]
                yield history, gr.update(), state_messages, current_chat_id

        state_messages.append({"role": "assistant", "content": history[-1]["content"]})
        # 保存到 SQLite 数据库
        title = generate_chat_title(state_messages)
        save_session(current_chat_id, title, history, state_messages)

        # 【核心：静默保存到 SQLite 数据库】
        title = generate_chat_title(state_messages)
        save_session(current_chat_id, title, history, state_messages)
        # ✅ 【重磅新增】：当文本全部生成完毕后，如果勾选了 TTS，则触发语音播报

    except Exception as e:
        if history and history[-1].get("metadata", {}).get("title") == "🤔 思考中":
            history.pop()
        history.append({"role": "assistant", "content": f"⚠️ 处理失败: {str(e)}"})

    yield history, gr.update(), state_messages, current_chat_id


def refresh_kb_choices():
    return gr.update(choices=get_kb_list())


def create_chat_tab():
    # 隐藏状态，用于追踪数据库中的 session_id
    current_chat_id = gr.State(None)
    state_messages = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            session_btn = gr.Button("+ 新建对话", variant="primary")
            gr.Markdown("---")
            # 这里的下拉框直接从数据库获取数据
            history_dropdown = gr.Dropdown(choices=get_all_sessions(), label="📂 历史对话记录", interactive=True)
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新", size="sm")
                delete_btn = gr.Button("🗑️ 删除", size="sm", variant="stop")

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label='agent', avatar_images=("./assert/user.png", "./assert/bot.png"), height=600)
            with gr.Row():
                tts_btn = gr.Button("🔊 朗读最新回复", size="sm", variant="secondary")

            chat_input = gr.MultimodalTextbox(
                file_types=["image", "audio"],
                file_count="multiple",
                placeholder="请输入消息，或点击右侧上传图片/录音...",
                show_label=False,
                sources=["upload", "microphone"]
            )

        with gr.Column(scale=2):
            # 模型切换列表（前端展示名，后端真实名）
            model_dropdown = gr.Dropdown(
                choices=[
                    ("智谱 GLM-5.1", "glm-4-flash"),
                    ("通义千问 Qwen-Max", "qwen-max"),
                    ("MiniMax 2.5", "abab6.5s-chat")
                ],
                label="🧠 切换大模型",
                value="qwen-max",
                info="通过 Higress 网关统一调用"
            )
            kb_dropdown = gr.Dropdown(choices=get_kb_list(), label="📚 挂载知识库 (可选)",
                                      info="选择后，AI 回答时将查阅此库")

            with gr.Row():
                refresh_kb_btn = gr.Button("🔄 刷新知识库", size="sm", variant="secondary")
                clear_kb_btn = gr.Button("❌ 取消挂载", size="sm", variant="stop")

            gr.Markdown("---")
            system_prompt = gr.Text(label="系统提示词", lines=2, value="你是一个精通编程和图像识别的 AI 助手。")
            max_token = gr.Number(label="Maxtoken", value=4096, interactive=True)
            temperature = gr.Number(label="temperature", value=0.5, interactive=True)

    # --- 事件绑定 ---
    # 1. 聊天发送
    chat_input.submit(
        fn=chat,
        inputs=[
            chat_input, chatbot, state_messages,
            system_prompt, max_token, temperature,
            kb_dropdown, model_dropdown, current_chat_id
        ],
        outputs=[chatbot, chat_input, state_messages, current_chat_id]
    )

    # 2. 新建对话
    session_btn.click(
        fn=start_new_chat,
        inputs=[],
        outputs=[chatbot, chat_input, state_messages, current_chat_id, history_dropdown]
    )

    # 3. 读取数据库历史记录
    history_dropdown.select(
        fn=load_session,
        inputs=[history_dropdown],
        outputs=[chatbot, state_messages, current_chat_id]
    )

    # 4. 删除历史记录
    delete_btn.click(
        fn=ui_delete_chat,
        inputs=[history_dropdown, current_chat_id],
        outputs=[chatbot, chat_input, state_messages, history_dropdown, current_chat_id]
    )

    # 5. 刷新与取消
    refresh_btn.click(fn=lambda: gr.update(choices=get_all_sessions()), inputs=[], outputs=[history_dropdown])
    refresh_kb_btn.click(fn=refresh_kb_choices, inputs=[], outputs=[kb_dropdown])
    clear_kb_btn.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[kb_dropdown])
    tts_btn.click(
        fn=play_latest_tts,
        inputs=[chatbot],  # 把聊天记录传给函数提取文字
        outputs=[]
    )
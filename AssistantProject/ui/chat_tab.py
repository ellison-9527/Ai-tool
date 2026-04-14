# ui/chat_tab.py
import base64
import re
import uuid
import mimetypes
import asyncio
import json
import os
import gradio as gr

from AssistantProject.core.my_tts import tts, stop_current_tts
from AssistantProject.core.agent import simple_agent_chat, get_asr_text
from AssistantProject.core.rag_manager import get_kb_list
from AssistantProject.core.db_manager import get_all_sessions, load_session, save_session, delete_session


# ================= 工具函数：读取动态 Skill 配置 =================
def get_dynamic_skills():
    """从本地 JSON 文件读取最新的 Skill 列表"""
    try:
        with open("data/skills.json", "r", encoding="utf-8") as f:
            skills = json.load(f)
            return list(skills.keys())
    except Exception:
        # 如果还没创建 JSON，给一个空列表保底
        return []


def get_skill_prompts(selected_names):
    """根据选中的专家名称，提取他们对应的 Prompt"""
    try:
        with open("data/skills.json", "r", encoding="utf-8") as f:
            skills = json.load(f)
            return [skills[name]["prompt"] for name in selected_names if name in skills]
    except Exception:
        return []


# =============================================================

CURRENT_PLAYING_TEXT = None
TTS_TASK = None


def generate_chat_title(state_messages):
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
    return [], {"text": "", "files": []}, [], None, gr.update(choices=get_all_sessions(), value=None)


async def toggle_message_tts(evt: gr.SelectData):
    """基于后台任务的点读机逻辑"""
    global CURRENT_PLAYING_TEXT, TTS_TASK
    clicked_text = evt.value

    if not isinstance(clicked_text, str) or not clicked_text.strip() or "🤔 思考中" in clicked_text:
        return

    if CURRENT_PLAYING_TEXT == clicked_text:
        stop_current_tts()
        if TTS_TASK and not TTS_TASK.done():
            TTS_TASK.cancel()
        CURRENT_PLAYING_TEXT = None
        gr.Info("🛑 语音播报已停止")
        return

    if CURRENT_PLAYING_TEXT:
        stop_current_tts()
        if TTS_TASK and not TTS_TASK.done():
            TTS_TASK.cancel()

    CURRENT_PLAYING_TEXT = clicked_text
    gr.Info("🔊 开始播报，再次点击该文本即可中断...")
    clean_text = clicked_text.replace("\n", " ")[:2000]

    async def background_tts():
        global CURRENT_PLAYING_TEXT
        try:
            await tts(clean_text)
        except asyncio.CancelledError:
            pass
        finally:
            if CURRENT_PLAYING_TEXT == clicked_text:
                CURRENT_PLAYING_TEXT = None

    TTS_TASK = asyncio.create_task(background_tts())


def ui_delete_chat(display_name, current_chat_id):
    if not display_name:
        gr.Warning("请先在列表中选择要删除的对话！")
        return gr.update(), gr.update(), gr.update(), gr.update(), current_chat_id

    delete_session(display_name)
    gr.Info("对话已成功删除")

    if current_chat_id and current_chat_id in display_name:
        return [], {"text": "", "files": []}, [], gr.update(choices=get_all_sessions(), value=None), None
    else:
        return gr.update(), gr.update(), gr.update(), gr.update(choices=get_all_sessions(), value=None), current_chat_id


# 【核心修改】：接收前端传来的 agent_mode 和 selected_skills
async def chat(message, history, state_messages, system_prompt, max_token, temperature, kb_name, target_model,
               current_chat_id, agent_mode, selected_skills):
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

    if not current_chat_id:
        current_chat_id = uuid.uuid4().hex[:8]

    yield history, {"text": "", "files": []}, state_messages, current_chat_id

    # ================== 多智能体 Prompt 融合逻辑 ==================
    final_sys_prompt = system_prompt
    if agent_mode == "👥 专家团队模式 (多智能体)" and selected_skills:
        expert_prompts = get_skill_prompts(selected_skills)
        if expert_prompts:
            # 将所有选中的专家 Prompt 拼接起来，赋予 AI 多重人格
            combined_experts = "\n\n".join([f"【专家角色设定】:\n{p}" for p in expert_prompts])
            final_sys_prompt += f"\n\n你现在需要扮演一个多专家协作团队。以下是你拥有的子团队成员设定，请综合他们的专业视角，分步骤、全面地解答用户的问题：\n{combined_experts}"
    # ============================================================

    try:
        is_first_response = True
        # 【修改】：将融合后的 final_sys_prompt 传给核心 Agent
        async for event in simple_agent_chat(state_messages, final_sys_prompt, max_token, temperature, target_model,
                                             kb_name):
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
        title = generate_chat_title(state_messages)
        save_session(current_chat_id, title, history, state_messages)

    except Exception as e:
        if history and history[-1].get("metadata", {}).get("title") == "🤔 思考中":
            history.pop()
        history.append({"role": "assistant", "content": f"⚠️ 处理失败: {str(e)}"})

    yield history, gr.update(), state_messages, current_chat_id


def refresh_kb_choices():
    return gr.update(choices=get_kb_list())


def create_chat_tab():
    current_chat_id = gr.State(None)
    state_messages = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            session_btn = gr.Button("+ 新建对话", variant="primary")
            gr.Markdown("---")
            history_dropdown = gr.Dropdown(choices=get_all_sessions(), label="📂 历史对话记录", interactive=True)
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新", size="sm")
                delete_btn = gr.Button("🗑️ 删除", size="sm", variant="stop")

        with gr.Column(scale=6):
            gr.Markdown("💡 **交互提示:** 点击聊天框中的任意一条回答文字，即可开始语音播报；播报中再次点击即可停止。")
            chatbot = gr.Chatbot(label='agent', avatar_images=("./assert/user.png", "./assert/bot.png"), height=600)

            chat_input = gr.MultimodalTextbox(
                file_types=["image", "audio"],
                file_count="multiple",
                placeholder="请输入消息，或点击右侧上传图片/录音...",
                show_label=False,
                sources=["upload", "microphone"]
            )

        with gr.Column(scale=2):
            agent_mode = gr.Radio(
                choices=["🤖 单体全能模式", "👥 专家团队模式 (多智能体)"],
                label="Agent 运行模式",
                value="🤖 单体全能模式"
            )

            with gr.Group(visible=False) as multi_agent_group:
                gr.Markdown("#### 👨‍💻 参与协作的专家团队")
                # 【动态加载】：启动时从 json 加载技能列表
                multi_agent_checkboxes = gr.CheckboxGroup(
                    choices=get_dynamic_skills(),
                    value=[],
                    label="选择本局对话启用的 Skill",
                    interactive=True
                )
                refresh_skills_btn = gr.Button("🔄 刷新 Skill 列表", size="sm")
            gr.Markdown("---")

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

            with gr.Accordion("⚙️ 高级对话设置", open=False):
                system_prompt = gr.Text(label="系统提示词", lines=2, value="你是一个精通编程和图像识别的 AI 助手。")
                max_token = gr.Number(label="Maxtoken", value=4096, interactive=True)
                temperature = gr.Number(label="temperature", value=0.5, interactive=True)

    # --- 界面动态交互逻辑 ---
    def toggle_agent_mode(mode):
        if mode == "👥 专家团队模式 (多智能体)":
            return gr.update(visible=True)
        return gr.update(visible=False)

    agent_mode.change(
        fn=toggle_agent_mode,
        inputs=[agent_mode],
        outputs=[multi_agent_group]
    )

    # 【新增刷新】：点击时从 json 重新读取最新数据
    refresh_skills_btn.click(
        fn=lambda: gr.update(choices=get_dynamic_skills()),
        inputs=[],
        outputs=[multi_agent_checkboxes]
    )

    # 【核心修改】：提交对话时，带上模式和选中的专家列表
    chat_input.submit(
        fn=chat,
        inputs=[
            chat_input, chatbot, state_messages,
            system_prompt, max_token, temperature,
            kb_dropdown, model_dropdown, current_chat_id,
            agent_mode, multi_agent_checkboxes  # 新增两个传参
        ],
        outputs=[chatbot, chat_input, state_messages, current_chat_id]
    )

    chatbot.select(
        fn=toggle_message_tts,
        inputs=[],
        outputs=[]
    )

    session_btn.click(
        fn=start_new_chat,
        inputs=[],
        outputs=[chatbot, chat_input, state_messages, current_chat_id, history_dropdown]
    )

    history_dropdown.select(
        fn=load_session,
        inputs=[history_dropdown],
        outputs=[chatbot, state_messages, current_chat_id]
    )

    delete_btn.click(
        fn=ui_delete_chat,
        inputs=[history_dropdown, current_chat_id],
        outputs=[chatbot, chat_input, state_messages, history_dropdown, current_chat_id]
    )

    refresh_btn.click(fn=lambda: gr.update(choices=get_all_sessions()), inputs=[], outputs=[history_dropdown])
    refresh_kb_btn.click(fn=refresh_kb_choices, inputs=[], outputs=[kb_dropdown])
    clear_kb_btn.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[kb_dropdown])
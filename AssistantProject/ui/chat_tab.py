# AssistantProject/ui/chat_tab.py
import base64
import re
import uuid
import mimetypes
import asyncio
import os
import urllib.parse
import gradio as gr

from AssistantProject.core.my_tts import tts, stop_current_tts
from AssistantProject.core.agent import simple_agent_chat, get_asr_text
from AssistantProject.core.rag_manager import get_kb_list
from AssistantProject.core.db_manager import get_all_sessions, load_session, save_session, delete_session
from AssistantProject.core.skill_manager import get_skill_prompts_map, get_skill_choices as get_dynamic_skills
from AssistantProject.core.logger import logger
from AssistantProject.core.utils import generate_chat_title, process_user_input, extract_markdown_images

CURRENT_PLAYING_TEXT = None
TTS_TASK = None





def start_new_chat():
    return [], {"text": "", "files": []}, [], None, gr.update(choices=get_all_sessions(), value=None)


async def toggle_message_tts(evt: gr.SelectData):
    global CURRENT_PLAYING_TEXT, TTS_TASK
    clicked_text = evt.value
    if not isinstance(clicked_text, str) or not clicked_text.strip() or "🤔 思考中" in clicked_text:
        return
    if CURRENT_PLAYING_TEXT == clicked_text:
        stop_current_tts()
        if TTS_TASK and not TTS_TASK.done(): TTS_TASK.cancel()
        CURRENT_PLAYING_TEXT = None
        gr.Info("🛑 语音播报已停止")
        return
    if CURRENT_PLAYING_TEXT:
        stop_current_tts()
        if TTS_TASK and not TTS_TASK.done(): TTS_TASK.cancel()

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


async def chat(message, history, state_messages, system_prompt, max_token, temperature, kb_name, target_model,
               current_chat_id, agent_mode, selected_skills):
    user_text = message.get("text", "")
    files = message.get("files", [])

    is_valid, history, state_messages, user_input_for_agent = process_user_input(user_text, files, history, state_messages)
    if not is_valid:
        gr.Warning("不能发送空消息哦！")
        yield history, {"text": "", "files": []}, state_messages, current_chat_id
        return

    if not current_chat_id:
        current_chat_id = uuid.uuid4().hex[:8]

    yield history, {"text": "", "files": []}, state_messages, current_chat_id

    final_sys_prompt = system_prompt
    expert_prompts_map = None
    if agent_mode == "👥 专家团队模式 (多智能体)" and selected_skills:
        expert_prompts_map = get_skill_prompts_map(selected_skills)

    try:
        is_first_response = True
        async for event in simple_agent_chat(
                user_message_content=user_input_for_agent,
                sys_prompt=final_sys_prompt,
                max_token=max_token,
                temperature=temperature,
                target_model=target_model,
                thread_id=current_chat_id,
                kb_name=kb_name,
                expert_prompts_map=expert_prompts_map
        ):
            if is_first_response:
                if history and history[-1].get("metadata", {}).get("title") == "🤔 思考中":
                    history.pop()
                is_first_response = False

            if event["type"] == "tool_call":
                history.append({
                    "role": "assistant",
                    "content": "",
                    "metadata": {"title": f"调用工具: {event['name']}", "status": "pending"}
                })
                yield history, gr.update(), state_messages, current_chat_id

            elif event["type"] == "tools_result":
                if history and history[-1].get("metadata"):
                    tool_content = event.get('content', '')
                    if isinstance(tool_content, list):
                        texts = [item.get("text", "") for item in tool_content if item.get("type") == "text"]
                        tool_content = "\n".join(texts)

                    # 🌟【防重复渲染】：在中间工具环节，强制剥离图片 Markdown，只显示文本提示
                    clean_tool_ui = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', '*(📊 图表已在后台生成)*', str(tool_content))
                    history[-1]["content"] += f"\n\n--- 返回结果 ---\n{clean_tool_ui}"
                    history[-1]["metadata"]["status"] = "done"
                    yield history, gr.update(), state_messages, current_chat_id

            elif event["type"] == "token":
                # 如果上一条消息是原生的字典图片，必须开个新气泡存放文字
                if not history or history[-1]["role"] != "assistant" or history[-1].get(
                        "metadata") is not None or isinstance(history[-1]["content"], dict):
                    history.append({"role": "assistant", "content": ""})
                history[-1]["content"] += (event.get("content") or "")
                yield history, gr.update(), state_messages, current_chat_id

        # ==========================================
        # 🟢 最终收尾：提取路径，转换为原生画廊多模态组件
        # ==========================================
        if history and history[-1]["role"] == "assistant" and isinstance(history[-1]["content"], str):
            raw_text = history[-1]["content"]
            state_messages.append({"role": "assistant", "content": raw_text})

            clean_final_text, extracted_images = extract_markdown_images(raw_text)

            if clean_final_text:
                history[-1]["content"] = clean_final_text
            else:
                history.pop()  # 如果过滤后不剩文字了，删掉空文本气泡

            # 🌟【终极多模态】：交给 Gradio 官方渲染图片组件（自带全屏放大和下载）
            for img in extracted_images:
                history.append({"role": "assistant", "content": {"path": img}})

            yield history, gr.update(), state_messages, current_chat_id

        else:
            if history and not isinstance(history[-1]["content"], dict):
                state_messages.append({"role": "assistant", "content": "⚠️ 代理未返回任何有效内容。"})

        title = generate_chat_title(state_messages)
        save_session(current_chat_id, title, history, state_messages)

    except Exception as e:
        logger.error(f"❌ UI 层聊天代理发生异常: {str(e)}", exc_info=True)
        if history and history[-1].get("metadata", {}).get("title") == "🤔 思考中":
            history.pop()
        history.append({"role": "assistant", "content": f"⚠️ 系统内部错误: {str(e)}"})

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
            gr.Markdown("💡 **交互提示:** 点击聊天框文字可语音播报；生成的图表点击即可全屏缩放。")
            # 开启原生多模态组件支持
            chatbot = gr.Chatbot(label='agent', avatar_images=("./assert/user.png", "./assert/bot.png"), height=600)

            with gr.Row():
                chat_input = gr.MultimodalTextbox(
                    scale=9,
                    file_types=["image", "audio"],
                    file_count="multiple",
                    placeholder="请输入消息，或点击右侧上传图片/录音...",
                    show_label=False,
                    sources=["upload", "microphone"]
                )
                stop_btn = gr.Button("🛑 停止", variant="stop", visible=False, scale=1, min_width=60)

        with gr.Column(scale=2):
            agent_mode = gr.Radio(choices=["🤖 单体全能模式", "👥 专家团队模式 (多智能体)"], label="Agent 运行模式",
                                  value="🤖 单体全能模式")

            with gr.Group(visible=False) as multi_agent_group:
                gr.Markdown("#### 👨‍💻 参与协作的专家团队")
                multi_agent_checkboxes = gr.CheckboxGroup(choices=get_dynamic_skills(), value=[],
                                                          label="选择本局对话启用的 Skill", interactive=True)
                refresh_skills_btn = gr.Button("🔄 刷新 Skill 列表", size="sm")
            gr.Markdown("---")

            model_dropdown = gr.Dropdown(
                choices=[("智谱 GLM-4-Plus (最新)", "glm-4-plus"), ("通义千问 Qwen-Max", "qwen-max"),
                         ("MiniMax 2.5", "abab6.5s-chat")],
                label="🧠 切换大模型", value="qwen-max", info="通过 Higress 网关统一调用"
            )
            kb_dropdown = gr.Dropdown(choices=get_kb_list(), label="📚 挂载知识库 (可选)",
                                      info="选择后，AI 回答时将查阅此库")

            with gr.Row():
                refresh_kb_btn = gr.Button("🔄 刷新知识库", size="sm", variant="secondary")
                clear_kb_btn = gr.Button("❌ 取消挂载", size="sm", variant="stop")

            gr.Markdown("---")

            with gr.Accordion("⚙️ 高级对话设置", open=False):
                system_prompt = gr.Text(label="系统提示词", lines=2, value="你是一个精通编程和数据分析的 AI 助手。")
                max_token = gr.Number(label="Maxtoken", value=4096, interactive=True)
                temperature = gr.Number(label="temperature", value=0.5, interactive=True)

    def toggle_agent_mode(mode):
        return gr.update(visible=True) if mode == "👥 专家团队模式 (多智能体)" else gr.update(visible=False)

    agent_mode.change(fn=toggle_agent_mode, inputs=[agent_mode], outputs=[multi_agent_group])
    refresh_skills_btn.click(fn=lambda: gr.update(choices=get_dynamic_skills()), inputs=[],
                             outputs=[multi_agent_checkboxes])

    # ==============================
    # 交互事件：发送与停止逻辑
    # ==============================
    generate_event = chat_input.submit(
        fn=lambda: (gr.update(interactive=False), gr.update(visible=True)),
        inputs=[],
        outputs=[chat_input, stop_btn]
    ).then(
        fn=chat,
        inputs=[
            chat_input, chatbot, state_messages,
            system_prompt, max_token, temperature,
            kb_dropdown, model_dropdown, current_chat_id,
            agent_mode, multi_agent_checkboxes
        ],
        outputs=[chatbot, chat_input, state_messages, current_chat_id]
    )

    # 成功生成完后恢复按钮状态
    generate_event.then(
        fn=lambda: (gr.update(interactive=True), gr.update(visible=False)),
        inputs=[],
        outputs=[chat_input, stop_btn]
    )

    # 点击停止按钮中断生成，并恢复状态
    stop_btn.click(
        fn=lambda: (gr.update(interactive=True), gr.update(visible=False)),
        inputs=[],
        outputs=[chat_input, stop_btn],
        cancels=[generate_event]
    )

    chatbot.select(fn=toggle_message_tts, inputs=[], outputs=[])
    session_btn.click(fn=start_new_chat, inputs=[],
                      outputs=[chatbot, chat_input, state_messages, current_chat_id, history_dropdown])
    history_dropdown.select(fn=load_session, inputs=[history_dropdown],
                            outputs=[chatbot, state_messages, current_chat_id])
    delete_btn.click(fn=ui_delete_chat, inputs=[history_dropdown, current_chat_id],
                     outputs=[chatbot, chat_input, state_messages, history_dropdown, current_chat_id])
    refresh_btn.click(fn=lambda: gr.update(choices=get_all_sessions()), inputs=[], outputs=[history_dropdown])
    refresh_kb_btn.click(fn=refresh_kb_choices, inputs=[], outputs=[kb_dropdown])
    clear_kb_btn.click(fn=lambda: gr.update(value=None), inputs=[], outputs=[kb_dropdown])
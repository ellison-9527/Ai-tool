# app.py
import gradio as gr

from ui.chat_tab import create_chat_tab
from ui.mcp_tab import create_mcp_tab
from ui.rag_tab import create_rag_tab
# 【新增】引入刚才写好的 Skill 页面
from ui.skill_tab import create_skill_tab

def create_app():
    with gr.Blocks(title="AssistantPro") as demo:
        gr.Markdown("### 🤖 AssistantPro")
        gr.Markdown("AI 私人助手 · 多模型 · RAG 知识库 · MCP 工具 · 自定义 Skill · 多智能体")

        with gr.Tabs():
            with gr.Tab("💬 私人助手"):
                create_chat_tab()

            with gr.Tab("📚 RAG 管理"):
                create_rag_tab()

            with gr.Tab("🔌 MCP 配置"):
                create_mcp_tab()

            with gr.Tab("🧩 Skill 配置"):
                # 【修改】替换掉之前的 Markdown 提示，挂载页面
                create_skill_tab()

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=8080, theme=gr.themes.Soft())
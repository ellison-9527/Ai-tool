# app.py
import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.chat_tab import create_chat_tab
from ui.mcp_tab import create_mcp_tab
from ui.rag_tab import create_rag_tab
# 【新增】引入刚才写好的 Skill 页面
from ui.skill_tab import create_skill_tab

# 获取项目根目录下的 workspace 绝对路径
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace"))
def create_app():
    with gr.Blocks(title="AI 私人助手") as demo:
        gr.Markdown("### AI 私人助手")
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        theme=gr.themes.Soft(),
        allowed_paths=[workspace_path]  # [核心修改]：允许 Gradio 访问 workspace 文件夹
    )
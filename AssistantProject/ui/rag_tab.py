# ui/rag_tab.py
import gradio as gr
from AssistantProject.core.rag_manager import (
    process_and_store_documents,
    retrieve_documents,
    delete_knowledge_base,
    get_kb_list
)
# 【新增】引入我们刚写的评估模块
from AssistantProject.core.rag_eval import run_rag_evaluation


def create_rag_tab():
    gr.Markdown("### 📚 RAG 知识库管理与质量评测")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 1. 创建 / 追加知识库")

            kb_name_input = gr.Textbox(
                label="知识库名称",
                placeholder="例如: ai_docs_01 (仅限英文/数字/下划线)",
                info="如果输入的名字已存在，文件将追加进去；如果不存在，将创建新库。"
            )

            uploaded_files = gr.File(
                label="支持上传 PDF / TXT 文档",
                file_count="multiple",
                type="filepath"
            )

            with gr.Accordion("⚙️ 解析与检索策略设置", open=False):
                chunk_size = gr.Number(label="分块大小 (Chunk Size)", value=500)
                chunk_overlap = gr.Number(label="分块重叠 (Overlap)", value=50)
                gr.Markdown("---")
                retrieval_strategy = gr.Dropdown(
                    choices=["基础向量检索 (当前)", "混合检索 + BGE Rerank", "自适应 RAG"],
                    label="检索模式",
                    value="混合检索 + BGE Rerank",
                    info="（开发中）选择不同的检索及重排算法模型"
                )

            process_btn = gr.Button("🚀 解析并注入知识库", variant="primary")
            status_box = gr.Textbox(label="操作状态", lines=3, interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("#### 2. 知识库管理")

            with gr.Row():
                kb_dropdown = gr.Dropdown(
                    choices=get_kb_list(),
                    label="选择当前工作的知识库",
                    interactive=True
                )
                refresh_kb_btn = gr.Button("🔄 刷新列表", size="sm")
                delete_kb_btn = gr.Button("🗑️ 删除该库", size="sm", variant="stop")

            gr.Markdown("---")

            gr.Markdown("#### 3. 🔍 检索与评测中心")
            test_query = gr.Textbox(label="输入测试问题", placeholder="例如：本文档的核心观点是什么？")

            with gr.Row():
                test_btn = gr.Button("1️⃣ 仅获取底层检索片段")
                # 【新增】自动化评估按钮
                eval_btn = gr.Button("2️⃣ 🚀 运行完整 RAG 自动化评估", variant="primary")

            test_result = gr.Textbox(label="底层召回结果 (Top 3 Chunks)", lines=6, interactive=False)

            # 【新增】精美的 Markdown 报告输出框
            eval_result = gr.Markdown(label="评估报告")

    # --- 事件绑定 ---

    refresh_kb_btn.click(
        fn=lambda: gr.update(choices=get_kb_list()),
        inputs=[],
        outputs=[kb_dropdown]
    )

    process_btn.click(
        fn=process_and_store_documents,
        inputs=[uploaded_files, kb_name_input, chunk_size, chunk_overlap],
        outputs=[status_box, kb_dropdown]
    )

    delete_kb_btn.click(
        fn=delete_knowledge_base,
        inputs=[kb_dropdown],
        outputs=[status_box, kb_dropdown]
    )

    test_btn.click(
        fn=retrieve_documents,
        inputs=[kb_dropdown, test_query, retrieval_strategy],
        outputs=[test_result]
    )

    # 【新增】绑定评估事件
    eval_btn.click(
        fn=lambda: "⏳ 正在拉取资料、生成回答并召唤 AI 裁判打分，请耐心等待约 10-20 秒...",
        outputs=[eval_result]
    ).then(
        fn=run_rag_evaluation,
        inputs=[kb_dropdown, test_query],
        outputs=[eval_result]
    )
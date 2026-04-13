# ui/rag_tab.py
import gradio as gr
from AssistantProject.core.rag_manager import (
    process_and_store_documents,
    retrieve_documents,
    delete_knowledge_base,
    get_kb_list
)


def create_rag_tab():
    gr.Markdown("### 📚 RAG 知识库管理")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 1. 创建 / 追加知识库")

            # ✅ 新增：让用户给知识库起名字
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

            with gr.Accordion("高级解析设置", open=False):
                chunk_size = gr.Number(label="分块大小 (Chunk Size)", value=500)
                chunk_overlap = gr.Number(label="分块重叠 (Overlap)", value=50)

            process_btn = gr.Button("🚀 解析并注入知识库", variant="primary")
            status_box = gr.Textbox(label="操作状态", lines=3, interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("#### 2. 知识库管理")

            with gr.Row():
                # ✅ 新增：可选择的知识库下拉菜单
                kb_dropdown = gr.Dropdown(
                    choices=get_kb_list(),
                    label="选择当前工作的知识库",
                    interactive=True
                )
                refresh_kb_btn = gr.Button("🔄 刷新列表", size="sm")
                # ✅ 新增：红色的删除按钮
                delete_kb_btn = gr.Button("🗑️ 删除该库", size="sm", variant="stop")

            gr.Markdown("---")

            gr.Markdown("#### 3. 🔍 检索测试")
            test_query = gr.Textbox(label="输入测试问题", placeholder="例如：本文档的核心观点是什么？")
            test_btn = gr.Button("检索关联片段")
            test_result = gr.Textbox(label="检索召回结果 (Top 3 Chunks)", lines=12, interactive=False)

    # --- 事件绑定 ---

    # 刷新下拉列表
    refresh_kb_btn.click(
        fn=lambda: gr.update(choices=get_kb_list()),
        inputs=[],
        outputs=[kb_dropdown]
    )

    # 解析并注入文件（将知识库名字和文件一起传给后端）
    process_btn.click(
        fn=process_and_store_documents,
        inputs=[uploaded_files, kb_name_input, chunk_size, chunk_overlap],
        outputs=[status_box, kb_dropdown]  # 成功后会自动更新并选中右侧的下拉框
    )

    # 删除选中的知识库
    delete_kb_btn.click(
        fn=delete_knowledge_base,
        inputs=[kb_dropdown],
        outputs=[status_box, kb_dropdown]
    )

    # 检索测试（需要知道去哪个知识库里搜）
    test_btn.click(
        fn=retrieve_documents,
        inputs=[kb_dropdown, test_query],
        outputs=[test_result]
    )
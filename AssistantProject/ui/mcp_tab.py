# ui/mcp_tab.py
import os
import json
import asyncio
import gradio as gr

from AssistantProject.core.mcp_manager import get_dynamic_mcp_tools

CONFIG_DIR = "data"
MCP_CONFIG_PATH = os.path.join(CONFIG_DIR, "mcp_server.json")


def ensure_config_exists():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(MCP_CONFIG_PATH):
        with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


def get_mcp_dataframe():
    ensure_config_exists()
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            servers = json.load(f)
        except:
            servers = {}

    data = []
    for name, config in servers.items():
        transport = config.get("transport", "stdio")
        status = "connected" if config.get("enable", False) else "disconnect"
        if transport == "stdio":
            details = f"{config.get('command', '')} {' '.join(config.get('args', []))}"
        else:
            details = config.get("url", "N/A")
        data.append([name, transport, details, status])
    return data


def add_mcp_server(name, transport, command, args):
    if not name.strip() or not command.strip():
        return gr.update(), "⚠️ 名称和命令不能为空"

    ensure_config_exists()
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            servers = json.load(f)
        except:
            servers = {}

    new_config = {"transport": transport, "enable": False}
    if transport == "stdio":
        new_config["command"] = command.strip()
        new_config["args"] = [a for a in args.split(" ") if a.strip()]
    else:
        new_config["url"] = command.strip()

    servers[name.strip()] = new_config
    with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(servers, f, ensure_ascii=False, indent=2)

    return gr.update(value=get_mcp_dataframe()), f"✅ 成功添加: {name}"


def delete_mcp_server(name):
    if not name:
        return gr.update(), "⚠️ 请先在列表中点击选择一个服务！"

    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        servers = json.load(f)

    if name in servers:
        del servers[name]
        with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(servers, f, ensure_ascii=False, indent=2)
        return gr.update(value=get_mcp_dataframe()), f"🗑️ 已删除服务: {name}"
    return gr.update(), "⚠️ 找不到该服务配置"


async def get_tools_markdown():
    md_text = "**内置工具:**\n"
    md_text += "* `fetch_url`: 抓取指定 URL 网页内容\n"
    md_text += "* `tavily_search`: 搜索引擎工具\n"
    md_text += "* `bash`: 执行本地终端命令\n\n"

    md_text += "**🔌 动态 MCP 工具:**\n"
    try:
        mcp_schemas, _ = await get_dynamic_mcp_tools()
        if mcp_schemas:
            for schema in mcp_schemas:
                func = schema.get("function", {})
                name = func.get("name", "Unknown")
                desc = func.get("description", "暂无描述信息")
                short_desc = desc[:80] + "..." if len(desc) > 80 else desc
                md_text += f"* `{name}`: *{short_desc}*\n"
        else:
            md_text += "* *(当前没有已启用的 MCP 工具)*\n"
    except Exception as e:
        md_text += f"* ⚠️ 拉取外部工具失败: {str(e)}\n"

    return md_text


async def update_mcp_enable(name, is_enable):
    if not name:
        yield gr.update(), "⚠️ 请先在列表中点击选择一个服务！", gr.update()
        return

    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        servers = json.load(f)

    if name in servers:
        servers[name]["enable"] = is_enable
        with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(servers, f, ensure_ascii=False, indent=2)

        action = "连接" if is_enable else "断开"

        yield (
            gr.update(value=get_mcp_dataframe()),
            f"⏳ 正在{action} {name}，并向服务请求工具列表...",
            gr.update()
        )

        tools_md = await get_tools_markdown()

        yield (
            gr.update(),
            f"✅ 成功{action}服务: {name}",
            gr.update(value=tools_md)
        )
    else:
        yield gr.update(), "⚠️ 找不到该服务配置", gr.update()


# ==========================================
# 核心修复区：消除 Lambda 带来的所有错误
# ==========================================

# 1. 修复表格点击事件：精准获取选中行的第一列（名称）
def on_row_select(df, evt: gr.SelectData):
    # evt.index 的格式是 [行号, 列号]
    row_idx = evt.index[0]
    return df[row_idx][0]


# 2. 修复异步连接/断开按钮的包装函数
async def connect_action(name):
    async for res in update_mcp_enable(name, True):
        yield res


async def disconnect_action(name):
    async for res in update_mcp_enable(name, False):
        yield res


def create_mcp_tab():
    gr.Markdown("### 🔌 MCP 服务管理")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 添加服务")
            mcp_name = gr.Textbox(label="服务名称")
            mcp_transport = gr.Radio(choices=["stdio", "sse", "streamable_http", "websocket"], label="传输协议",
                                     value="stdio")
            mcp_command = gr.Textbox(label="命令/URL")
            mcp_args = gr.Textbox(label="参数 (空格分隔)")

            with gr.Row():
                add_btn = gr.Button("添加", variant="primary")
                connect_btn = gr.Button("连接")
            with gr.Row():
                disconnect_btn = gr.Button("断开")
                delete_btn = gr.Button("删除", variant="stop")
            status_output = gr.Textbox(label="操作状态", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("#### 服务列表 (点击行进行选择)")
            server_list = gr.Dataframe(
                headers=["名称", "类型", "详情", "状态"],
                datatype=["str", "str", "str", "str"],
                type="array",  # 强制按二维数组处理数据，配合上方点击事件
                value=get_mcp_dataframe(),
                interactive=False
            )

            gr.Markdown("---")
            with gr.Row():
                gr.Markdown("#### 可用工具列表")
                refresh_tools_btn = gr.Button("🔄 手动刷新工具", size="sm")

            tools_display = gr.Markdown("*(点击连接或刷新按钮以获取动态工具列表...)*")

    # --- 交互绑定 ---
    # 替换了导致 0 arguments 报错的 lambda
    server_list.select(fn=on_row_select, inputs=[server_list], outputs=[mcp_name])

    add_btn.click(add_mcp_server, [mcp_name, mcp_transport, mcp_command, mcp_args], [server_list, status_output])
    delete_btn.click(delete_mcp_server, mcp_name, [server_list, status_output])

    # 替换了导致界面 Error 飘红的异步 lambda
    connect_btn.click(fn=connect_action, inputs=mcp_name, outputs=[server_list, status_output, tools_display])
    disconnect_btn.click(fn=disconnect_action, inputs=mcp_name, outputs=[server_list, status_output, tools_display])

    refresh_tools_btn.click(fn=get_tools_markdown, inputs=[], outputs=[tools_display])
# core/mcp_manager.py
import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.utils.function_calling import convert_to_openai_tool

# 指向你的配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mcp_server.json")


async def get_dynamic_mcp_tools():
    """
    读取 JSON 配置，连接启用的 MCP 服务，
    并返回大模型需要的 (身份牌列表, 真实执行函数映射)
    """
    if not os.path.exists(CONFIG_PATH):
        return [], {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            servers = json.load(f)
        except Exception:
            servers = {}

    # 1. 过滤出被用户点亮（enable=True）的服务
    active_servers = {}
    for name, config in servers.items():
        if config.get("enable", False):
            # 复制一份配置，剔除不需要的 enable 字段
            cfg_copy = config.copy()
            cfg_copy.pop("enable", None)
            active_servers[name] = cfg_copy

    if not active_servers:
        return [], {}

    try:
        print(f"【MCP】正在桥接启用的外部服务: {list(active_servers.keys())} ...")
        # 2. 核心：通过 LangChain 的适配器统一连接这些离散的服务
        mcp_client = MultiServerMCPClient(active_servers)
        lc_tools = await mcp_client.get_tools()
    except Exception as e:
        print(f"【MCP】连接失败，请检查服务配置或环境: {e}")
        return [], {}

    # 3. 把庞大的 LangChain 工具格式，一键转换为极简的 OpenAI 格式身份牌
    mcp_schemas = [convert_to_openai_tool(t) for t in lc_tools]

    # 4. 把这些工具的真实触发动作，打包成 Python 异步函数
    mcp_callables = {}
    for tool in lc_tools:
        # 用闭包锁定当前的 tool 变量
        def make_async_callable(t):
            async def wrapper(**kwargs):
                print(f"【⚡ 触发 MCP 动作】调用外部工具: {t.name} | 参数: {kwargs}")
                return await t.ainvoke(kwargs)

            return wrapper

        mcp_callables[tool.name] = make_async_callable(tool)

    return mcp_schemas, mcp_callables
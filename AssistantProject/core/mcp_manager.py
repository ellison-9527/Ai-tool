# core/mcp_manager.py
import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.utils.function_calling import convert_to_openai_tool

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mcp_server.json")


async def get_langchain_mcp_tools():
    """【新增核心】直接返回 LangChain 原生工具列表，供 Agent 大脑使用"""
    if not os.path.exists(CONFIG_PATH):
        return []

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            servers = json.load(f)
        except Exception:
            servers = {}

    active_servers = {}
    for name, config in servers.items():
        if config.get("enable", False):
            cfg_copy = config.copy()
            cfg_copy.pop("enable", None)
            active_servers[name] = cfg_copy

    if not active_servers:
        return []

    try:
        # 通过 LangChain 的适配器连接 MCP 服务
        mcp_client = MultiServerMCPClient(active_servers)
        return await mcp_client.get_tools()
    except Exception as e:
        print(f"【MCP引擎】连接外部服务失败: {e}")
        return []


async def get_dynamic_mcp_tools():
    """提供给 UI 界面显示的兼容函数"""
    lc_tools = await get_langchain_mcp_tools()
    if not lc_tools:
        return [], {}

    mcp_schemas = [convert_to_openai_tool(t) for t in lc_tools]
    mcp_callables = {}

    for tool in lc_tools:
        def make_async_callable(t):
            async def wrapper(**kwargs):
                return await t.ainvoke(kwargs)

            return wrapper

        mcp_callables[tool.name] = make_async_callable(tool)

    return mcp_schemas, mcp_callables
# core/mcp_manager.py
import os
import sys
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.utils.function_calling import convert_to_openai_tool
from AssistantProject.core.logger import logger

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mcp_server.json")
# 新增：可视化脚本绝对路径
VISUALIZATION_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_servers", "visualization_server.py")


async def get_langchain_mcp_tools():
    """【新增核心】直接返回 LangChain 原生工具列表，供 Agent 大脑使用"""
    active_servers = {}

    # 1. 读取并加载 mcp_server.json 中启用的外部服务 (如 12306, 天气等)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                servers = json.load(f)
                for name, config in servers.items():
                    if config.get("enable", False):
                        cfg_copy = config.copy()
                        cfg_copy.pop("enable", None)
                        active_servers[name] = cfg_copy
            except json.JSONDecodeError as e:
                logger.error(f"❌ MCP 配置文件 JSON 格式错误: {e}")
            except Exception as e:
                logger.error(f"⚠️ 读取 MCP 配置文件失败: {e}")

        # 2. 【核心新增】代码层动态注入我们刚刚写好的“可视化服务”
        if os.path.exists(VISUALIZATION_SCRIPT_PATH):
            active_servers["visual_server"] = {
                "command": sys.executable,  # 使用当前虚拟环境的 Python
                "args": [VISUALIZATION_SCRIPT_PATH],
                "transport": "stdio"  # <--- 【新增这行】显式声明通信协议
            }
            logger.info("📊 成功将 [可视化数据] MCP Server 加入挂载队列！")
    # 如果没有任何服务，直接返回空
    if not active_servers:
        return []

    try:
        # 通过 LangChain 的适配器统一连接所有启用的 MCP 服务
        mcp_client = MultiServerMCPClient(active_servers)
        return await mcp_client.get_tools()
    except Exception as e:
        logger.error(f"❌ 【MCP引擎】连接外部服务失败，请检查服务路径或依赖: {e}")
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
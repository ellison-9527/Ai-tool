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


def get_available_servers():
    """获取所有已启用的 MCP 服务名称"""
    active_servers = []
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                servers = json.load(f)
                for name, config in servers.items():
                    if config.get("enable", False):
                        active_servers.append(name)
        except Exception:
            pass
    if os.path.exists(VISUALIZATION_SCRIPT_PATH):
        active_servers.append("visual_server")
    return active_servers

async def get_mcp_tools_by_server():
    """按服务器分组返回 MCP 工具，供沙箱隔离鉴权使用"""
    active_servers = {}

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

        if os.path.exists(VISUALIZATION_SCRIPT_PATH):
            active_servers["visual_server"] = {
                "command": sys.executable,
                "args": [VISUALIZATION_SCRIPT_PATH],
                "transport": "stdio"
            }
            
    if not active_servers:
        return {}

    server_tools = {}
    for name, cfg in active_servers.items():
        try:
            mcp_client = MultiServerMCPClient({name: cfg})
            tools = await mcp_client.get_tools()
            server_tools[name] = tools
            logger.info(f"✅ 成功加载 MCP 环境: [{name}] (包含 {len(tools)} 个工具)")
        except Exception as e:
            logger.error(f"❌ 加载 MCP 环境 [{name}] 失败: {e}")
            
    return server_tools

async def get_langchain_mcp_tools():
    """向下兼容的全局获取接口"""
    server_tools = await get_mcp_tools_by_server()
    flat_tools = []
    for tools in server_tools.values():
        flat_tools.extend(tools)
    return flat_tools


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
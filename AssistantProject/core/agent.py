# core/agent.py (完全体代码)
import os
import requests
import asyncio
import aiosqlite  # 异步数据库驱动
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from AssistantProject.core.tools import (
    fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file, tavily_search
)
from AssistantProject.core.rag_manager import retrieve_documents
from AssistantProject.core.mcp_manager import get_langchain_mcp_tools

load_dotenv()

# ==========================================
# 🧠 全局持久化记忆单例管理
# ==========================================
_global_memory_instance = None
_memory_lock = asyncio.Lock()
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agent_memory.sqlite"))


async def get_memory_instance():
    global _global_memory_instance
    async with _memory_lock:
        if _global_memory_instance is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            conn = await aiosqlite.connect(DB_PATH)
            _global_memory_instance = AsyncSqliteSaver(conn)
            await _global_memory_instance.setup()
            print(f"📁 [记忆系统] 持久化大脑已就绪: {DB_PATH}")
    return _global_memory_instance


_cached_mcp_tools = None


async def get_mcp_tools_safely():
    global _cached_mcp_tools
    if _cached_mcp_tools is None:
        try:
            print("🔄 首次加载 MCP Server 工具中...")
            _cached_mcp_tools = await get_langchain_mcp_tools()
            print("✅ MCP Server 加载完成！")
        except Exception as e:
            print(f"⚠️ MCP 初始化失败: {e}")
            _cached_mcp_tools = []
    return _cached_mcp_tools


def get_asr_text(file_path: str) -> str:
    url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    headers = {"Authorization": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")}
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(url, headers=headers, data={"model": "glm-asr-2512", "stream": "false"},
                                 files={"file": f})
            return resp.json().get("text", "")
    except Exception as e:
        return ""


def get_llm(target_model, max_token, temperature):
    return ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-higress-any-key", model=target_model,
                      max_tokens=int(max_token), temperature=float(temperature), streaming=True)


async def simple_agent_chat(user_message_content, sys_prompt, max_token, temperature, target_model,
                            thread_id="default_chat", kb_name=None):
    llm = get_llm(target_model, max_token, temperature)
    tools = [fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file]
    if tavily_search: tools.append(tavily_search)

    mcp_tools = await get_mcp_tools_safely()
    tools.extend(mcp_tools)

    if kb_name:
        def search_local_kb(query: str) -> str:
            return retrieve_documents(kb_name, query)

        rag_tool = StructuredTool.from_function(func=search_local_kb, name="search_knowledge_base",
                                                description=f"查阅知识库【{kb_name}】。")
        tools.append(rag_tool)
        sys_prompt = (sys_prompt or "") + f"\n\n注意：当前知识库【{kb_name}】已激活，查阅请调用 search_knowledge_base 工具。"

    environment_constraints = "\n\n=== 🟢 【系统法则】 ===\n你有完整的本地文件读写和命令执行权限！\n==========================================="
    final_sys_prompt = (sys_prompt or "") + environment_constraints

    memory = await get_memory_instance()
    config = {"configurable": {"thread_id": thread_id}}

    existing_state = await memory.aget(config)
    history_count = len(existing_state.get("channel_values", {}).get("messages", [])) if existing_state else 0
    print(f"🧠 [记忆系统] 会话 {thread_id} 已加载，包含历史消息数: {history_count}")

    try:
        agent = create_react_agent(llm, tools=tools, checkpointer=memory, state_modifier=final_sys_prompt)
    except TypeError:
        try:
            agent = create_react_agent(llm, tools=tools, checkpointer=memory, messages_modifier=final_sys_prompt)
        except TypeError:
            agent = create_react_agent(llm, tools=tools, checkpointer=memory)

    lc_messages = [HumanMessage(content=user_message_content)]

    try:
        async for event, metadata in agent.astream({"messages": lc_messages}, config=config, stream_mode="messages"):
            node = metadata.get("langgraph_node", "")
            if node == "agent":
                if hasattr(event, "tool_calls") and event.tool_calls:
                    for tc in event.tool_calls:
                        yield {"type": "tool_call", "name": tc["name"], "args": str(tc.get("args", ""))}
                elif hasattr(event, "tool_call_chunks") and event.tool_call_chunks:
                    for tc in event.tool_call_chunks:
                        if tc.get("name"): yield {"type": "tool_call", "name": tc["name"], "args": "..."}
                elif event.content:
                    yield {"type": "token", "content": event.content}
            elif node == "tools":
                if hasattr(event, "content") and event.content:
                    tool_content = event.content
                    # 🌟【核心修复】：拆解 MCP 返回的 JSON 列表，提取纯文本
                    if isinstance(tool_content, list):
                        texts = [item.get("text", "") for item in tool_content if item.get("type") == "text"]
                        tool_content = "\n".join(texts)
                    yield {"type": "tools_result", "name": getattr(event, "name", "tool"),
                           "content": str(tool_content)[:1500]}

        print(f"✅ [记忆系统] 会话 {thread_id} 当前轮次保存成功。")
    except Exception as e:
        yield {"type": "token", "content": f"\n\n*(⚠️ 执行发生异常: {e})*"}
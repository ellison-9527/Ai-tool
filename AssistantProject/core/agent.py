# core/agent.py
import os
import requests
import asyncio
import aiosqlite
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI

from AssistantProject.core.tools import (
    fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file, tavily_search, run_background_program
)
from AssistantProject.core.rag_manager import retrieve_documents
from AssistantProject.core.mcp_manager import get_langchain_mcp_tools
from AssistantProject.core.logger import logger

load_dotenv(override=True)

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
            logger.info(f"📁 [记忆系统] 持久化大脑已就绪: {DB_PATH}")
    return _global_memory_instance


_cached_mcp_tools = None

async def get_mcp_tools_safely():
    global _cached_mcp_tools
    if _cached_mcp_tools is None:
        try:
            logger.info("🔄 首次加载 MCP Server 工具中...")
            _cached_mcp_tools = await get_langchain_mcp_tools()
            logger.info("✅ MCP Server 加载完成！")
        except Exception as e:
            logger.error(f"⚠️ MCP 初始化失败: {e}")
            _cached_mcp_tools = []
    return _cached_mcp_tools


def get_asr_text(file_path: str) -> str:
    url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    # ASR 强制使用专属的智谱 Key
    api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": api_key}
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(url, headers=headers, data={"model": "glm-asr-2512", "stream": "false"}, files={"file": f})
            return resp.json().get("text", "")
    except Exception as e:
        logger.error(f"❌ ASR 语音识别失败: {e}")
        return ""


def get_llm(target_model, max_token, temperature):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "sk-higress-any-key")
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000/v1")
    return ChatOpenAI(
        base_url=base_url, 
        api_key=api_key, 
        model=target_model,
        max_tokens=int(max_token), 
        temperature=float(temperature), 
        streaming=True,
        timeout=60.0 # 增加 60 秒超时，防止网络波动导致无限期“思考中”
    )


def create_expert_tool(expert_name: str, expert_prompt: str, target_model: str, max_token: int, temperature: float):
    """
    【真实多智能体模式实现】：Agent as a Tool 模式
    将专家封装为独立工具，主 Agent (Router) 可以调用特定专家来获取结果。
    """
    @tool(f"ask_{expert_name.replace('-', '_')}_expert")
    def expert_node(query: str) -> str:
        f"向 {expert_name} 专家提问。用于需要专业分析、特定角色视角的任务。参数是给专家的问题描述。"
        logger.info(f"👥 [多智能体协作] 唤醒专家: {expert_name}...")
        llm = get_llm(target_model, max_token, temperature)
        try:
            res = llm.invoke([
                {"role": "system", "content": f"你是一个专业领域的专家。\n{expert_prompt}"},
                {"role": "user", "content": query}
            ])
            return res.content
        except Exception as e:
            return f"{expert_name} 专家回复失败: {e}"
            
    return expert_node


async def simple_agent_chat(user_message_content, sys_prompt, max_token, temperature, target_model,
                            thread_id="default_chat", kb_name=None, expert_prompts_map=None):
    llm = get_llm(target_model, max_token, temperature)
    
    # 基础工具组
    tools = [fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file, run_background_program]
    if tavily_search: 
        tools.append(tavily_search)

    # 动态挂载 MCP 工具
    mcp_tools = await get_mcp_tools_safely()
    tools.extend(mcp_tools)

    # 动态挂载知识库检索工具
    if kb_name:
        def search_local_kb(query: str) -> str:
            return retrieve_documents(kb_name, query)

        rag_tool = StructuredTool.from_function(
            func=search_local_kb, 
            name="search_knowledge_base",
            description=f"查阅知识库【{kb_name}】。"
        )
        tools.append(rag_tool)
        sys_prompt = (sys_prompt or "") + f"\n\n注意：当前知识库【{kb_name}】已激活，必须优先调用 search_knowledge_base 工具获取参考资料！"

    # =================================================================
    # 【重构点】: 多智能体 (Multi-Agent) 真正落地！引入真实的 DAG 编排
    # =================================================================
    if expert_prompts_map:
        from AssistantProject.core.multi_agent import build_multi_agent_graph
        # 把 tools 传进去，让专家不仅能说话，还能操作
        graph, graph_llm = build_multi_agent_graph(target_model, max_token, temperature, expert_prompts_map, tools)
        
        # 将用户的输入组装进入状态
        initial_state = {
            "messages": [HumanMessage(content=user_message_content)],
            "selected_expert": "",
            "expert_response": "",
            "router_reasoning": ""
        }
        
        try:
            logger.info("🚀 触发多智能体协作图流式执行...")
            # 使用 astream_events 捕获所有的 Token 流和 Tool 执行过程
            async for chunk in graph.astream_events(
                initial_state,
                version="v2",
                config={"configurable": {"thread_id": thread_id}}
            ):
                kind = chunk["event"]
                
                if kind == "on_chat_model_stream":
                    if chunk["data"]["chunk"].content:
                        yield {"type": "token", "content": chunk["data"]["chunk"].content}
                        
                elif kind == "on_tool_start":
                    yield {"type": "tool_call", "name": chunk["name"]}
                    
                elif kind == "on_tool_end":
                    output = chunk["data"].get("output")
                    # 工具的输出可能包含太多不需要前台展示的内容，只通知完成了即可，详情留给大模型总结
                    # 或者如果有需要可以打印简略的输出
                    if output:
                        yield {"type": "tools_result", "content": str(output)[:1000] + "..." if len(str(output)) > 1000 else str(output)}

            return # 多智能体模式在这里直接返回，不再执行后续的单 Agent 逻辑
        except Exception as e:
            logger.error(f"❌ 多智能体图执行失败: {e}")
            yield {"type": "token", "content": f"\n\n*(⚠️ 多智能体图执行异常: {e})*"}
            return

    environment_constraints = "\n\n=== 🟢 【系统法则】 ===\n你有完整的本地文件读写和命令执行权限！\n==========================================="
    final_sys_prompt = (sys_prompt or "") + environment_constraints

    memory = await get_memory_instance()
    config = {"configurable": {"thread_id": thread_id}}

    existing_state = await memory.aget(config)
    history_count = len(existing_state.get("channel_values", {}).get("messages", [])) if existing_state else 0
    logger.info(f"🧠 [记忆系统] 会话 {thread_id} 已加载，包含历史消息数: {history_count}")

    # 兼容不同版本的 LangGraph API
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
                        if tc.get("name"): 
                            yield {"type": "tool_call", "name": tc["name"], "args": "..."}
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

        logger.info(f"✅ [记忆系统] 会话 {thread_id} 当前轮次保存成功。")
    except Exception as e:
        logger.error(f"❌ 代理执行异常: {e}")
        yield {"type": "token", "content": f"\n\n*(⚠️ 执行发生异常: {e})*"}
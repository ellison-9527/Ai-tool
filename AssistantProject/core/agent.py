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
    fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file, tavily_search, run_background_program, duckduckgo_search
)
from AssistantProject.core.rag_manager import retrieve_documents
from AssistantProject.core.mcp_manager import get_langchain_mcp_tools, get_mcp_tools_by_server
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


_cached_mcp_tools_by_server = None

async def get_mcp_tools_by_server_safely():
    global _cached_mcp_tools_by_server
    if _cached_mcp_tools_by_server is None:
        try:
            logger.info("🔄 首次加载按服务隔离的 MCP Server 工具...")
            _cached_mcp_tools_by_server = await get_mcp_tools_by_server()
            logger.info(f"✅ MCP Server 沙箱加载完成！发现 {len(_cached_mcp_tools_by_server)} 个环境。")
        except Exception as e:
            logger.error(f"⚠️ MCP 初始化失败: {e}")
            _cached_mcp_tools_by_server = {}
    return _cached_mcp_tools_by_server


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


async def prune_memory_if_needed(graph_or_agent, config: dict, max_messages: int = 20):
    """【滑动窗口记忆修剪】防止长时间对话导致 Token 溢出"""
    try:
        current_state = await graph_or_agent.aget_state(config)
        messages = current_state.values.get("messages", [])
        if len(messages) > max_messages:
            from langchain_core.messages import RemoveMessage
            delete_count = len(messages) - max_messages
            # 删除最老的 delete_count 条消息
            delete_actions = [RemoveMessage(id=msg.id) for msg in messages[:delete_count] if hasattr(msg, 'id') and msg.id]
            if delete_actions:
                logger.warning(f"✂️ [记忆修剪] 会话历史过长 ({len(messages)}条)，已物理删除最老的 {len(delete_actions)} 条记忆。")
                await graph_or_agent.aupdate_state(config, {"messages": delete_actions})
    except Exception as e:
        logger.error(f"记忆修剪失败: {e}")


async def simple_agent_chat(user_message_content: list, sys_prompt: str, max_token: int,
                            temperature: float, target_model: str, thread_id: str,
                            kb_name: str = None, expert_prompts_map: dict = None,
                            agent_mode: str = "🤖 单体全能模式", custom_team_id: str = None,
                            allowed_mcp_servers: list = None):
    llm = get_llm(target_model, max_token, temperature)
    
    # 基础工具组
    base_tools = [duckduckgo_search, fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file, run_background_program]
    if tavily_search: 
        base_tools.append(tavily_search)

    # 动态挂载知识库检索工具
    if kb_name:
        def search_local_kb(query: str) -> str:
            return retrieve_documents(kb_name, query, target_model=target_model)

        rag_tool = StructuredTool.from_function(
            func=search_local_kb, 
            name="search_local_kb",
            description="当需要根据提供的本地知识库或私有文档来回答问题时，使用此工具进行语义检索。输入应为简练的搜索关键词。"
        )
        base_tools.append(rag_tool)

    tools = list(base_tools)
    # 动态沙箱挂载 MCP 工具 (供单体及普通团队使用)
    mcp_tools_map = await get_mcp_tools_by_server_safely()
    if allowed_mcp_servers is not None:
        for srv in allowed_mcp_servers:
            if srv in mcp_tools_map:
                tools.extend(mcp_tools_map[srv])
    else:
        for ts in mcp_tools_map.values():
            tools.extend(ts)

    # 动态挂载知识库检索工具
    if kb_name:
        def search_local_kb(query: str) -> str:
            return retrieve_documents(kb_name, query, target_model=target_model)

        rag_tool = StructuredTool.from_function(
            func=search_local_kb, 
            name="search_knowledge_base",
            description=f"查阅知识库【{kb_name}】。"
        )
        tools.append(rag_tool)
        sys_prompt = (sys_prompt or "") + f"\n\n注意：当前知识库【{kb_name}】已激活，必须优先调用 search_knowledge_base 工具获取参考资料！获取资料后，请务必提取最核心的信息进行简明扼要的回答，不要大段复制粘贴原文内容。"

    environment_constraints = "\n\n=== 🟢 【系统法则】 ===\n你有完整的本地文件读写和命令执行权限！\n==========================================="
    final_sys_prompt = (sys_prompt or "") + environment_constraints

    memory = await get_memory_instance()
    config = {"configurable": {"thread_id": thread_id}}

    existing_state = await memory.aget(config)
    history_count = len(existing_state.get("channel_values", {}).get("messages", [])) if existing_state else 0
    logger.info(f"🧠 [记忆系统] 会话 {thread_id} 已加载，包含历史消息数: {history_count}")

    # =================================================================
    # 【重构点】: 多智能体 (Multi-Agent) 真正落地！引入真实的 DAG 编排
    # =================================================================
    if agent_mode != "🤖 单体全能模式":
        from AssistantProject.core.multi_agent import build_multi_agent_graph, build_pipeline_team_graph, build_debate_team_graph
        
        if agent_mode == "👥 动态路由团队 (Router-Worker)":
            if not expert_prompts_map:
                yield {"type": "token", "content": "⚠️ 请先在右侧选择参与协作的专家团队 (Skill)，否则无法启动动态路由！"}
                return
            graph, graph_llm = build_multi_agent_graph(target_model, max_token, temperature, expert_prompts_map, tools, memory=memory)
            initial_state = {
                "messages": [HumanMessage(content=user_message_content)],
                "selected_expert": "",
                "expert_response": "",
                "router_reasoning": ""
            }
        elif agent_mode == "🏭 流水线审查团队 (Writer-Reviewer)":
            graph, graph_llm = build_pipeline_team_graph(target_model, max_token, temperature, tools, memory=memory)
            initial_state = {
                "messages": [HumanMessage(content=user_message_content)],
                "draft": "",
                "review": ""
            }
        elif agent_mode == "⚖️ 深度辩论团队 (Proposer-Critic)":
            graph, graph_llm = build_debate_team_graph(target_model, max_token, temperature, tools, memory=memory)
            initial_state = {
                "messages": [HumanMessage(content=user_message_content)],
                "proposer_arg": "",
                "critic_arg": "",
                "round_count": 0
            }
        elif agent_mode == "⚙️ 自定义流水线 (动态编排)":
            from AssistantProject.core.team_manager import get_team_config
            from AssistantProject.core.multi_agent import build_dynamic_pipeline_team_graph
            
            if not custom_team_id:
                yield {"type": "token", "content": "⚠️ 请先在右侧选择你要运行的编排团队！"}
                return
                
            team_cfg = get_team_config(custom_team_id)
            if not team_cfg:
                yield {"type": "token", "content": f"⚠️ 找不到团队 [{custom_team_id}] 的配置文件！"}
                return
                
            graph, graph_llm = build_dynamic_pipeline_team_graph(
                target_model=target_model,
                max_token=max_token,
                temperature=temperature,
                tools=base_tools,  # 仅传入基础工具，MCP 工具由各节点自行动态挂载
                team_config=team_cfg,
                memory=memory
            )
            initial_state = {
                "messages": [HumanMessage(content=user_message_content)]
            }
        else:
            # 安全兜底：如果没有匹配的模式，跳出这个 if，继续执行下方单体 Agent
            pass
        
        # 只要 graph 被赋值了，就开始执行 DAG
        if 'graph' in locals():
            try:
                # 在图流式执行前，先修剪超长历史记忆
                await prune_memory_if_needed(graph, config, max_messages=20)
                
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
                        if output:
                            yield {"type": "tools_result", "content": str(output)[:1000] + "..." if len(str(output)) > 1000 else str(output)}

                return # 多智能体模式在这里直接返回，不再执行后续的单 Agent 逻辑
            except Exception as e:
                logger.error(f"❌ 多智能体图执行失败: {e}")
                yield {"type": "token", "content": f"\n\n*(⚠️ 多智能体图执行异常: {e})*"}
                return


    # 兼容不同版本的 LangGraph API
    try:
        agent = create_react_agent(llm, tools=tools, checkpointer=memory, state_modifier=final_sys_prompt)
    except TypeError:
        try:
            agent = create_react_agent(llm, tools=tools, checkpointer=memory, messages_modifier=final_sys_prompt)
        except TypeError:
            agent = create_react_agent(llm, tools=tools, checkpointer=memory)

    # 在执行前，修剪超长历史记忆
    await prune_memory_if_needed(agent, config, max_messages=20)

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
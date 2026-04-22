from typing import TypedDict, Annotated, Sequence, Dict, Any, List
import operator
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from AssistantProject.core.agent import get_llm
from AssistantProject.core.logger import logger

# ==========================================
# 状态定义
# ==========================================
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    selected_expert: str        # 路由决定的下个执行专家
    expert_response: str        # 专家执行结果
    router_reasoning: str       # 路由分析原因

# ==========================================
# 节点：总调度路由 (Router)
# ==========================================
async def router_node(state: MultiAgentState, llm, expert_prompts_map: Dict[str, str], config: RunnableConfig):
    logger.info("🧭 [多智能体] 启动总路由 (Router) 分析任务分配...")
    
    expert_names = list(expert_prompts_map.keys())
    if not expert_names:
        return {"selected_expert": "FINISH", "router_reasoning": "无可用专家"}

    system_prompt = f"""你是一个高级任务路由分配器。
目前系统中有以下专家可用：{expert_names}。
请分析用户的最新消息，决定是否需要指派给某个专家。
如果问题很简单，你可以自己回答，请输出 "FINISH"。
如果需要专家介入，请明确输出该专家的名字。
输出格式要求：请严格只输出一个词（专家的名字，或者 FINISH），不要有任何其他多余字符。"""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    
    try:
        response = await llm.ainvoke(messages, config=config)
        decision = response.content.strip()
        logger.info(f"🧭 [多智能体] 路由决定: {decision}")
        
        # 兜底校验：如果 AI 胡言乱语没输出准确专家名，直接退出
        if decision not in expert_names and decision != "FINISH":
            # 尝试做个模糊匹配
            matched = "FINISH"
            for name in expert_names:
                if name in decision:
                    matched = name
                    break
            decision = matched
            logger.info(f"🧭 [多智能体] 模糊匹配纠正为: {decision}")
            
        return {"selected_expert": decision, "router_reasoning": "由大模型自动分发"}
    except Exception as e:
        logger.error(f"❌ 路由分析失败: {e}")
        return {"selected_expert": "FINISH"}

# ==========================================
# 边：条件路由分支
# ==========================================
def route_to_expert(state: MultiAgentState):
    next_node = state.get("selected_expert", "FINISH")
    if next_node == "FINISH":
        return "synthesizer" # 如果不需要专家，直接跳到合成器兜底回复
    return next_node

# ==========================================
# 节点生成器：专家节点 (Expert)
# ==========================================
def create_expert_node(expert_name: str, expert_prompt: str, llm, tools: list):
    """
    【架构演进：从单体模型到真正的子 Agent】
    将每一个专家包装成具备独立工具使用权限的 React Agent 子图。
    这样专家不仅能回答问题，还能根据需要执行命令、查知识库等。
    """
    # 编译专属这个专家的子 React 智能体 (兼容不同版本的 LangGraph API)
    agent_prompt = f"你是【{expert_name}】专家。\n{expert_prompt}\n请基于你的专业知识，通过调用工具收集信息，最终深度分析并回答用户的问题。"
    try:
        expert_agent = create_react_agent(model=llm, tools=tools, state_modifier=agent_prompt)
        inject_sys_msg = False
    except TypeError:
        try:
            expert_agent = create_react_agent(model=llm, tools=tools, messages_modifier=agent_prompt)
            inject_sys_msg = False
        except TypeError:
            expert_agent = create_react_agent(model=llm, tools=tools)
            inject_sys_msg = True

    async def expert_logic(state: MultiAgentState, config: RunnableConfig):
        logger.info(f"👥 [多智能体] 专家子智能体 {expert_name} 接管任务并开始思考...")
        
        try:
            # 根据是否需要注入系统提示词来构造输入
            input_messages = [SystemMessage(content=agent_prompt)] + list(state["messages"]) if inject_sys_msg else state["messages"]
            local_config = {**config, "recursion_limit": 5} if config else {"recursion_limit": 5}
            result = await expert_agent.ainvoke({"messages": input_messages}, config=local_config)
            final_msg = result["messages"][-1].content
            logger.info(f"👥 [多智能体] 专家 {expert_name} 处理完毕。")
            return {"expert_response": f"【来自核心专家 {expert_name} 的深度执行与分析】\n\n{final_msg}"}
        except GraphRecursionError:
            logger.warning(f"专家 {expert_name} 触发死循环拦截。")
            return {"expert_response": f"【来自核心专家 {expert_name} 的深度执行与分析】\n\n⚠️ [系统物理级拦截] 智能体尝试连续调用外部工具超过允许次数，涉嫌陷入死循环，已被系统强制阻断。"}
        except Exception as e:
            logger.error(f"❌ 专家 {expert_name} 执行异常: {e}")
            return {"expert_response": f"专家 {expert_name} 处理失败: {e}"}
            
    return expert_logic

# ==========================================
# 节点：总成器 (Synthesizer)
# ==========================================
async def synthesizer_node(state: MultiAgentState, llm, config: RunnableConfig):
    logger.info("📝 [多智能体] 启动合成器 (Synthesizer) 组装最终结果...")
    
    expert_response = state.get("expert_response", "")
    if not expert_response:
        # 说明路由器觉得没必要用专家，直接让主模型直接回答
        sys_msg = SystemMessage(content="你是一个全能助手。请直接回答用户的问题。")
        messages = [sys_msg] + list(state["messages"])
    else:
        # 有专家的结论，需要整合润色
        sys_msg = SystemMessage(content="你是一个高级汇编员。你的下属专家刚才给出了一份专业的分析报告。请你基于专家的报告，以极其专业、友好且结构化的语气向用户传达结论。")
        messages = list(state["messages"]) + [SystemMessage(content=f"这是专家的原始返回结果，请以此为基准进行润色解答：\n{expert_response}")]
        
    try:
        # 这里真实调用大模型，config 会将 token 源源不断推送到前端
        response = await llm.ainvoke(messages, config=config)
        return {"messages": [response]} 
    except Exception as e:
        logger.error(f"❌ 合成失败: {e}")
        return {"messages": []}

# ==========================================
# 构造图 (Build DAG)
# ==========================================
def build_multi_agent_graph(target_model: str, max_token: int, temperature: float, expert_prompts_map: Dict[str, str], tools: list, memory=None):
    """构建真正的多智能体协作网络图"""
    logger.info("🏗️ 正在构建 LangGraph 真实多智能体 DAG 网络...")
    
    llm = get_llm(target_model, max_token, temperature)
    
    # 初始化状态图
    workflow = StateGraph(MultiAgentState)
    
    # 使用真正的 async wrapper 替代 lambda，接收并透传 config
    async def run_router(state: MultiAgentState, config: RunnableConfig):
        return await router_node(state, llm, expert_prompts_map, config)
        
    async def run_synthesizer(state: MultiAgentState, config: RunnableConfig):
        return await synthesizer_node(state, llm, config)
    
    # 1. 挂载路由节点
    workflow.add_node("router", run_router)
    
    # 2. 挂载合成器节点
    workflow.add_node("synthesizer", run_synthesizer)
    
    # 3. 动态挂载所有专家节点 (每一个节点都是一个 Sub-Agent)
    for name, prompt in expert_prompts_map.items():
        workflow.add_node(name, create_expert_node(name, prompt, llm, tools))
        # 所有专家执行完毕后，统一流向合成器
        workflow.add_edge(name, "synthesizer")
        
    # 4. 路由逻辑连接 (Edges)
    # 起点 -> 路由器
    workflow.add_edge(START, "router")
    
    # 路由器 -> (条件分支) -> 各路专家 / 直接合成
    workflow.add_conditional_edges(
        "router",
        route_to_expert,
        {name: name for name in expert_prompts_map.keys()} | {"synthesizer": "synthesizer"}
    )
    
    # 合成器 -> 终点
    workflow.add_edge("synthesizer", END)
    
    # 编译执行图
    logger.info("✅ 多智能体网络 DAG 编译完成！")
    return workflow.compile(checkpointer=memory) if memory else workflow.compile(), llm

# ==========================================
# 扩展模板 1：流水线审查团队 (Writer -> Reviewer -> Editor)
# ==========================================
class PipelineState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    draft: str
    review: str

async def writer_node(state: PipelineState, llm, tools, config: RunnableConfig):
    logger.info("🏭 [流水线] Writer 正在起草...")
    prompt = "你是一个专业撰稿人(Writer)。请根据用户需求，可以且应该使用工具进行深度研究，并给出详尽的初稿。\n⚠️ 警告：如果使用搜索工具 2 次都没有找到满意结果，或者用户是在询问关于你们的聊天记录和上下文，请**绝对不要**再调用搜索工具！直接利用你的记忆和现有常识作答，严禁无限次重试！\n\n【重要排版要求】：请务必在你的回答最开头加上 `### 📝 Writer 初稿：\n`"
    try:
        writer_agent = create_react_agent(model=llm, tools=tools, state_modifier=prompt)
        msgs = state["messages"]
    except TypeError:
        try:
            writer_agent = create_react_agent(model=llm, tools=tools, messages_modifier=prompt)
            msgs = state["messages"]
        except TypeError:
            writer_agent = create_react_agent(model=llm, tools=tools)
            msgs = [SystemMessage(content=prompt)] + list(state["messages"])
    try:
        local_config = {**config, "recursion_limit": 5} if config else {"recursion_limit": 5}
        res = await writer_agent.ainvoke({"messages": msgs}, config=local_config)
        content = res["messages"][-1].content
    except GraphRecursionError:
        logger.warning("Writer 触发死循环拦截。")
        content = "⚠️ [系统物理级拦截] Writer 尝试调用外部工具次数超限，已被强制阻断。"
    except Exception as e:
        content = f"起草失败: {e}"
    return {"draft": content}

async def reviewer_node(state: PipelineState, llm, tools, config: RunnableConfig):
    logger.info("🏭 [流水线] Reviewer 正在审查...")
    prompt = f"你是一个严苛的审查员(Reviewer)。请审查以下初稿：\n\n{state.get('draft', '')}\n\n请指出逻辑漏洞、事实错误或语言问题，并提出明确的修改建议。你可以使用工具核实事实。\n⚠️ 警告：最多只允许尝试调用工具 2 次！如果用户询问的是聊天历史记录，绝对禁止调用搜索工具。不要无限重试！\n\n【重要排版要求】：请务必在你的回答最开头加上 `\n\n---\n### 🧐 Reviewer 审查意见：\n`"
    try:
        reviewer_agent = create_react_agent(model=llm, tools=tools, state_modifier=prompt)
        msgs = state["messages"]
    except TypeError:
        try:
            reviewer_agent = create_react_agent(model=llm, tools=tools, messages_modifier=prompt)
            msgs = state["messages"]
        except TypeError:
            reviewer_agent = create_react_agent(model=llm, tools=tools)
            msgs = [SystemMessage(content=prompt)] + list(state["messages"])
    try:
        local_config = {**config, "recursion_limit": 5} if config else {"recursion_limit": 5}
        res = await reviewer_agent.ainvoke({"messages": msgs}, config=local_config)
        content = res["messages"][-1].content
    except GraphRecursionError:
        logger.warning("Reviewer 触发死循环拦截。")
        content = "⚠️ [系统物理级拦截] Reviewer 尝试调用外部工具次数超限，已被强制阻断。"
    except Exception as e:
        content = f"审查失败: {e}"
    return {"review": content}

async def editor_node(state: PipelineState, llm, config: RunnableConfig):
    logger.info("🏭 [流水线] Editor 正在定稿...")
    sys_msg = SystemMessage(content="你是一个终审主编(Editor)。请结合用户的原始需求、Writer的初稿以及Reviewer的审查意见，撰写最终的完美回答。直接输出最终结果，无需解释过程。\n\n【重要排版要求】：请务必在你的回答最开头加上 `\n\n---\n### 👑 Editor 最终定稿：\n`")
    last_human_msg = next((m.content for m in reversed(state['messages']) if isinstance(m, HumanMessage)), state['messages'][-1].content)
    user_msg = HumanMessage(content=f"【原始需求】: {last_human_msg}\n\n【初稿】: {state.get('draft')}\n\n【审查意见】: {state.get('review')}")
    res = await llm.ainvoke([sys_msg, user_msg], config=config)
    return {"messages": [res]}

def build_pipeline_team_graph(target_model: str, max_token: int, temperature: float, tools: list, memory=None):
    logger.info("🏗️ 构建流水线审查团队 (Writer -> Reviewer -> Editor)")
    llm = get_llm(target_model, max_token, temperature)
    workflow = StateGraph(PipelineState)
    
    async def run_w(state, config): return await writer_node(state, llm, tools, config)
    async def run_r(state, config): return await reviewer_node(state, llm, tools, config)
    async def run_e(state, config): return await editor_node(state, llm, config)
    
    workflow.add_node("writer", run_w)
    workflow.add_node("reviewer", run_r)
    workflow.add_node("editor", run_e)
    
    workflow.add_edge(START, "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", "editor")
    workflow.add_edge("editor", END)
    
    return workflow.compile(checkpointer=memory) if memory else workflow.compile(), llm


# ==========================================
# 扩展模板 2：深度辩论团队 (Proposer <-> Critic -> Judge)
# ==========================================
class DebateState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    proposer_arg: str
    critic_arg: str
    round_count: int

async def proposer_node(state: DebateState, llm, tools, config: RunnableConfig):
    logger.info("⚖️ [辩论] Proposer 立论...")
    prompt = "你是一个辩论的正方(Proposer)。请提出极具说服力的论点或方案。如果有反方的反驳意见，请针对性地进行辩护和完善方案。可以使用工具辅助论证。\n⚠️ 警告：最多只允许尝试调用工具 2 次！如果用户询问的是聊天历史记录，绝对禁止调用搜索工具。不要无限次重试！直接基于你的逻辑进行辩护！\n\n【重要排版要求】：请务必在你的回答最开头加上 `\n\n---\n### 🟢 正方 (Proposer) 观点：\n`"
    if state.get("critic_arg"):
        prompt += f"\n\n反方上一轮的反驳意见为: {state['critic_arg']}"
    try:
        agent = create_react_agent(model=llm, tools=tools, state_modifier=prompt)
        msgs = state["messages"]
    except TypeError:
        try:
            agent = create_react_agent(model=llm, tools=tools, messages_modifier=prompt)
            msgs = state["messages"]
        except TypeError:
            agent = create_react_agent(model=llm, tools=tools)
            msgs = [SystemMessage(content=prompt)] + list(state["messages"])
    try:
        local_config = {**config, "recursion_limit": 5} if config else {"recursion_limit": 5}
        res = await agent.ainvoke({"messages": msgs}, config=local_config)
        content = res["messages"][-1].content
    except GraphRecursionError:
        logger.warning("Proposer 触发死循环拦截。")
        content = "⚠️ [系统物理级拦截] Proposer 尝试调用外部工具次数超限，已被强制阻断。"
    except Exception as e:
        content = f"立论失败: {e}"
    return {"proposer_arg": content}

async def critic_node(state: DebateState, llm, tools, config: RunnableConfig):
    logger.info("⚖️ [辩论] Critic 反驳...")
    prompt = f"你是一个尖锐的反方(Critic)。请针对正方的论点进行猛烈抨击，指出缺陷、风险或不可行之处。可以使用工具找反例。\n⚠️ 警告：最多只允许尝试调用工具 2 次！如果用户询问的是聊天历史记录，绝对禁止调用搜索工具。严禁陷入死循环调用！直接基于你的逻辑进行反驳！\n\n【重要排版要求】：请务必在你的回答最开头加上 `\n\n---\n### 🔴 反方 (Critic) 驳斥：\n`\n\n正方当前论点为: {state.get('proposer_arg', '')}"
    try:
        agent = create_react_agent(model=llm, tools=tools, state_modifier=prompt)
        msgs = state["messages"]
    except TypeError:
        try:
            agent = create_react_agent(model=llm, tools=tools, messages_modifier=prompt)
            msgs = state["messages"]
        except TypeError:
            agent = create_react_agent(model=llm, tools=tools)
            msgs = [SystemMessage(content=prompt)] + list(state["messages"])
    try:
        local_config = {**config, "recursion_limit": 5} if config else {"recursion_limit": 5}
        res = await agent.ainvoke({"messages": msgs}, config=local_config)
        content = res["messages"][-1].content
    except GraphRecursionError:
        logger.warning("Critic 触发死循环拦截。")
        content = "⚠️ [系统物理级拦截] Critic 尝试调用外部工具次数超限，已被强制阻断。"
    except Exception as e:
        content = f"反驳失败: {e}"
    return {"critic_arg": content, "round_count": state.get("round_count", 0) + 1}

def debate_router(state: DebateState):
    if state.get("round_count", 0) >= 1: # 跑 1 个回合的反驳就结束，避免耗时太长
        return "judge"
    return "proposer"

async def judge_node(state: DebateState, llm, config: RunnableConfig):
    logger.info("⚖️ [辩论] Judge 裁决...")
    sys_msg = SystemMessage(content="你是一个中立的法官(Judge)。请基于用户的原始问题、正方的最终方案和反方的核心质疑，给出一个最客观、最中肯、最完善的最终结论。直接输出结论。\n\n【重要排版要求】：请务必在你的回答最开头加上 `\n\n---\n### ⚖️ 法官 (Judge) 最终裁决：\n`")
    last_human_msg = next((m.content for m in reversed(state['messages']) if isinstance(m, HumanMessage)), state['messages'][-1].content)
    user_msg = HumanMessage(content=f"【原始需求】: {last_human_msg}\n\n【正方方案】: {state.get('proposer_arg')}\n\n【反方质疑】: {state.get('critic_arg')}")
    res = await llm.ainvoke([sys_msg, user_msg], config=config)
    return {"messages": [res]}

def build_debate_team_graph(target_model: str, max_token: int, temperature: float, tools: list, memory=None):
    logger.info("🏗️ 构建深度辩论团队 (Proposer <-> Critic -> Judge)")
    llm = get_llm(target_model, max_token, temperature)
    workflow = StateGraph(DebateState)
    
    async def run_p(state, config): return await proposer_node(state, llm, tools, config)
    async def run_c(state, config): return await critic_node(state, llm, tools, config)
    async def run_j(state, config): return await judge_node(state, llm, config)
    
    workflow.add_node("proposer", run_p)
    workflow.add_node("critic", run_c)
    workflow.add_node("judge", run_j)
    
    workflow.add_edge(START, "proposer")
    workflow.add_edge("proposer", "critic")
    workflow.add_conditional_edges("critic", debate_router, {"proposer": "proposer", "judge": "judge"})
    workflow.add_edge("judge", END)
    
    return workflow.compile(checkpointer=memory) if memory else workflow.compile(), llm

# ==========================================
# 扩展模板 3：动态编排流水线团队 (Team Orchestrator)
# ==========================================
class DynamicPipelineState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_dynamic_pipeline_team_graph(target_model: str, max_token: int, temperature: float, tools: list, team_config: dict, memory=None):
    from langchain_core.messages import AIMessage
    logger.info(f"🏗️ 构建动态编排流水线团队: {team_config.get('team_name', 'Unnamed')}")
    llm = get_llm(target_model, max_token, temperature)
    
    workflow = StateGraph(DynamicPipelineState)
    nodes_info = team_config.get('nodes', [])
    if not nodes_info:
        raise ValueError("Team has no nodes configured")
        
    def create_dynamic_node(name, prompt, allowed_mcps):
        async def node_func(state: DynamicPipelineState, config: RunnableConfig):
            logger.info(f"🏭 [动态流水线] {name} 节点接管任务...")
            
            # --- 构建节点专属工具箱 ---
            node_tools = list(tools)
            if allowed_mcps:
                from AssistantProject.core.agent import get_mcp_tools_by_server_safely
                mcp_tools_map = await get_mcp_tools_by_server_safely()
                for srv in allowed_mcps:
                    if srv in mcp_tools_map:
                        node_tools.extend(mcp_tools_map[srv])
            
            # 追加工具限制防止死循环
            safe_prompt = prompt + "\n⚠️ 警告：最多只允许尝试调用工具 2 次！如果用户询问的是聊天历史记录，绝对禁止调用搜索工具。直接基于当前知识作答！"
            try:
                agent = create_react_agent(model=llm, tools=node_tools, state_modifier=safe_prompt)
                msgs = state["messages"]
            except TypeError:
                try:
                    agent = create_react_agent(model=llm, tools=node_tools, messages_modifier=safe_prompt)
                    msgs = state["messages"]
                except TypeError:
                    agent = create_react_agent(model=llm, tools=node_tools)
                    msgs = [SystemMessage(content=safe_prompt)] + list(state["messages"])
            try:
                local_config = {**config, "recursion_limit": 5} if config else {"recursion_limit": 5}
                res = await agent.ainvoke({"messages": msgs}, config=local_config)
                output_msg = res["messages"][-1].content
            except GraphRecursionError:
                logger.warning(f"动态节点 {name} 触发死循环拦截。")
                output_msg = "⚠️ [系统物理级拦截] 该节点尝试调用外部工具次数超限，已被强制阻断。"
            except Exception as e:
                output_msg = f"执行异常: {e}"
            formatted_output = f"\n\n---\n### 🏭 {name} 节点输出：\n{output_msg}"
            return {"messages": [AIMessage(content=formatted_output)]}
        return node_func

    node_names = []
    for node in nodes_info:
        node_name = node["name"]
        node_prompt = node["prompt"]
        allowed_mcps = node.get("allowed_mcps", [])
        workflow.add_node(node_name, create_dynamic_node(node_name, node_prompt, allowed_mcps))
        node_names.append(node_name)
        
    workflow.add_edge(START, node_names[0])
    for i in range(len(node_names)-1):
        workflow.add_edge(node_names[i], node_names[i+1])
    workflow.add_edge(node_names[-1], END)
    
    return workflow.compile(checkpointer=memory) if memory else workflow.compile(), llm

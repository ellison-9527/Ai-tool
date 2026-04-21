from typing import TypedDict, Annotated, Sequence, Dict, Any, List
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from AssistantProject.core.agent import get_llm
from AssistantProject.core.logger import logger

# ==========================================
# 状态定义
# ==========================================
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
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
    except TypeError:
        try:
            expert_agent = create_react_agent(model=llm, tools=tools, messages_modifier=agent_prompt)
        except TypeError:
            expert_agent = create_react_agent(model=llm, tools=tools)

    async def expert_logic(state: MultiAgentState, config: RunnableConfig):
        logger.info(f"👥 [多智能体] 专家子智能体 {expert_name} 接管任务并开始思考...")
        
        # 只传入原始对话给专家去发挥
        try:
            # 运行子图，必须透传 config 才能让前端 astream_events 捕获到工具调用和 token 流
            result = await expert_agent.ainvoke({"messages": state["messages"]}, config=config)
            final_msg = result["messages"][-1].content
            logger.info(f"👥 [多智能体] 专家 {expert_name} 处理完毕。")
            return {"expert_response": f"【来自核心专家 {expert_name} 的深度执行与分析】\n\n{final_msg}"}
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
def build_multi_agent_graph(target_model: str, max_token: int, temperature: float, expert_prompts_map: Dict[str, str], tools: list):
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
    return workflow.compile(), llm

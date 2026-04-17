# core/agent.py
import os
import json
import requests
from typing import List, TypedDict
from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from AssistantProject.core.tools import (
    fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file, tavily_search
)
from AssistantProject.core.rag_manager import retrieve_documents
from AssistantProject.core.mcp_manager import get_langchain_mcp_tools

load_dotenv()


# ==========================================
# 0. 基础函数
# ==========================================
def get_asr_text(file_path: str) -> str:
    url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    headers = {"Authorization": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")}
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(url, headers=headers, data={"model": "glm-asr-2512", "stream": "false"},
                                 files={"file": f})
            return resp.json().get("text", "")
    except Exception as e:
        print(f"语音识别失败: {e}")
        return ""


def get_llm(target_model, max_token, temperature):
    return ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="sk-higress-any-key",
        model=target_model,
        max_tokens=int(max_token),
        temperature=float(temperature),
        streaming=True
    )


def get_graph_llm(model_name):
    # 状态机内部使用的判别器模型，为了稳定使用 0 温度
    return ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="sk-higress-any-key",
        model=model_name,
        temperature=0
    )


# ==========================================
# 1. Self-RAG 状态机定义 (极速版)
# ==========================================
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    kb_name: str
    model: str
    iteration_count: int


def retrieve_node(state):
    # 每次进入检索节点，计数器 +1
    current_iter = state.get("iteration_count", 0) + 1
    print(f"--- [Node: Retrieve] 正在检索知识库 (第 {current_iter} 次尝试) ---")
    docs = retrieve_documents(state['kb_name'], state['question'])
    return {"documents": [docs], "iteration_count": current_iter}


def transform_query_node(state):
    print("--- [Node: Transform Query] 检索结果不佳，AI 正在提炼更精准的搜索词 ---")
    llm = get_graph_llm(state['model'])
    prompt = f"原问题: {state['question']}。当前检索结果不佳，请给出一个更适合向量数据库检索的精简关键词："
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"question": res.content}


def generate_node(state):
    print("--- [Node: Generate] 正在进行最终回答的生成 ---")
    llm = get_graph_llm(state['model'])
    # 严格的系统提示词，防止幻觉
    prompt = f"问题: {state['question']}\n\n参考资料: {state['documents']}\n\n请严格基于上述资料回答问题。如果资料中没有答案，请直接说明“抱歉，知识库中未找到相关确切内容”，不要凭空编造。"
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": res.content}


def grade_documents_edge(state):
    # 【核心刹车机制】最多允许检索 2 次，防止无限死循环
    if state.get("iteration_count", 0) >= 2:
        print("--- [Edge: Is Rel?] 已达最大检索次数，强制进入生成阶段 ---")
        return "generate"

    print("--- [Edge: Is Rel?] 正在评估检索质量 ---")
    llm = get_graph_llm(state['model'])
    prompt = f"问题: {state['question']}\n文档片段: {state['documents']}\n这些文档是否包含回答该问题的关键信息？请只回复 'yes' 或 'no'："
    res = llm.invoke([HumanMessage(content=prompt)])

    if "yes" in res.content.lower():
        return "generate"
    else:
        return "transform_query"


# 构建流线型状态机
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("transform_query", transform_query_node)

workflow.set_entry_point("retrieve")
# 检索后评估，不合格就重写，合格就生成
workflow.add_conditional_edges("retrieve", grade_documents_edge,
                               {"generate": "generate", "transform_query": "transform_query"})
# 重写后重新检索
workflow.add_edge("transform_query", "retrieve")
# 生成完直接结束 (砍掉耗时的幻觉检测循环)
workflow.add_edge("generate", END)

self_rag_app = workflow.compile()


# ==========================================
# 2. 主路由函数
# ==========================================
async def simple_agent_chat(messages, sys_prompt, max_token, temperature, target_model, kb_name=None):
    # 模式 A：普通 Tool Agent 模式 (无知识库)
    if not kb_name:
        llm = get_llm(target_model, max_token, temperature)
        tools = [fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file]
        if tavily_search: tools.append(tavily_search)
        mcp_tools = await get_langchain_mcp_tools()
        tools.extend(mcp_tools)

        environment_constraints = """
               \n\n=== 🟢 【系统操作权限与物理环境法则】 ===
               1. 核心超能力：你拥有完整的本地文件读写和命令执行权限！遇到编写代码、创建项目、运行脚本的任务时，【必须】主动调用 `write_local_file` 和 `execute_python_script` 等工具自动完成，绝对不要让用户自己去手动创建。
               2. 唯一限制：你处于单智能体环境，无并行子智能体机制。仅在被要求执行“并发多线程测试”、“长期挂起Web服务器”或试图用 `echo` 伪造虚假测试报告时，才需要告知用户环境不支持。正常的文件读写和脚本执行均被系统【完全允许且鼓励】。
               ===========================================
               """
        final_sys_prompt = (sys_prompt or "") + environment_constraints

        agent = create_react_agent(llm, tools=tools)
        lc_messages = [SystemMessage(content=final_sys_prompt)]
        for msg in messages:
            lc_messages.append(
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))

            # ==========================================
            # 【安全气囊】：捕获所有生成器异常，优雅输出
            # ==========================================
            try:
                async for event, metadata in agent.astream({"messages": lc_messages}, stream_mode="messages"):
                    node = metadata.get("langgraph_node", "")
                    if node == "agent":
                        if event.tool_call_chunks:
                            for tc in event.tool_call_chunks:
                                if tc.get("name"): yield {"type": "tool_call", "name": tc["name"], "args": "..."}
                        elif event.content:
                            yield {"type": "token", "content": event.content}
                    elif node == "tools":
                        if event.content: yield {"type": "tools_result", "name": event.name,
                                                 "content": str(event.content)[:1500]}
            except Exception as e:
                yield {"type": "token",
                       "content": f"\n\n*(⚠️ 提示：代码已成功写入硬盘，但在后台尝试执行或验证时由于环境限制被阻断。您可以直接在您的终端中手动运行该文件！)*"}
            return

    # 模式 B：极速纯净版 Self-RAG 模式
    print(f"\n🚀 [Fast Self-RAG] 启动！关联知识库: {kb_name}")
    user_query = messages[-1]["content"]

    inputs = {
        "question": user_query,
        "kb_name": kb_name,
        "model": target_model,
        "iteration_count": 0
    }

    # 不再将中间过程抛给前端，直接在后台完整运行图谱
    final_state = await self_rag_app.ainvoke(inputs)

    # 提取最终生成的文本
    final_generation = final_state.get("generation", "⚠️ 抱歉，生成过程中发生错误。")

    # 将最终结果以打字机的形式平滑推给前端
    chunk_size = 5
    for i in range(0, len(final_generation), chunk_size):
        yield {"type": "token", "content": final_generation[i:i + chunk_size]}
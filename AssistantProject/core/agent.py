# core/agent.py
import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from AssistantProject.core.tools import fetch_url, bash_execute, tavily_search
from AssistantProject.core.rag_manager import retrieve_documents
from AssistantProject.core.mcp_manager import get_langchain_mcp_tools

load_dotenv()

CURRENT_KB_NAME = None

def get_asr_text(file_path: str) -> str:
    url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
    headers = {"Authorization": os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")}
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(url, headers=headers, data={"model": "glm-asr-2512", "stream": "false"}, files={"file": f})
            return resp.json().get("text", "")
    except Exception as e:
        print(f"语音识别失败: {e}")
        return ""

@tool
def search_knowledge_base(query: str) -> str:
    """当用户询问特定领域或本地私有知识库内容时，必须调用此工具进行检索。"""
    global CURRENT_KB_NAME
    if not CURRENT_KB_NAME:
        return "⚠️ 当前未挂载任何知识库，无法检索。"
    return retrieve_documents(kb_name=CURRENT_KB_NAME, query_text=query)

def get_llm(target_model, max_token, temperature):
    return ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="sk-higress-any-key",
        model=target_model,
        max_tokens=int(max_token),
        temperature=float(temperature),
        streaming=True
    )

async def simple_agent_chat(messages, sys_prompt, max_token, temperature, target_model, kb_name=None):
    global CURRENT_KB_NAME
    CURRENT_KB_NAME = kb_name

    llm = get_llm(target_model, max_token, temperature)

    # 1. 装载基础工具箱
    tools = [fetch_url, bash_execute]
    if tavily_search:
        tools.append(tavily_search)
    if kb_name:
        tools.append(search_knowledge_base)

    # 2. 【核心打通】：动态拉取已开启的 MCP 工具并放入工具箱
    mcp_tools = await get_langchain_mcp_tools()
    tools.extend(mcp_tools)

    # 创建带工具的 Agent
    agent = create_react_agent(llm, tools=tools)

    lc_messages = []
    if sys_prompt:
        lc_messages.append(SystemMessage(content=sys_prompt))
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    async for event, metadata in agent.astream({"messages": lc_messages}, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")

        if node == "agent":
            if event.tool_call_chunks:
                for tc in event.tool_call_chunks:
                    if tc.get("name"):
                        yield {"type": "tool_call", "name": tc["name"], "args": "..."}
            elif event.content:
                yield {"type": "token", "content": event.content}

        elif node == "tools":
            if event.content:
                yield {"type": "tools_result", "name": event.name, "content": str(event.content)[:1500]}
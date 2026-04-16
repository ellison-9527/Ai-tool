# core/agent.py
import os
import requests
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 【新增引入写文件工具】
from AssistantProject.core.tools import (
    fetch_url, bash_execute, tavily_search,
    read_local_file, execute_python_script, write_local_file
)
from AssistantProject.core.rag_manager import retrieve_documents
from AssistantProject.core.mcp_manager import get_langchain_mcp_tools

load_dotenv()
CURRENT_KB_NAME = None
CURRENT_MODEL = "qwen-max"


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


def grade_and_rewrite(query: str, context: str) -> dict:
    if "📭" in context or "⚠️" in context: return {"score": 0, "rewrite_query": query, "reason": "未找到"}
    llm = get_llm(CURRENT_MODEL, max_token=512, temperature=0.1)
    prompt = f"评估RAG片段质量。问题:{query}\n片段:{context[:2000]}\n输出JSON: {{\"score\":0-100, \"reason\":\"...\", \"rewrite_query\":\"...\"}}"
    try:
        text = llm.invoke([SystemMessage(content=prompt)]).content.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"score": 100, "rewrite_query": ""}


from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """搜索本地私有知识库。"""
    global CURRENT_KB_NAME
    if not CURRENT_KB_NAME: return "⚠️ 未挂载知识库。"
    context = retrieve_documents(kb_name=CURRENT_KB_NAME, query_text=query)
    eval_res = grade_and_rewrite(query, context)
    if eval_res.get("score", 100) >= 60: return context
    new_query = eval_res.get("rewrite_query", query)
    return retrieve_documents(kb_name=CURRENT_KB_NAME, query_text=new_query)


async def simple_agent_chat(messages, sys_prompt, max_token, temperature, target_model, kb_name=None):
    global CURRENT_KB_NAME, CURRENT_MODEL
    CURRENT_KB_NAME = kb_name
    CURRENT_MODEL = target_model

    llm = get_llm(target_model, max_token, temperature)

    # 【赋予它毁天灭地的读写能力】
    tools = [fetch_url, bash_execute, read_local_file, execute_python_script, write_local_file]

    if tavily_search: tools.append(tavily_search)
    if kb_name: tools.append(search_knowledge_base)

    mcp_tools = await get_langchain_mcp_tools()
    tools.extend(mcp_tools)

    agent = create_react_agent(llm, tools=tools)

    lc_messages = [SystemMessage(content=sys_prompt)] if sys_prompt else []
    for msg in messages:
        lc_messages.append(
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))

    async for event, metadata in agent.astream({"messages": lc_messages}, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")
        if node == "agent":
            if event.tool_call_chunks:
                for tc in event.tool_call_chunks:
                    if tc.get("name"): yield {"type": "tool_call", "name": tc["name"], "args": "..."}
            elif event.content:
                yield {"type": "token", "content": event.content}
        elif node == "tools":
            if event.content: yield {"type": "tools_result", "name": event.name, "content": str(event.content)[:1500]}
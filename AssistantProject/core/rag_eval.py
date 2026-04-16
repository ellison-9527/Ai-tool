# core/rag_eval.py
import json
from AssistantProject.core.agent import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from AssistantProject.core.rag_manager import retrieve_documents


def run_rag_evaluation(kb_name, query, target_model="qwen-max"):
    """执行完整的 RAG 评测管线"""
    if not kb_name or not query:
        return "⚠️ 请提供知识库名称和测试问题。"

    # 1. 执行最高级别的检索
    context = retrieve_documents(kb_name, query, strategy="混合检索 + BGE Rerank", top_k=3)
    if "📭" in context or "⚠️" in context:
        return "⚠️ 检索失败或知识库为空，无法进行评估。"

    # 2. 让 AI 根据检索结果生成正式回答
    llm = get_llm(target_model, max_token=1024, temperature=0.3)
    gen_prompt = f"你是一个专业的知识库问答助手。请严格基于以下参考资料回答用户问题。如果资料中没有答案，请直接说“资料未提及”。\n\n【参考资料】:\n{context}\n\n【用户问题】: {query}"

    try:
        ans_res = llm.invoke([HumanMessage(content=gen_prompt)])
        ai_answer = ans_res.content
    except Exception as e:
        return f"⚠️ 生成回答失败: {e}"

    # 3. 召唤 AI 裁判 (LLM-as-a-Judge) 进行三维打分
    eval_sys = """你是一个严苛的 RAG (检索增强生成) 评估专家。
    你需要根据以下三个维度对 RAG 系统的表现进行 0-100 的打分：
    1. context_score (上下文相关性): 检索到的资料是否包含回答问题所需的核心信息？
    2. faithfulness_score (内容忠实度): AI 的回答是否完全基于检索到的资料，没有产生幻觉或编造信息？
    3. answer_score (回答相关性): AI 的回答是否直接、准确地解答了用户的问题？

    请严格按照以下 JSON 格式输出，不要输出任何额外的 Markdown 符号或废话：
    {
        "context_score": 90,
        "faithfulness_score": 85,
        "answer_score": 95,
        "reasoning": "简短的综合评价理由（指出优点和不足）"
    }"""

    eval_user = f"【用户问题】: {query}\n\n【底层检索到的参考资料】: {context}\n\n【前端最终生成的回答】: {ai_answer}"

    try:
        eval_res = llm.invoke([SystemMessage(content=eval_sys), HumanMessage(content=eval_user)])
        res_text = eval_res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(res_text)

        # 4. 组装精美的评测报告
        report = f"### 📊 RAG 自动化评估报告 (LLM-as-a-Judge)\n\n"
        report += f"**评测模型:** `{target_model}` | **测试知识库:** `{kb_name}`\n\n"
        report += f"#### 📈 核心指标打分\n"
        report += f"- 🎯 **上下文相关性 (Context Relevance):** `{data.get('context_score')}/100`\n"
        report += f"- 🛡️ **内容忠实度 (Faithfulness - 防幻觉):** `{data.get('faithfulness_score')}/100`\n"
        report += f"- 💡 **回答相关性 (Answer Relevance):** `{data.get('answer_score')}/100`\n\n"
        report += f"#### 📝 裁判点评\n> {data.get('reasoning')}\n\n"
        report += f"---\n#### 🤖 AI 最终生成的回答预览\n{ai_answer}"

        return report

    except Exception as e:
        return f"⚠️ 评估过程解析失败: {e}\n\n（可能由于模型未按 JSON 格式输出，您可以重试）"
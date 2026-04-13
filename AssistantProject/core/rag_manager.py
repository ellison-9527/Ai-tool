# core/rag_manager.py
import os
import uuid
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ 使用原生 pymilvus 直连你的独立服务器
from pymilvus import MilvusClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# ==========================================
# 核心改变：直连你本地运行的 Milvus 服务器端口
# ==========================================
MILVUS_URI = "http://127.0.0.1:19530"

try:
    print(f"尝试连接到 Milvus 服务器: {MILVUS_URI} ...")
    milvus_client = MilvusClient(uri=MILVUS_URI)
    print("✅ 成功连接到本地 Milvus 服务器！")
except Exception as e:
    print(f"❌ 无法连接到 Milvus 服务器，请检查 Docker 或服务是否开启: {e}")


def get_embedding(text):
    """调用智谱接口获取 1024 维向量"""
    try:
        resp = client.embeddings.create(model="embedding-2", input=text)
        return resp.data[0].embedding
    except Exception as e:
        print(f"向量化失败: {e}")
        return [0.0] * 1024


def get_kb_list():
    """获取当前所有知识库的名称列表"""
    try:
        return milvus_client.list_collections()
    except Exception:
        return []


def extract_text(file_path):
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
    return text.replace('\x00', '')


def process_and_store_documents(file_paths, kb_name, chunk_size, chunk_overlap):
    import gradio as gr
    if not file_paths:
        return "⚠️ 请先上传文件！", gr.update(choices=get_kb_list())
    if not kb_name:
        return "⚠️ 请输入知识库名称！", gr.update(choices=get_kb_list())

    # Milvus 的表名极其严格：只能用字母、数字、下划线
    safe_kb_name = re.sub(r'[^a-zA-Z0-9_]', '_', kb_name)

    # 如果在 Milvus 中没有这个知识库，就按照规范“建表”
    if not milvus_client.has_collection(collection_name=safe_kb_name):
        milvus_client.create_collection(
            collection_name=safe_kb_name,
            dimension=1024,  # 智谱 embedding-2 输出的维度是 1024
            auto_id=True,  # 让 Milvus 自动生成主键 ID
            metric_type="COSINE"  # 采用余弦相似度
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )

    total_chunks = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        raw_text = extract_text(file_path)
        if not raw_text.strip(): continue

        chunks = text_splitter.split_text(raw_text)
        if not chunks: continue

        # 将文本转化为 Milvus 要求的字典列表格式
        data_to_insert = []
        for chunk in chunks:
            vector = get_embedding(chunk)
            data_to_insert.append({
                "vector": vector,
                "text": chunk,
                "source": file_name
            })

        # 批量插入数据
        milvus_client.insert(collection_name=safe_kb_name, data=data_to_insert)
        total_chunks += len(chunks)

    return f"✅ 成功解析！知识库 [{safe_kb_name}] 新增 {total_chunks} 个片段。", gr.update(choices=get_kb_list(),
                                                                                        value=safe_kb_name)


def retrieve_documents(kb_name, query_text, top_k=3):
    """在 Milvus 知识库中进行余弦检索"""
    if not kb_name: return "⚠️ 请先在上方选择一个知识库！"
    if not query_text.strip(): return "⚠️ 请输入测试问题！"

    try:
        if not milvus_client.has_collection(collection_name=kb_name):
            return f"⚠️ 知识库 {kb_name} 不存在！"

        query_vector = get_embedding(query_text)

        search_res = milvus_client.search(
            collection_name=kb_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "source"],
            search_params={"metric_type": "COSINE"}
        )

        if not search_res or not search_res[0]:
            return "📭 知识库中未检索到相关内容。"

        res_str = ""
        for i, hit in enumerate(search_res[0]):
            entity = hit.get("entity", {})
            source = entity.get("source", "未知文件")
            text = entity.get("text", "")
            similarity = hit.get("distance", 0.0)

            res_str += f"📄 【片段 {i + 1}】 (来源: {source} | 相似度: {similarity:.4f})\n{text}\n"
            res_str += "-" * 40 + "\n\n"

        return res_str.strip()
    except Exception as e:
        return f"⚠️ 检索失败: {e}"


def delete_knowledge_base(kb_name):
    import gradio as gr
    if not kb_name:
        return "⚠️ 请先在下拉列表中选择要删除的知识库", gr.update()
    try:
        if milvus_client.has_collection(collection_name=kb_name):
            milvus_client.drop_collection(collection_name=kb_name)
        return f"🗑️ 已成功彻底删除知识库: {kb_name}", gr.update(choices=get_kb_list(), value=None)
    except Exception as e:
        return f"⚠️ 删除失败: {e}", gr.update()
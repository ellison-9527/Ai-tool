# core/rag_manager.py
import os
import re
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 【新增引入】语义切片与非结构化文档解析
# ==========================================
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

# ==========================================
# 1. 配置 Milvus 客户端与本地模型 (GPU 加速)
# ==========================================
MILVUS_URI = "http://127.0.0.1:19530"

try:
    print(f"尝试连接到 Milvus 服务器: {MILVUS_URI} ...")
    milvus_client = MilvusClient(uri=MILVUS_URI)
    print("✅ 成功连接到本地 Milvus 服务器！")
except Exception as e:
    print(f"❌ 无法连接到 Milvus 服务器，请检查 Docker 或服务是否开启: {e}")

MODEL_PATH = r"D:\桌面等\人工智能助手\AssistantProject\models\BAAI\bge-m3"
RERANK_MODEL_PATH = r"D:\桌面等\人工智能助手\AssistantProject\models\BAAI\bge-reranker-v2-m3"

try:
    print(f"尝试加载本地 BGE-M3 模型: {MODEL_PATH} ...")
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=MODEL_PATH,
        use_fp16=True,
        device="cuda:0"
    )
    print("✅ 本地 BGE-M3 模型加载成功 (GPU 加速开启)！")
except Exception as e:
    print(f"❌ 本地 BGE-M3 模型加载失败: {e}")

try:
    print(f"尝试加载本地 BGE-Reranker 模型: {RERANK_MODEL_PATH} ...")
    bge_reranker = BGERerankFunction(
        model_name=RERANK_MODEL_PATH,
        device="cuda:0"
    )
    print("✅ 本地 BGE-Reranker 模型加载成功 (GPU 加速开启)！")
except Exception as e:
    print(f"❌ 本地 BGE-Reranker 模型加载失败，请检查路径: {e}")


# ==========================================
# 【核心新增】让 LangChain 语义切片器白嫖我们本地的 BGE GPU 模型
# ==========================================
class LocalBGEEmbeddings(Embeddings):
    def __init__(self, bge_ef):
        self.bge_ef = bge_ef

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 只提取 Dense 密集向量用于语义突变计算
        embeddings = self.bge_ef.encode_documents(texts)["dense"]
        return [emb.tolist() if hasattr(emb, "tolist") else [float(v) for v in emb] for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        emb = self.bge_ef.encode_queries([text])["dense"][0]
        return emb.tolist() if hasattr(emb, "tolist") else [float(v) for v in emb]


# ==========================================
# 【核心新增】Markdown 层级结构保留引擎
# ==========================================
def merge_title_content(datas):
    merged_data = []
    parent_dict = {}
    for document in datas:
        metadata = document.metadata
        if 'languages' in metadata:
            metadata.pop('languages')

        parent_id = metadata.get('parent_id', None)
        category = metadata.get('category', None)
        element_id = metadata.get('element_id', None)

        if category == 'NarrativeText' and parent_id is None:
            merged_data.append(document)
        if category == 'Title':
            document.metadata['title'] = document.page_content
            if parent_id in parent_dict:
                document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                document.metadata['title'] = document.page_content
            parent_dict[element_id] = document
        if category != 'Title' and parent_id and parent_id in parent_dict:
            parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + '\n' + document.page_content
            parent_dict[parent_id].metadata['category'] = 'content'

    if parent_dict:
        merged_data.extend(parent_dict.values())
    return merged_data


# ==========================================
# 2. 结构化解析与分块逻辑 (进化为 SemanticChunker)
# ==========================================
def extract_and_split(file_path, chunk_size, chunk_overlap):
    chunks = []
    try:
        # 实例化基于本地 GPU 向量的语义切片器
        local_embeddings = LocalBGEEmbeddings(bge_m3_ef)
        # percentile 模式：计算所有句子间差异，切断差异最大的前 X%
        semantic_chunker = SemanticChunker(local_embeddings, breakpoint_threshold_type="percentile")

        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # PDF 也升级为语义切片，大幅保留学术上下文
            semantic_docs = semantic_chunker.split_documents(docs)
            chunks = [doc.page_content for doc in semantic_docs]

        elif file_path.lower().endswith('.md'):
            try:
                # 尝试使用高级非结构化引擎读取
                loader = UnstructuredMarkdownLoader(file_path=file_path, mode='elements', strategy='fast')
                raw_docs = list(loader.lazy_load())
                merged_docs = merge_title_content(raw_docs)

                # 针对每一个携带标题属性的块进行二次语义切分
                for d in merged_docs:
                    if len(d.page_content) > int(chunk_size):
                        split_result = semantic_chunker.split_documents([d])
                        for split_doc in split_result:
                            title_prefix = f"[{split_doc.metadata.get('title', '')}]\n" if split_doc.metadata.get(
                                'title') else ""
                            chunks.append(title_prefix + split_doc.page_content)
                    else:
                        title_prefix = f"[{d.metadata.get('title', '')}]\n" if d.metadata.get('title') else ""
                        chunks.append(title_prefix + d.page_content)
            except Exception as unstruct_e:
                print(f"⚠️ Unstructured 解析失败，降级为普通语义切块: {unstruct_e}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs = [Document(page_content=text)]
                chunks = [doc.page_content for doc in semantic_chunker.split_documents(docs)]

        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs = [Document(page_content=text)]
            chunks = [doc.page_content for doc in semantic_chunker.split_documents(docs)]

    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}")

    return [c.replace('\x00', '').strip() for c in chunks if c.strip()]


# ==========================================
# 3. 知识库管理与数据插入 (双向量融合)
# ==========================================
def get_kb_list():
    try:
        return milvus_client.list_collections()
    except Exception:
        return []


def process_and_store_documents(file_paths, kb_name, chunk_size, chunk_overlap):
    import gradio as gr
    if not file_paths:
        return "⚠️ 请先上传文件！", gr.update(choices=get_kb_list())
    if not kb_name:
        return "⚠️ 请输入知识库名称！", gr.update(choices=get_kb_list())

    if re.search(r'[^a-zA-Z0-9_]', kb_name):
        return "⚠️ 知识库名称仅支持【英文、数字、下划线】！Milvus底层不支持中文，请修改。", gr.update()

    safe_kb_name = kb_name

    if not milvus_client.has_collection(collection_name=safe_kb_name):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(
            collection_name=safe_kb_name,
            schema=schema,
            index_params=index_params
        )

    total_chunks = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        chunks = extract_and_split(file_path, chunk_size, chunk_overlap)
        if not chunks: continue

        embeddings = bge_m3_ef.encode_documents(chunks)
        dense_vecs = embeddings["dense"]
        sparse_vecs = embeddings["sparse"]

        safe_sparse_dicts = []
        if hasattr(sparse_vecs, "indptr"):
            indptr = sparse_vecs.indptr
            indices = sparse_vecs.indices
            data = sparse_vecs.data
            for i in range(sparse_vecs.shape[0]):
                start, end = indptr[i], indptr[i + 1]
                safe_sparse_dicts.append({int(indices[j]): float(data[j]) for j in range(start, end)})
        elif isinstance(sparse_vecs, list):
            safe_sparse_dicts = sparse_vecs

        data_to_insert = []
        for i in range(len(chunks)):
            safe_dense = dense_vecs[i].tolist() if hasattr(dense_vecs[i], "tolist") else [float(v) for v in
                                                                                          dense_vecs[i]]
            data_to_insert.append({
                "dense_vector": safe_dense,
                "sparse_vector": safe_sparse_dicts[i],
                "text": chunks[i],
                "source": file_name
            })

        milvus_client.insert(collection_name=safe_kb_name, data=data_to_insert)
        total_chunks += len(chunks)

    return f"✅ 成功！知识库 [{safe_kb_name}] 注入 {total_chunks} 个片段 (语义级切片 | GPU加速)。", gr.update(
        choices=get_kb_list(), value=safe_kb_name)


# ==========================================
# 4. 动态路由检索（基础 Dense / 混合重排 / 自适应）
# ==========================================
def retrieve_documents(kb_name, query_text, strategy="混合检索 + BGE Rerank", top_k=3):
    if not kb_name: return "⚠️ 请先在上方选择一个知识库！"
    if not query_text.strip(): return "⚠️ 请输入测试问题！"

    try:
        if not milvus_client.has_collection(collection_name=kb_name):
            return f"⚠️ 知识库 {kb_name} 不存在！"

        query_embeddings = bge_m3_ef.encode_queries([query_text])

        raw_dense_query = query_embeddings["dense"][0]
        safe_dense_query = raw_dense_query.tolist() if hasattr(raw_dense_query, "tolist") else [float(v) for v in
                                                                                                raw_dense_query]

        if "基础向量检索" in strategy:
            search_res = milvus_client.search(
                collection_name=kb_name,
                data=[safe_dense_query],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE"},
                limit=top_k,
                output_fields=["text", "source"]
            )

            if not search_res or not search_res[0]:
                return "📭 知识库中未检索到相关内容。"

            res_str = "💡 [当前模式：基础向量检索 (仅基于语意)]\n\n"
            for i, hit in enumerate(search_res[0]):
                entity = hit.get("entity", {})
                source = entity.get("source", "未知文件")
                text = entity.get("text", "")
                distance = hit.get("distance", 0.0)

                res_str += f"📄 【片段 {i + 1}】 (来源: {source} | 纯语义相似度: {distance:.4f})\n{text}\n"
                res_str += "-" * 40 + "\n\n"

            return res_str.strip()

        else:
            raw_sparse_query = query_embeddings["sparse"]
            if hasattr(raw_sparse_query, "indptr"):
                indptr = raw_sparse_query.indptr
                indices = raw_sparse_query.indices
                data = raw_sparse_query.data
                start, end = indptr[0], indptr[1]
                safe_sparse_query = {int(indices[j]): float(data[j]) for j in range(start, end)}
            else:
                safe_sparse_query = raw_sparse_query[0] if isinstance(raw_sparse_query, list) else raw_sparse_query

            candidate_limit = top_k * 5

            dense_req = AnnSearchRequest(
                data=[safe_dense_query],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                limit=candidate_limit
            )

            sparse_req = AnnSearchRequest(
                data=[safe_sparse_query],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=candidate_limit
            )

            search_res = milvus_client.hybrid_search(
                collection_name=kb_name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(k=60),
                limit=candidate_limit,
                output_fields=["text", "source"]
            )

            if not search_res or not search_res[0]:
                return "📭 知识库中未检索到相关内容。"

            candidates = []
            for hit in search_res[0]:
                entity = hit.get("entity", {})
                candidates.append({
                    "text": entity.get("text", ""),
                    "source": entity.get("source", "未知文件")
                })

            texts_to_rerank = [c["text"] for c in candidates]
            rerank_results = bge_reranker(query=query_text, documents=texts_to_rerank)

            final_results = []
            for res in rerank_results:
                original_doc = candidates[res.index]
                original_doc["rerank_score"] = res.score
                final_results.append(original_doc)

            final_top_k = final_results[:top_k]

            prefix_msg = "💡 [当前模式：混合检索 + BGE Rerank 重排]\n\n"
            if "自适应" in strategy:
                prefix_msg = "🚧 [自适应 RAG 路由仍在开发中... 自动降级为：混合重排模式]\n\n"

            res_str = prefix_msg
            for i, hit in enumerate(final_top_k):
                source = hit["source"]
                text = hit["text"]
                score = hit["rerank_score"]

                res_str += f"📄 【片段 {i + 1}】 (来源: {source} | 重排精准度打分: {score:.4f})\n{text}\n"
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
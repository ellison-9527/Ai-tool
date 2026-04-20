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
from markitdown import MarkItDown  # [新增] 微软文档解析神器

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


class LocalBGEEmbeddings(Embeddings):
    def __init__(self, bge_ef):
        self.bge_ef = bge_ef

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.bge_ef.encode_documents(texts)["dense"]
        return [emb.tolist() if hasattr(emb, "tolist") else [float(v) for v in emb] for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        emb = self.bge_ef.encode_queries([text])["dense"][0]
        return emb.tolist() if hasattr(emb, "tolist") else [float(v) for v in emb]


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
# 2. 结构化解析与分块逻辑 (进化为 SemanticChunker + MarkItDown)
# ==========================================
def extract_and_split(file_path, chunk_size, chunk_overlap):
    chunks = []
    try:
        local_embeddings = LocalBGEEmbeddings(bge_m3_ef)
        semantic_chunker = SemanticChunker(local_embeddings, breakpoint_threshold_type="percentile")
        file_ext = file_path.lower()

        # 【修复点】：拦截所有复杂二进制/富文本格式，交给 markitdown 处理
        if file_ext.endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
            print(f"📄 正在使用 MarkItDown 解析复杂文档: {file_path}")
            md_converter = MarkItDown()
            result = md_converter.convert(file_path)
            # 转化为 Markdown 后，按语义切片
            docs = [Document(page_content=result.text_content)]
            semantic_docs = semantic_chunker.split_documents(docs)
            chunks = [doc.page_content for doc in semantic_docs]

        elif file_ext.endswith('.md'):
            try:
                loader = UnstructuredMarkdownLoader(file_path=file_path, mode='elements', strategy='fast')
                raw_docs = list(loader.lazy_load())
                merged_docs = merge_title_content(raw_docs)

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
            # 纯文本格式 (txt 等)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            docs = [Document(page_content=text)]
            chunks = [doc.page_content for doc in semantic_chunker.split_documents(docs)]

    except Exception as e:
        print(f"解析文件 {file_path} 失败: {e}")

    return [c.replace('\x00', '').strip() for c in chunks if c.strip()]


# ==========================================
# 3. 知识库管理与数据插入 (双向量融合) - 保持不变
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

        # 修复点：如果由于各种原因没有提取出文本块，安全跳过此文件
        if not chunks:
            print(f"⚠️ 文件 {file_name} 解析为空，已跳过。")
            continue

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

        # 修复点：确保只有在有数据的情况下，才执行插入操作，并将它放进循环内
        if data_to_insert:
            milvus_client.insert(collection_name=safe_kb_name, data=data_to_insert)
            total_chunks += len(chunks)

    # 循环结束后，统一进行落盘和加载
    if total_chunks > 0:
        try:
            print(f"💾 正在将知识库 [{safe_kb_name}] 强制落盘并加载到内存...")
            milvus_client.flush(collection_name=safe_kb_name)
            milvus_client.load_collection(collection_name=safe_kb_name)
            print("✅ 知识库就绪！")
        except Exception as e:
            print(f"⚠️ 知识库落盘失败: {e}")
            return f"⚠️ 数据已插入，但知识库就绪失败: {e}", gr.update()
    else:
        return "⚠️ 未能从文件中提取到有效内容，知识库构建失败。", gr.update()

    return f"✅ 成功！知识库 [{safe_kb_name}] 注入 {total_chunks} 个片段 (语义级切片 | GPU加速)。", gr.update(
        choices=get_kb_list(), value=safe_kb_name)


def retrieve_documents(kb_name, query_text, strategy="混合检索 + BGE Rerank", top_k=3):
    # 保持原有检索逻辑不变
    if not kb_name: return "⚠️ 请先在上方选择一个知识库！"
    if not query_text.strip(): return "⚠️ 请输入测试问题！"
    try:
        if not milvus_client.has_collection(collection_name=kb_name):
            return f"⚠️ 知识库 {kb_name} 不存在！"
            # 👇👇👇【核心修复】：在检索前，强制将此知识库加载到内存中！
            milvus_client.load_collection(collection_name=kb_name)
        query_embeddings = bge_m3_ef.encode_queries([query_text])
        raw_dense_query = query_embeddings["dense"][0]
        safe_dense_query = raw_dense_query.tolist() if hasattr(raw_dense_query, "tolist") else [float(v) for v in
                                                                                                raw_dense_query]

        if "基础向量检索" in strategy:
            search_res = milvus_client.search(collection_name=kb_name, data=[safe_dense_query],
                                              anns_field="dense_vector", search_params={"metric_type": "COSINE"},
                                              limit=top_k, output_fields=["text", "source"])
            if not search_res or not search_res[0]: return "📭 知识库中未检索到相关内容。"
            res_str = "💡 [当前模式：基础向量检索]\n\n"
            for i, hit in enumerate(search_res[0]):
                res_str += f"📄 【片段 {i + 1}】 (来源: {hit['entity']['source']})\n{hit['entity']['text']}\n" + "-" * 40 + "\n\n"
            return res_str.strip()
        else:
            raw_sparse_query = query_embeddings["sparse"]
            if hasattr(raw_sparse_query, "indptr"):
                safe_sparse_query = {int(raw_sparse_query.indices[j]): float(raw_sparse_query.data[j]) for j in
                                     range(raw_sparse_query.indptr[0], raw_sparse_query.indptr[1])}
            else:
                safe_sparse_query = raw_sparse_query[0] if isinstance(raw_sparse_query, list) else raw_sparse_query

            candidate_limit = top_k * 5
            dense_req = AnnSearchRequest(data=[safe_dense_query], anns_field="dense_vector",
                                         param={"metric_type": "COSINE"}, limit=candidate_limit)
            sparse_req = AnnSearchRequest(data=[safe_sparse_query], anns_field="sparse_vector",
                                          param={"metric_type": "IP"}, limit=candidate_limit)

            search_res = milvus_client.hybrid_search(collection_name=kb_name, reqs=[dense_req, sparse_req],
                                                     ranker=RRFRanker(k=60), limit=candidate_limit,
                                                     output_fields=["text", "source"])
            if not search_res or not search_res[0]: return "📭 知识库中未检索到相关内容。"

            candidates = [{"text": hit["entity"]["text"], "source": hit["entity"]["source"]} for hit in search_res[0]]
            rerank_results = bge_reranker(query=query_text, documents=[c["text"] for c in candidates])

            final_top_k = []
            for res in rerank_results[:top_k]:
                doc = candidates[res.index]
                doc["rerank_score"] = res.score
                final_top_k.append(doc)

            res_str = "💡 [当前模式：混合检索 + BGE Rerank]\n\n"
            for i, hit in enumerate(final_top_k):
                res_str += f"📄 【片段 {i + 1}】 (来源: {hit['source']} | 重排打分: {hit['rerank_score']:.4f})\n{hit['text']}\n" + "-" * 40 + "\n\n"
            return res_str.strip()
    except Exception as e:
        return f"⚠️ 检索失败: {e}"


def delete_knowledge_base(kb_name):
    import gradio as gr
    if not kb_name: return "⚠️ 请先在下拉列表中选择要删除的知识库", gr.update()
    try:
        if milvus_client.has_collection(collection_name=kb_name):
            milvus_client.drop_collection(collection_name=kb_name)
        return f"🗑️ 已成功彻底删除知识库: {kb_name}", gr.update(choices=get_kb_list(), value=None)
    except Exception as e:
        return f"⚠️ 删除失败: {e}", gr.update()
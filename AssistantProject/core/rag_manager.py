# core/rag_manager.py
import os
import re
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from markitdown import MarkItDown
from AssistantProject.core.logger import logger

# ==========================================
# 1. 延迟加载 (Lazy Load) 与全局单例模式管理
# ==========================================
MILVUS_URI = "http://127.0.0.1:19530"
MODEL_PATH = r"D:\桌面等\人工智能助手\AssistantProject\models\BAAI\bge-m3"
RERANK_MODEL_PATH = r"D:\桌面等\人工智能助手\AssistantProject\models\BAAI\bge-reranker-v2-m3"

_milvus_client = None
_bge_m3_ef = None
_bge_reranker = None


def get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        try:
            logger.info(f"尝试连接到 Milvus 服务器: {MILVUS_URI} ...")
            _milvus_client = MilvusClient(uri=MILVUS_URI)
            logger.info("✅ 成功连接到本地 Milvus 服务器！")
        except Exception as e:
            logger.error(f"❌ 无法连接到 Milvus 服务器: {e}")
            raise e
    return _milvus_client


def check_milvus_connection() -> bool:
    """【状态守卫】: 检查 Milvus 数据库是否在线，防止无意义的卡顿和报错"""
    try:
        from pymilvus import connections
        # 尝试快速连接探测
        connections.connect("default", uri=MILVUS_URI, timeout=2)
        return True
    except Exception as e:
        logger.warning(f"⚠️ Milvus 状态守卫拦截：数据库未连接 ({e})")
        return False


def get_bge_m3_ef():
    global _bge_m3_ef
    if _bge_m3_ef is None:
        try:
            logger.info(f"尝试加载本地 BGE-M3 模型: {MODEL_PATH} ...")
            _bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=MODEL_PATH,
                use_fp16=True,
                device="cuda:0"
            )
            logger.info("✅ 本地 BGE-M3 模型加载成功 (GPU 加速开启)！")
        except Exception as e:
            logger.error(f"❌ 本地 BGE-M3 模型加载失败: {e}")
            raise e
    return _bge_m3_ef


def get_bge_reranker():
    global _bge_reranker
    if _bge_reranker is None:
        try:
            logger.info(f"尝试加载本地 BGE-Reranker 模型: {RERANK_MODEL_PATH} ...")
            _bge_reranker = BGERerankFunction(
                model_name=RERANK_MODEL_PATH,
                device="cuda:0"
            )
            logger.info("✅ 本地 BGE-Reranker 模型加载成功 (GPU 加速开启)！")
        except Exception as e:
            logger.error(f"❌ 本地 BGE-Reranker 模型加载失败: {e}")
            raise e
    return _bge_reranker


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
# 2. 结构化解析与分块逻辑
# ==========================================
def extract_and_split(file_path, chunk_size, chunk_overlap):
    chunks = []
    try:
        bge_ef = get_bge_m3_ef()
        local_embeddings = LocalBGEEmbeddings(bge_ef)
        semantic_chunker = SemanticChunker(local_embeddings, breakpoint_threshold_type="percentile")
        file_ext = file_path.lower()

        if file_ext.endswith(('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx')):
            logger.info(f"📄 正在使用 MarkItDown 解析复杂文档: {file_path}")
            md_converter = MarkItDown()
            result = md_converter.convert(file_path)
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
                            title_prefix = f"[{split_doc.metadata.get('title', '')}]\n" if split_doc.metadata.get('title') else ""
                            chunks.append(title_prefix + split_doc.page_content)
                    else:
                        title_prefix = f"[{d.metadata.get('title', '')}]\n" if d.metadata.get('title') else ""
                        chunks.append(title_prefix + d.page_content)
            except Exception as unstruct_e:
                logger.warning(f"⚠️ Unstructured 解析失败，降级为普通语义切块: {unstruct_e}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs = [Document(page_content=text)]
                chunks = [doc.page_content for doc in semantic_chunker.split_documents(docs)]

        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            docs = [Document(page_content=text)]
            chunks = [doc.page_content for doc in semantic_chunker.split_documents(docs)]

    except Exception as e:
        logger.error(f"❌ 解析文件 {file_path} 失败: {e}")

    return [c.replace('\x00', '').strip() for c in chunks if c.strip()]


# ==========================================
# 3. 知识库管理与数据插入 (双向量融合)
# ==========================================
def get_kb_list():
    if not check_milvus_connection():
        return []
    try:
        client = get_milvus_client()
        return client.list_collections()
    except Exception as e:
        logger.error(f"❌ 获取知识库列表失败: {e}")
        return []


def process_and_store_documents(file_paths, kb_name, chunk_size, chunk_overlap):
    if not file_paths:
        raise ValueError("请先上传文件！")
    if not kb_name:
        raise ValueError("请输入知识库名称！")

    if re.search(r'[^a-zA-Z0-9_]', kb_name):
        raise ValueError("知识库名称仅支持【英文、数字、下划线】！")

    safe_kb_name = kb_name
    
    try:
        client = get_milvus_client()
        bge_ef = get_bge_m3_ef()
    except Exception as e:
        raise RuntimeError(f"核心组件加载失败: {e}")

    try:
        if not client.has_collection(collection_name=safe_kb_name):
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1000)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

            index_params = client.prepare_index_params()
            index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

            client.create_collection(
                collection_name=safe_kb_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"✨ 成功创建知识库集合: {safe_kb_name}")
    except Exception as e:
        logger.error(f"❌ 创建集合失败: {e}")
        raise RuntimeError(f"创建知识库失败: {e}")

    total_chunks = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        chunks = extract_and_split(file_path, chunk_size, chunk_overlap)

        if not chunks:
            logger.warning(f"⚠️ 文件 {file_name} 解析为空，已跳过。")
            continue

        try:
            embeddings = bge_ef.encode_documents(chunks)
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
                safe_dense = dense_vecs[i].tolist() if hasattr(dense_vecs[i], "tolist") else [float(v) for v in dense_vecs[i]]
                data_to_insert.append({
                    "dense_vector": safe_dense,
                    "sparse_vector": safe_sparse_dicts[i],
                    "text": chunks[i],
                    "source": file_name
                })

            if data_to_insert:
                client.insert(collection_name=safe_kb_name, data=data_to_insert)
                total_chunks += len(chunks)
                logger.info(f"✅ 文件 {file_name} 插入 {len(chunks)} 个分块")
        except Exception as e:
            logger.error(f"❌ 插入文件 {file_name} 向量数据失败: {e}")

    if total_chunks > 0:
        try:
            logger.info(f"💾 正在将知识库 [{safe_kb_name}] 强制落盘并加载到内存...")
            client.flush(collection_name=safe_kb_name)
            client.load_collection(collection_name=safe_kb_name)
            logger.info("✅ 知识库就绪！")
        except Exception as e:
            logger.error(f"⚠️ 知识库落盘失败: {e}")
            raise RuntimeError(f"数据已插入，但知识库加载失败: {e}")
    else:
        raise ValueError("未能从文件中提取到有效内容，知识库构建失败。")

    return f"✅ 成功！知识库 [{safe_kb_name}] 注入 {total_chunks} 个片段 (语义级切片 | GPU加速)。"


def retrieve_documents(kb_name, query_text, strategy="混合检索 + BGE Rerank", top_k=2, target_model="qwen-max"):
    if not kb_name: return "⚠️ 请先在上方选择一个知识库！"
    if not query_text.strip(): return "⚠️ 请输入测试问题！"
    
    try:
        client = get_milvus_client()
        bge_ef = get_bge_m3_ef()
    except Exception as e:
        return f"⚠️ 底层组件加载失败: {e}"

    # 1. 自适应 RAG (Adaptive RAG) - 意图路由
    if strategy == "自适应 RAG":
        try:
            from AssistantProject.core.agent import get_llm
            from langchain_core.messages import SystemMessage, HumanMessage
            
            router_llm = get_llm(target_model, 100, 0.1)
            sys_msg = "你是一个智能路由系统。请判断用户的输入是否属于日常闲聊、打招呼等无需查阅专业知识库的内容。如果是闲聊，请回复 'CHITCHAT'，如果需要专业知识，请回复 'KNOWLEDGE_BASE'。请仅回复这两个词之一。"
            res = router_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=query_text)])
            if "CHITCHAT" in res.content:
                return "💡 [自适应 RAG 判定] 当前问题被识别为日常闲聊，无需查询知识库。"
            
            # 继续执行知识库检索
            strategy = "混合检索 + BGE Rerank"
        except Exception as e:
            logger.warning(f"自适应 RAG 路由失败: {e}，降级为普通检索")
            strategy = "混合检索 + BGE Rerank"

    try:
        if not client.has_collection(collection_name=kb_name):
            return f"⚠️ 知识库 {kb_name} 不存在！"
            
        client.load_collection(collection_name=kb_name)
        
        query_embeddings = bge_ef.encode_queries([query_text])
        raw_dense_query = query_embeddings["dense"][0]
        safe_dense_query = raw_dense_query.tolist() if hasattr(raw_dense_query, "tolist") else [float(v) for v in raw_dense_query]

        if "基础向量检索" in strategy:
            search_res = client.search(collection_name=kb_name, data=[safe_dense_query],
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
                safe_sparse_query = {int(raw_sparse_query.indices[j]): float(raw_sparse_query.data[j]) for j in range(raw_sparse_query.indptr[0], raw_sparse_query.indptr[1])}
            else:
                safe_sparse_query = raw_sparse_query[0] if isinstance(raw_sparse_query, list) else raw_sparse_query

            candidate_limit = top_k * 5
            dense_req = AnnSearchRequest(data=[safe_dense_query], anns_field="dense_vector",
                                         param={"metric_type": "COSINE"}, limit=candidate_limit)
            sparse_req = AnnSearchRequest(data=[safe_sparse_query], anns_field="sparse_vector",
                                          param={"metric_type": "IP"}, limit=candidate_limit)

            search_res = client.hybrid_search(collection_name=kb_name, reqs=[dense_req, sparse_req],
                                              ranker=RRFRanker(k=60), limit=candidate_limit,
                                              output_fields=["text", "source"])
            if not search_res or not search_res[0]: return "📭 知识库中未检索到相关内容。"

            candidates = [{"text": hit["entity"]["text"], "source": hit["entity"]["source"]} for hit in search_res[0]]
            
            # 使用重排模型
            reranker = get_bge_reranker()
            rerank_results = reranker(query=query_text, documents=[c["text"] for c in candidates])

            final_top_k = []
            for res in rerank_results:
                if res.score > 0.3:  # 提高阈值，过滤低关联度文本（>0.3更严格）
                    doc = candidates[res.index]
                    doc["rerank_score"] = res.score
                    final_top_k.append(doc)
                if len(final_top_k) >= top_k:
                    break
                    
            if not final_top_k:
                return "📭 未在知识库中找到高关联性的内容 (已自动过滤低分垃圾片段)。"

            res_str = f"💡 [当前模式：{strategy}]\n\n"
            for i, hit in enumerate(final_top_k):
                res_str += f"📄 【片段 {i + 1}】 (来源: {hit['source']} | 重排打分: {hit['rerank_score']:.4f})\n{hit['text']}\n" + "-" * 40 + "\n\n"
            
            # 2. 自我纠正 RAG (Self-Corrective RAG)
            if strategy == "自我纠正 RAG":
                try:
                    from AssistantProject.core.agent import get_llm
                    from langchain_core.messages import SystemMessage, HumanMessage
                    
                    eval_llm = get_llm(target_model, 100, 0.1)
                    context_text = "\n".join([c["text"] for c in final_top_k])
                    sys_msg = "你是一个检索评估员。请判断提供的参考资料是否能够回答用户的问题。如果完全不相关或无法回答，请回复 'IRRELEVANT'，并给出一个重写后的更清晰的查询建议。如果相关，请回复 'RELEVANT'。请严格按照 '判断结果|重写建议' 的格式返回，例如 'IRRELEVANT|什么是XXX'"
                    res = eval_llm.invoke([
                        SystemMessage(content=sys_msg),
                        HumanMessage(content=f"参考资料: {context_text}\n用户问题: {query_text}")
                    ])
                    
                    if "IRRELEVANT" in res.content:
                        suggestion = res.content.split("|")[-1].strip() if "|" in res.content else query_text
                        res_str = f"> [!WARNING]\n> **自我纠正 RAG 判定：检索失败**\n> 检索到的资料可能与问题无关。系统建议大模型将问题重写为: `{suggestion}`，并直接使用自身知识库回答。\n\n---\n" + res_str
                except Exception as e:
                    logger.warning(f"自我纠正 RAG 评估失败: {e}")

            return res_str.strip()
    except Exception as e:
        logger.error(f"❌ 检索失败: {e}")
        return f"⚠️ 检索失败: {e}"


def delete_knowledge_base(kb_name):
    if not kb_name: 
        raise ValueError("请先在下拉列表中选择要删除的知识库")
    try:
        client = get_milvus_client()
        if client.has_collection(collection_name=kb_name):
            client.drop_collection(collection_name=kb_name)
        logger.info(f"🗑️ 已成功彻底删除知识库: {kb_name}")
        return f"🗑️ 已成功彻底删除知识库: {kb_name}"
    except Exception as e:
        logger.error(f"❌ 删除知识库 {kb_name} 失败: {e}")
        raise RuntimeError(f"删除失败: {e}")

# ==========================================
# 4. 细粒度文档管理 (CRUD at Document Level)
# ==========================================
def get_kb_files(kb_name):
    if not kb_name:
        return []
    try:
        client = get_milvus_client()
        if not client.has_collection(collection_name=kb_name):
            return []
        
        # 宽泛查询以获取唯一的 source
        res = client.query(
            collection_name=kb_name,
            filter="id >= 0", 
            output_fields=["source"],
            limit=16384
        )
        unique_sources = list(set([hit["source"] for hit in res]))
        return sorted(unique_sources)
    except Exception as e:
        logger.error(f"❌ 获取知识库文件列表失败: {e}")
        return []

def delete_file_from_kb(kb_name, file_name):
    if not kb_name or not file_name:
        raise ValueError("请提供知识库名称和要删除的文件名！")
    try:
        client = get_milvus_client()
        if not client.has_collection(collection_name=kb_name):
            raise ValueError(f"知识库 {kb_name} 不存在。")
            
        client.delete(collection_name=kb_name, filter=f"source == '{file_name}'")
        logger.info(f"🗑️ 已成功从知识库 {kb_name} 中删除文件: {file_name}")
        return f"✅ 成功从知识库 [{kb_name}] 删除文档: {file_name}"
    except Exception as e:
        logger.error(f"❌ 删除知识库文件失败: {e}")
        raise RuntimeError(f"删除文档失败: {e}")
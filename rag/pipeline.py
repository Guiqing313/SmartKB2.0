# rag/pipeline.py —— 检索流水线（Step2：向量 + BM25 + RRF + 重排）
import jieba
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import MilvusClient

_model = None
_client = None
_bm25 = None
_bm25_docs = None
_reranker = None

def get_model():
    """加载 bge-m3（全局只加载一次）"""
    global _model
    if _model is None:
        _model = BGEM3FlagModel("D:/codex使用文件夹/SmartKB2.0/models/bge-m3", use_fp16=True)
    return _model

def get_client():
    """打开 Milvus（全局只开一次）"""
    global _client
    if _client is None:
        _client = MilvusClient("D:/milvus_lite/milvus_lite.db")
        _client.load_collection("kb_bge")
    return _client

def get_bm25():
    """从 Milvus 拉全部文档，构建 BM25 索引（只建一次）"""
    global _bm25, _bm25_docs
    if _bm25 is None:
        client = get_client()
        rows = client.query("kb_bge", filter="id >= 0", output_fields=["text"])
        _bm25_docs = [r["text"] for r in rows]
        _bm25 = BM25Okapi([list(jieba.cut(d)) for d in _bm25_docs])
    return _bm25, _bm25_docs

def get_reranker():
    """加载 bge-reranker（只加载一次）"""
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("D:/codex使用文件夹/SmartKB2.0/models/bge-reranker-large", use_fp16=True)
    return _reranker

def retrieve(query, top_k=3, recall_n=6):
    """混合检索：向量 + BM25 → RRF 融合 → 重排 → top_k"""
    model = get_model()
    client = get_client()
    bm25, docs = get_bm25()

    # 1. 向量路：Milvus 检索 top-N
    q_vec = model.encode([query], return_dense=True)["dense_vecs"][0]
    vec_res = client.search("kb_bge", data=[q_vec.tolist()], limit=recall_n, output_fields=["text"])
    vec_rank = {hit["entity"]["text"]: i for i, hit in enumerate(vec_res[0])}

    # 2. BM25 路：全量打分排名
    scores = bm25.get_scores(list(jieba.cut(query)))
    bm25_order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    bm25_rank = {docs[i]: r for r, i in enumerate(bm25_order)}

    # 3. RRF 融合（k=60）
    k = 60
    rrf = {}
    for doc, r in vec_rank.items():
        rrf[doc] = 1 / (k + r + 1)
    for doc, r in bm25_rank.items():
        rrf[doc] = rrf.get(doc, 0) + 1 / (k + r + 1)
    candidates = sorted(rrf, key=lambda d: rrf[d], reverse=True)[:recall_n]

    # 4. bge-reranker 精排
    reranker = get_reranker()
    pairs = [[query, d] for d in candidates]
    r_scores = reranker.compute_score(pairs)
    ordered = [d for _, d in sorted(zip(r_scores, candidates), key=lambda x: x[0], reverse=True)]

    return ordered[:top_k]
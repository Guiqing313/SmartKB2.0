# rag/pipeline.py —— 检索流水线（阶段4第一步：向量检索）
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient

_model = None
_client = None

def get_model():
    """加载 bge-m3（只加载一次，全局复用）"""
    global _model
    if _model is None:
        _model = BGEM3FlagModel("D:/codex使用文件夹/SmartKB2.0/models/bge-m3", use_fp16=True)
    return _model

def get_client():
    """打开 Milvus Lite（只打开一次）"""
    global _client
    if _client is None:
        _client = MilvusClient("D:/milvus_lite/milvus_lite.db")
        _client.load_collection("kb_bge")
    return _client

def retrieve(query, top_k=3):
    """检索：query → 向量 → Milvus → 返回相关文档列表"""
    model = get_model()
    client = get_client()
    q_vec = model.encode([query], return_dense=True)["dense_vecs"][0]
    res = client.search("kb_bge", data=[q_vec.tolist()], limit=top_k, output_fields=["text"])
    return [hit["entity"]["text"] for hit in res[0]]
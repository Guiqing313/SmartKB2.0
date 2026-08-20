# rag/test_milvus.py
from pymilvus import MilvusClient

# 1. 创建 Milvus Lite 客户端（本地文件模式，免 Docker）
client = MilvusClient("D:/codex使用文件夹/SmartKB2.0/data/milvus_lite.db")

# 2. 建 collection：定义字段（先用 4 维测试，bge-m3 实际是 1024 维）
client.create_collection(
    collection_name="test_kb",
    dimension=4,
    metric_type="COSINE",   # 余弦相似度
)

# 3. 插入 3 条"文档片段"（先用假向量测试，后面换 bge-m3 真实向量）
client.insert("test_kb", [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "text": "RAG 是检索增强生成"},
    {"id": 2, "vector": [0.9, 0.8, 0.7, 0.6], "text": "LoRA 是低秩微调"},
    {"id": 3, "vector": [0.2, 0.1, 0.4, 0.3], "text": "向量数据库存向量"},
])

# 4. 搜索：用查询向量找最像的 2 条
res = client.search(
    collection_name="test_kb",
    data=[[0.15, 0.25, 0.35, 0.45]],   # 查询向量（故意接近第 1 条）
    limit=2,
    output_fields=["text"],
)
for hit in res[0]:
    print(f"相似度 {hit['distance']:.4f} -> {hit['entity']['text']}")
# rag/ingest_bge.py
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient

# 1. 加载 bge-m3
model = BGEM3FlagModel("D:/codex使用文件夹/SmartKB2.0/models/bge-m3", use_fp16=True)

# 2. 知识库"文档块"（模拟）
docs = [
    "RAG 是检索增强生成，通过检索相关文档辅助大模型回答",
    "LoRA 通过低秩矩阵微调大模型，只训练少量参数",
    "向量数据库用于存储和检索高维向量，如 Milvus",
    "bge-m3 是支持中英文的嵌入模型，输出 1024 维向量",
]

# 3. 文档 → 真实向量（1024 维）
vecs = model.encode(docs, return_dense=True)["dense_vecs"]

# 4. 存入 Milvus（注意 dimension=1024，和 bge-m3 一致！）
client = MilvusClient("D:/codex使用文件夹/SmartKB2.0/data/milvus_lite.db")
client.create_collection(collection_name="kb_bge", dimension=1024, metric_type="COSINE")
rows = [{"id": i+1, "vector": vecs[i].tolist(), "text": docs[i]} for i in range(len(docs))]
client.insert("kb_bge", rows)

# 5. 真实查询检索（top-2）
query = "怎么微调大模型？"
q_vec = model.encode([query], return_dense=True)["dense_vecs"][0]
res = client.search(collection_name="kb_bge", data=[q_vec.tolist()], limit=2, output_fields=["text"])
for hit in res[0]:
    print(f"{hit['distance']:.4f} -> {hit['entity']['text']}")
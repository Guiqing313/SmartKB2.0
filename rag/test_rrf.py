# rag/test_rrf.py —— 混合检索：向量 + BM25 → RRF 融合
import jieba
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient

docs = [
    "RAG 是检索增强生成，通过检索相关文档辅助大模型回答",
    "LoRA 通过低秩矩阵微调大模型，只训练少量参数",
    "向量数据库用于存储和检索高维向量，如 Milvus",
    "bge-m3 是支持中英文的嵌入模型，输出 1024 维向量",
]
query = "怎么微调大模型？"

# ===== 路 1：向量检索（bge-m3 + Milvus）=====
model = BGEM3FlagModel("D:/codex使用文件夹/SmartKB2.0/models/bge-m3", use_fp16=True)
q_vec = model.encode([query], return_dense=True)["dense_vecs"][0]
client = MilvusClient("D:/milvus_lite/milvus_lite.db")
client.load_collection("kb_bge")   # 搜索前必须把集合加载进内存
vec_res = client.search("kb_bge", data=[q_vec.tolist()], limit=4, output_fields=["text"])
vec_rank = []
for hit in vec_res[0]:
    vec_rank.append(docs.index(hit["entity"]["text"]))   # 按相似度从高到低的文档序号
print("向量路排名:", vec_rank)

# ===== 路 2：BM25 关键词检索 =====
tokenized = [list(jieba.cut(d)) for d in docs]
bm25 = BM25Okapi(tokenized)
scores = bm25.get_scores(list(jieba.cut(query)))
bm25_rank = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
print("BM25路排名:", bm25_rank)

# ===== RRF 融合 =====
k = 60
rrf = [0.0] * len(docs)
for rank, idx in enumerate(vec_rank):
    rrf[idx] += 1 / (k + rank + 1)      # rank 从 0 开始，公式用 rank+1
for rank, idx in enumerate(bm25_rank):
    rrf[idx] += 1 / (k + rank + 1)
print("RRF 融合得分:", [round(x, 5) for x in rrf])
final = sorted(range(len(docs)), key=lambda i: rrf[i], reverse=True)
print("最终排名:", final)
print("最终最相关:", docs[final[0]])
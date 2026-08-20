# rag/test_rerank.py —— bge-reranker 精排
from FlagEmbedding import FlagReranker

# 1. 加载 bge-reranker-large（你已下载到 models/）
reranker = FlagReranker("D:/codex使用文件夹/SmartKB2.0/models/bge-reranker-large", use_fp16=True)

query = "怎么微调大模型？"
docs = [
    "RAG 是检索增强生成，通过检索相关文档辅助大模型回答",
    "LoRA 通过低秩矩阵微调大模型，只训练少量参数",
    "向量数据库用于存储和检索高维向量，如 Milvus",
    "bge-m3 是支持中英文的嵌入模型，输出 1024 维向量",
]

# 2. 查询 + 每条文档"成对"打分
pairs = [[query, d] for d in docs]
scores = reranker.compute_score(pairs)
print("重排分数:", scores)

# 3. 按分数从高到低排序
order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
print("重排后顺序:", order)
print("最相关:", docs[order[0]])
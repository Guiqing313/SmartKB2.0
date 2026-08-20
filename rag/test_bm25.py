# rag/test_bm25.py
import jieba
from rank_bm25 import BM25Okapi

# 1. 同样的 4 个文档
docs = [
    "RAG 是检索增强生成，通过检索相关文档辅助大模型回答",
    "LoRA 通过低秩矩阵微调大模型，只训练少量参数",
    "向量数据库用于存储和检索高维向量，如 Milvus",
    "bge-m3 是支持中英文的嵌入模型，输出 1024 维向量",
]

# 2. jieba 分词（中文没有空格，必须分词）
tokenized = [list(jieba.cut(d)) for d in docs]
print("分词示例:", tokenized[1])

# 3. 构建 BM25 索引
bm25 = BM25Okapi(tokenized)

# 4. 查询："怎么微调大模型"
query = "怎么微调大模型"
q_tokens = list(jieba.cut(query))
scores = bm25.get_scores(q_tokens)
print("查询分词:", q_tokens)
print("各文档得分:", scores)
best = scores.argmax()
print("最相关文档:", docs[best])
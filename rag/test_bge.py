# rag/test_bge.py
from FlagEmbedding import BGEM3FlagModel

# 1. 加载本地 bge-m3 模型（你已下载到 models/bge-m3）
model = BGEM3FlagModel("D:/codex使用文件夹/SmartKB2.0/models/bge-m3", use_fp16=True)

# 2. 3 句话 → 向量（return_dense 出稠密向量）
sentences = ["RAG 是检索增强生成", "LoRA 是低秩微调", "向量数据库存向量"]
out = model.encode(sentences, return_dense=True)

# 3. 打印维度：应该 (3, 1024)
print("向量维度:", out["dense_vecs"].shape)

# 4. 查询："什么是RAG？" 转向量，算它和 3 句话的相似度
q = model.encode(["什么是RAG？"], return_dense=True)
sims = out["dense_vecs"] @ q["dense_vecs"].T
print("相似度:", sims.flatten())
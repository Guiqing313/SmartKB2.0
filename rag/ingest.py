# rag/ingest.py —— 真实文档入库：读取 → 分块 → bge-m3 → Milvus
import os
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient

DOCS_DIR = "D:/codex使用文件夹/SmartKB2.0/data/docs"
MILVUS_DB = "D:/milvus_lite/milvus_lite.db"
COLLECTION = "kb_bge"
CHUNK_SIZE = 500      # 每块 500 字符
OVERLAP = 50          # 重叠 50 字符

def split_text(text, source):
    """按字符分块，带 overlap（避免切断语义）"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({"text": text[start:end], "source": source})
        start = end - OVERLAP
    return chunks

def load_all_docs():
    """读取 data/docs 下所有 .md/.txt 并分块"""
    all_chunks = []
    for fname in os.listdir(DOCS_DIR):
        fpath = os.path.join(DOCS_DIR, fname)
        if not os.path.isfile(fpath) or not fname.endswith(('.md', '.txt')):
            continue
        with open(fpath, encoding='utf-8') as f:
            text = f.read()
        chunks = split_text(text, fname)
        all_chunks.extend(chunks)
        print(f"  {fname}: {len(chunks)} 块")
    return all_chunks

def main():
    print("=== 1. 读取文档并分块 ===")
    chunks = load_all_docs()
    print(f"共 {len(chunks)} 个分块")

    print("=== 2. bge-m3 向量化 ===")
    model = BGEM3FlagModel("D:/codex使用文件夹/SmartKB2.0/models/bge-m3", use_fp16=True)
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, return_dense=True)["dense_vecs"]
    print(f"向量维度: {vecs.shape}")

    print("=== 3. 入库 Milvus（重建集合，带 source 字段） ===")
    client = MilvusClient(MILVUS_DB)
    client.drop_collection(COLLECTION)   # 清掉旧的 4 条测试数据
    client.create_collection(COLLECTION, dimension=1024, metric_type="COSINE")
    rows = [
        {"id": i + 1, "vector": vecs[i].tolist(), "text": texts[i], "source": chunks[i]["source"]}
        for i in range(len(texts))
    ]
    client.insert(COLLECTION, rows)
    print(f"入库完成：{len(rows)} 条")

if __name__ == "__main__":
    main()
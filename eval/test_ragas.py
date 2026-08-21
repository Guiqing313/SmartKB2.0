# eval/test_ragas.py
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. 用 Ollama 当 LLM（OpenAI 兼容）
generator = LangchainLLMWrapper(ChatOpenAI(
    model="qwen2.5:7b", base_url="http://localhost:11434/v1", api_key="ollama",
))

# 2. 用 Ollama 的 embedding 模型
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434/v1", api_key="ollama",
))

# 3. 把 llm / embeddings 绑到指标上
for m in [faithfulness, answer_relevancy, context_precision, context_recall]:
    m.llm = generator
for m in [answer_relevancy, context_precision, context_recall]:
    m.embeddings = embeddings

# 4. 评测数据：问题 / 回答 / 检索到的上下文 / 参考答案
data = {
    "question": ["什么是RAG？", "怎么微调大模型？", "Milvus 是什么？", "bge-m3 支持什么语言？"],
    "answer": [
        "RAG 是检索增强生成，通过检索文档辅助生成。",
        "用 LoRA 低秩微调，只训练少量参数。",
        "Milvus 是向量数据库。",
        "bge-m3 支持中英文。",
    ],
    "contexts": [
        ["RAG 是检索增强生成，通过检索相关文档辅助大模型回答"],
        ["LoRA 通过低秩矩阵微调大模型，只训练少量参数"],
        ["向量数据库用于存储和检索高维向量，如 Milvus"],
        ["bge-m3 是支持中英文的嵌入模型，输出 1024 维向量"],
    ],
    "ground_truth": [
        "RAG 是检索增强生成（Retrieval-Augmented Generation）。",
        "通过 LoRA 等低秩方法微调。",
        "Milvus 是开源的向量数据库。",
        "bge-m3 支持中文和英文。",
    ],
}
ds = Dataset.from_dict(data)

# 5. 跑评测
result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
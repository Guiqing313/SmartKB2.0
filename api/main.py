# 1. 导入 FastAPI、Pydantic、requests
from fastapi import FastAPI          # FastAPI 主框架
from pydantic import BaseModel       # 用于定义请求体格式
import requests                      # 用来调用 Ollama 的 HTTP 接口
from rag.pipeline import retrieve   # 导入检索流水线
# 2. 创建 FastAPI 应用实例
app = FastAPI()

# 3. Ollama 服务地址（你在任务3验证过：http://localhost:11434）
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen2.5:7b"

# 4. 定义 /health：GET，返回 {"status": "ok"}
@app.get("/health")
def health():
    return {"status": "ok"}

# 5. 定义 /chat 的请求体格式：{"question": "..."}
class ChatRequest(BaseModel):
    question: str

# 6. 定义 /chat：POST，接收问题 → 调 Ollama → 返回回答
@app.post("/chat")
def chat(req: ChatRequest):
    # 1. 检索：找到相关资料（top-3）
    docs = retrieve(req.question, top_k=3)
    context = "\n".join(docs)

    # 2. 构造"带上下文"的提示词（关键：约束模型只用资料回答）
    prompt = (
        f"请仅基于以下资料回答用户问题；如果资料里没有答案，"
        f"直接回答'未找到相关信息'。\n\n资料：\n{context}\n\n问题：{req.question}"
    )
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    answer = resp.json()["choices"][0]["message"]["content"]

    # 3. 返回回答 + 引用来源
    return {"answer": answer, "citations": docs}
# 1. 导入 FastAPI、Pydantic、requests
from fastapi import FastAPI          # FastAPI 主框架
from pydantic import BaseModel       # 用于定义请求体格式
import requests                      # 用来调用 Ollama 的 HTTP 接口

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
    # 6.1 构造要发给 Ollama 的请求体（结构和任务3一模一样）
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": req.question}]
    }
    # 6.2 发送 POST 请求到 Ollama
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    # 6.3 从返回 JSON 里取出模型的回答（路径：choices[0].message.content）
    answer = resp.json()["choices"][0]["message"]["content"]
    # 6.4 包装成 {"answer": "..."} 返回给前端
    return {"answer": answer}
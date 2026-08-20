# SmartKB 2.0 — 企业级 RAG 知识库问答系统

> 把 SmartKB 从"调 API 的 RAG 知识库"升级为 **自部署模型 + 检索优化 + 微调 + 评测** 的全链路企业级知识库问答系统。

## ✨ 特性
- 🔍 **RAG 全链路**：文档解析 → 分块 → 向量化 → 混合检索 → 重排 → 生成
- 🧠 **本地推理**：Ollama + Qwen2.5-7B（OpenAI 兼容 API，数据不出本地）
- 🔀 **混合检索**：向量（bge-m3 dense+sparse）+ BM25 + RRF 融合 + bge-reranker 精排
- 📚 **多轮对话**：会话记忆 + 上下文截断 + 引用溯源（文档名/页码）
- 📊 **评测闭环**：RAGAS 四指标（faithfulness / answer_relevancy / context_precision / context_recall）
- 🎯 **微调能力**：LLaMA-Factory QLoRA 微调（8GB 显存可跑 7B）

## 🛠️ 技术栈
Python · FastAPI · Ollama · Qwen2.5-7B · Milvus Lite · bge-m3 · bge-reranker · BM25 · LLaMA-Factory · RAGAS · Streamlit · Docker

## 🚀 快速开始
```bash
# 1. 环境（Python 3.10 + CUDA torch）
# 2. 启动 Ollama 并拉取模型
ollama pull qwen2.5:7b
# 3. 启动后端（FastAPI 网关）
python -m uvicorn api.main:app --port 8000
# 4. 接口文档（Swagger）
# 打开 http://localhost:8000/docs
```

## 📁 目录结构
```
api/         FastAPI 网关（/health、/chat）
rag/         检索链路（retriever / reranker / generator / ingest）
inference/   模型推理配置（Ollama）
finetune/    LLaMA-Factory 微调配置与数据
eval/        RAGAS 评测脚本与评测集
frontend/    Streamlit 前端
deploy/      Docker 部署
```

## 📌 项目状态
阶段 0-2 完成（环境就绪、架构确认、FastAPI 网关调通）；阶段 3+ 进行中。

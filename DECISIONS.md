# DECISIONS — SmartKB 2.0 关键决策记录

> 记录影响架构/行为的决策与理由，防止方案被悄悄改变。

### 决策 1：模型选型
- 日期：2026-08-11
- 选项对比：
  - A：Qwen2.5-7B-Instruct（LoRA/QLoRA 微调 + vLLM 部署）—— 优点：大厂 JD 高频、生态成熟、简历含金量高；缺点：8GB 显存需量化/QLoRA
  - B：Qwen2.5-3B-Instruct —— 优点：轻量快；缺点：能力弱、简历价值低
  - C：DeepSeek 开源系 —— 优点：国产热点；缺点：7B 级工具链/文档成熟度不如 Qwen
- 选择：A（Qwen2.5-7B-Instruct）
- 理由：对标大厂 JD 的"Qwen + vLLM + LLaMA-Factory"组合；8GB 用 QLoRA(4bit) 微调 + AWQ 量化推理可行
- 后续影响：若 OOM，降级 3B 或减小 max_model_len / batch size

### 决策 2：是否保留 DeepSeek API 作为评测基线
- 日期：2026-08-11
- 选择：保留（作为 baseline 参与 RAGAS 三路对比：DeepSeek API / Qwen base / Qwen + LoRA）
- 理由：before/after 对比是面试亮点，证明"本地化+微调"的价值
- 复核点：API key 只放 .env，不进 git

### 决策 3：向量数据库
- 日期：2026-08-11
- 选项：A：Milvus（Docker standalone）；B：继续 Chroma；C：Qdrant
- 选择：A（Milvus）
- 理由：生产级、大厂 JD 高频、支持 dense+sparse 混合向量；与 SmartKB 的 Chroma 形成升级叙事
- 后续影响：Milvus standalone 需 etcd+minio，资源占用较大；若内存不足改用 pymilvus lite 兜底

### 决策 4：Embedding 与重排
- 日期：2026-08-11
- 选择：bge-m3（dense+sparse 双向量）+ bge-reranker-large
- 理由：中文检索强；一个模型出双向量省资源；重排是检索精度关键（对标企业级 RAG 做法）
- 复核点：bge-m3 权重约 2.2GB，需在 D 盘预留模型目录

### 决策 5：微调框架与策略
- 日期：2026-08-11
- 选择：LLaMA-Factory + QLoRA（4bit 量化微调）
- 理由：显存友好（8GB 可跑 7B）、主流工具、YAML 配置可复现、自带 WebUI 便于调试
- 复核点：微调数据集 200–500 条 alpaca/sharegpt 格式；训练时长预计 3–8 小时，安排夜间跑

### 决策 6：是否接入 Agent 模式
- 日期：2026-08-11
- 选择：默认不做（AgentForge 保留为独立项目）
- 理由：控制 30 天范围与复杂度；面试叙事为"新项目=RAG/部署/微调/评测 + AgentForge=Agent 编排"，互补不重复
- 复核点：若用户要求，可在 30 天后作为扩展（W5+）

### 决策 7：代码库组织
- 日期：2026-08-11
- 选择：新建独立 repo `SmartKB2.0`，从 SmartKB 迁移复用文档解析 / Streamlit / FastAPI 片段
- 理由：不破坏已有可演示的 SmartKB；新项目目录结构干净、便于简历单独呈现

### 决策 8：评测集与微调数据
- 日期：2026-08-11
- 选择：评测集 50–100 条真实领域问题（EVAL_SET.md）；微调数据集 200–500 条领域 QA（alpaca 格式）
- 理由：满足 RAGAS 最小样本量、微调效果可见；评测集由用户抽查断言（工作流 R6）

### 决策 9：推理引擎选型（2026-08-11 用户确认 → A）
- 日期：2026-08-11
- 选项对比：
  - A：Ollama（Windows 原生）—— 优点：免装 WSL/Docker、稳定、OpenAI 兼容 API、支持 LoRA 导入、真实 JD 认可"vLLM/Ollama 等至少一种"；缺点：性能弱于 vLLM、模型需 GGUF 格式
  - B：WSL2 + vLLM —— 优点：JD 关键词最响、企业级性能；缺点：本机未装 WSL、安装成本高、可能挤占 30 天工期
- 选择：A（Ollama）
- 理由：本机无 WSL/Docker；Ollama 是 30 天内最稳路径，先跑通再谈性能
- 后续影响/调整：
  - 模型格式：Qwen2.5-7B → GGUF（Ollama 拉取即用）
  - 微调衔接：LLaMA-Factory 导出 LoRA → Ollama Modelfile 导入（FROM + ADAPTER）
  - 向量库开发期：Milvus Lite（免 Docker）；生产期：Docker Compose + Milvus standalone
  - vLLM 保留为后续可选升级（不阻塞本期）

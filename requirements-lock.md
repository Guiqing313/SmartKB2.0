# requirements-lock.md —— 依赖清单（草稿，待网络环境安装后冻结）

> 用途：核对真实包名 + 锁版本（工作流 R8，防幻觉依赖）。安装完成后用 `pip freeze > requirements-lock-frozen.txt` 冻结精确版本。
> 环境：pytorch_env（Python 3.10.20，Windows + RTX 4060 8GB）

## 分组清单（均为真实 PyPI 包名）

### 推理（Ollama，Windows 原生）
| 包 | 说明 | 备注 |
|---|---|---|
| ollama | Ollama Python SDK（调本地服务） | 需先安装 Ollama 本体 |
| openai | OpenAI 兼容客户端（连 Ollama/DeepSeek） | 通用 |

### 微调（LLaMA-Factory + QLoRA）
| 包 | 说明 | 备注 |
|---|---|---|
| torch / torchvision / torchaudio | CUDA 版（cu124） | 官方源慢，可用交大镜像 |
| llamafactory | 微调框架 | 含 transformers/peft/datasets/accelerate 依赖 |
| bitsandbytes | 4bit 量化（QLoRA） | ⚠️ Windows 需预编译轮子（见下方风险） |
| unsloth | 替代方案（Windows 支持，免 bnb 折腾） | 备选 |

### 检索与向量
| 包 | 说明 | 备注 |
|---|---|---|
| pymilvus | Milvus/Milvus Lite 客户端 | 内嵌模式免 Docker |
| sentence-transformers | bge-m3 embedding | |
| FlagEmbedding | bge-m3 / bge-reranker 加载 | BAAI 官方 |
| rank-bm25 | BM25 关键词检索 | |
| jieba | 中文分词 | BM25 用 |

### 评测
| 包 | 说明 | 备注 |
|---|---|---|
| ragas | RAGAS 评测（faithfulness 等 4 指标） | |

### 后端 / 前端
| 包 | 说明 | 备注 |
|---|---|---|
| fastapi / uvicorn | API 网关 | |
| python-dotenv | .env 读取 | |
| pydantic | 数据校验 | |
| streamlit | 前端 | |
| slowapi | 限流（阶段 6） | |

### 文档解析
| 包 | 说明 | 备注 |
|---|---|---|
| pypdf | PDF 解析（pypdf2 已并入 pypdf） | 用 pypdf 而非 pypdf2 |
| python-docx | DOCX 解析 | |
| unstructured | 通用文档解析 | 可选，Windows 依赖多 |

### 工具
| 包 | 说明 | 备注 |
|---|---|---|
| huggingface-hub | 模型下载（HF 镜像） | |
| modelscope | 模型下载（国内源） | |
| tiktoken | token 计数/截断 | |
| ruff | lint（阶段 7） | |
| pip-audit | 依赖安全扫描（阶段 6） | 可选 |

## ⚠️ Windows 已知风险与对策
1. **bitsandbytes（QLoRA 4bit）在 Windows 上无官方轮子**：
   - 方案1：安装预编译轮子 `pip install bitsandbytes --index-url https://jllllll.github.io/bitsandbytes-windows-webui/`（社区维护）
   - 方案2（推荐备选）：改用 **unsloth**（`pip install unsloth`，已支持 Windows，自带 QLoRA 内核）
   - 方案3：若两者都不顺，微调改在 WSL2 跑（本期不首选）
2. **Ollama 拉取慢**：换网络/配置镜像，或浏览器手动下载 GGUF 后 `ollama create` 导入
3. **pypdf2 已并入 pypdf**：SmartKB 原代码用 pypdf2 的地方迁移到 pypdf

## 冻结方法（网络恢复后执行）
```bash
pip install -r requirements.txt   # 先装最小集
pip freeze > requirements-lock-frozen.txt   # 冻结精确版本
```

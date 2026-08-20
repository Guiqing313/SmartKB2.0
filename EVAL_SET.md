# EVAL_SET —— SmartKB 2.0 评测集（50 条，Day 13 定稿，Day 23 使用）

> 领域：AI 应用开发 / 大模型 / RAG / Agent / 微调 / 部署 / 评测。
> 用法：每条含 question / ground_truth 要点 / source_doc（入库文档）；Day 26 用户抽查 ≥10 条断言。
> 难度：基础（B）/ 进阶（M）/ 综合（H）。

## 一、LLM 基础（10 条）
| # | 问题 | 要点 | 难度 |
|---|---|---|---|
| 1 | 什么是大语言模型？ | 基于 Transformer、预测下一个 token | B |
| 2 | Transformer 的 Attention 是什么？ | 计算 token 间相关性权重 | B |
| 3 | KV Cache 为什么能加速推理？ | 缓存历史 K/V，避免重复计算 | M |
| 4 | 位置编码的作用？ | 注入 token 顺序信息 | M |
| 5 | 预训练、SFT、RLHF 各是什么？ | 三个阶段的目标差异 | M |
| 6 | temperature 参数影响什么？ | 输出随机性/确定性 | B |
| 7 | 上下文窗口超限怎么办？ | 截断/摘要/滑动窗口 | M |
| 8 | 什么是 tokenizer？ | 文本↔token 转换 | B |
| 9 | 模型为什么会幻觉？ | 预测统计规律而非事实 | M |
| 10 | 流式输出（SSE）原理？ | 分块响应、逐 token 推送 | M |

## 二、RAG（12 条）
| # | 问题 | 要点 | 难度 |
|---|---|---|---|
| 11 | 什么是 RAG？ | 检索增强生成：检索+生成 | B |
| 12 | RAG 相比纯 Prompt 的优势？ | 知识更新、可溯源、降幻觉 | B |
| 13 | RAG 完整流程？ | 解析→分块→向量化→存储→检索→重排→生成 | B |
| 14 | 分块策略怎么选？ | chunk_size/overlap 权衡 | M |
| 15 | 什么是混合检索？ | 向量+关键词互补 | M |
| 16 | RRF 融合怎么算？ | 1/(k+rank) 求和 | M |
| 17 | 为什么需要重排（rerank）？ | 初筛噪声多，精排提精度 | M |
| 18 | 什么是 HyDE？ | 先生成假设文档再检索 | M |
| 19 | Embedding 模型怎么选？ | 领域/语言/维度/成本 | M |
| 20 | 检索无结果怎么办？ | 兜底话术，不编造 | B |
| 21 | 引用溯源怎么做？ | chunk 带文档名/页码/相似度 | M |
| 22 | 向量库怎么选？ | Chroma 实验 vs Milvus 生产 | M |

## 三、Agent（8 条）
| # | 问题 | 要点 | 难度 |
|---|---|---|---|
| 23 | 什么是 Agent？ | 感知-规划-行动-反思循环 | B |
| 24 | ReAct 模式是什么？ | 思考→行动→观察循环 | M |
| 25 | LangGraph 状态图怎么设计？ | 节点/边/条件路由/State | M |
| 26 | Agent 间怎么通信？ | 共享 State / 消息 | M |
| 27 | 工具调用怎么实现？ | 函数 schema + 模型选工具 | M |
| 28 | 什么是 MCP？ | 工具服务器标准化协议 | M |
| 29 | 多 Agent 并行怎么实现？ | Send API / asyncio.gather | M |
| 30 | 如何防止 Agent 死循环？ | 最大迭代 + 评分阈值 | M |

## 四、微调（8 条）
| # | 问题 | 要点 | 难度 |
|---|---|---|---|
| 31 | 为什么要微调？ | 领域适配/风格对齐 | B |
| 32 | LoRA 原理？ | 低秩矩阵增量更新 | M |
| 33 | QLoRA 如何省显存？ | 4bit 量化 + 分页优化器 | M |
| 34 | 微调数据集一般多少条？ | 数百到数千即可见效 | M |
| 35 | alpaca 格式是什么？ | instruction/input/output | B |
| 36 | 微调可能带来什么问题？ | 灾难性遗忘/过拟合 | M |
| 37 | SFT 和 DPO 区别？ | 监督 vs 偏好优化 | M |
| 38 | 怎么评估微调效果？ | 评测集 + 指标对比 | M |

## 五、部署/推理（6 条）
| # | 问题 | 要点 | 难度 |
|---|---|---|---|
| 39 | Ollama 是什么？ | 本地模型运行工具，OpenAI 兼容 | B |
| 40 | GGUF 是什么格式？ | llama.cpp 量化格式 | M |
| 41 | 8GB 显存能跑多大模型？ | 7B 量化可跑，FP16 不够 | M |
| 42 | vLLM 与 Ollama 区别？ | 吞吐优化 vs 易用 | M |
| 43 | OpenAI 兼容 API 指什么？ | /v1/chat/completions 规范 | B |
| 44 | Docker 部署的好处？ | 环境隔离、一键拉起 | B |

## 六、评测与工程化（6 条）
| # | 问题 | 要点 | 难度 |
|---|---|---|---|
| 45 | RAGAS 四个指标是什么？ | faithfulness/answer_relevancy/context_precision/context_recall | M |
| 46 | faithfulness 衡量什么？ | 回答是否忠于上下文 | M |
| 47 | 评测集怎么构建？ | 真实业务问题 + 标准答案 | M |
| 48 | 怎么做 before/after 对比？ | 同一评测集多路跑 | M |
| 49 | 限流/成本控制怎么做？ | 限流器 + token 统计 | M |
| 50 | 生产 RAG 系统还要注意什么？ | 监控/安全/数据合规 | H |

## 使用方式
1. Day 13：把 50 条写入 `eval/eval_set.json`（question / ground_truth / source_doc）
2. Day 23：RAGAS 三路评测（DeepSeek API / qwen2.5:7b / qwen2.5-lora）
3. Day 26：请用户抽查 ≥10 条人工确认

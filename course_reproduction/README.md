# SERE 课程设计复现（资源受限版）

本目录给出可复查、可重复运行的 EventStoryLine 实验。实验以官方 SERE 仓库和论文同源的 ESC 1000 条样本为输入，随机种子固定为 42，并按 `doc_name` 做 60%/20%/20% 文档级划分。

## 三组实验

- E0：上游 `ChatGPT4CausalReasoning` 公开的 GPT-4 零样本响应。
- E1：SERE-local，`k=2`；包含概念路径代理、句法树代理、因果模式过滤和正负类平衡检索。
- E2：只把检索例数从 `k=2` 改为 `k=4`，其余参数保持不变。

## 运行

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiment.ps1
```

脚本仅使用 Python 标准库。结果写入 `results/`：`comparison.csv`、运行清单、真实样例、错例，以及 E0/E1/E2 的逐样本提示、响应、解析结果和检索轨迹。

## 复现边界

论文原版使用 GPT-4o-mini、Neo4j 中的 ConceptNet、Contriever 节点匹配和 spaCy 依存树。本轮课程实验尚未把完整 ConceptNet 和在线大模型推理链路接入统一脚本，因此使用透明的本地结构代理与确定性受约束推理器，不把结果表述为论文完整数值复现。E0 的 GPT-4 响应来自上游公开归档；E1/E2 不产生新的外部 API 调用或费用。

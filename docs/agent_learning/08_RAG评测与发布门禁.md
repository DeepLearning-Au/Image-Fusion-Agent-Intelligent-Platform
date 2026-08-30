# 08 RAG评测与发布门禁

## 目标

新知识库不能“构建完就上线”，必须先证明检索质量达标。

## 发布流程

`building → 结构校验 → 30题评测 → ready → 原子发布 → active`

任一指标不达标：`building → failed`，当前 active 版本不变。

## 三个门禁指标

| 指标 | 含义 | 当前门槛 |
|---|---|---:|
| Recall@5 | 前5条是否找到了正确文档 | ≥ 0.80 |
| MRR | 正确文档是否排得靠前 | ≥ 0.65 |
| nDCG@5 | 多个正确文档的排序质量 | ≥ 0.70 |

评测集至少 30 题，覆盖设备选型、产品参数、系统集成、融合、故障和维护。

## 数据存放位置

- 评测题：`storage/rag_eval_set.jsonl`
- 门槛配置：`config/rag.yaml`
- 评测报告：SQLite 表 `kb_generation_evaluations`
- 版本事件：SQLite 表 `kb_generation_events`

## 接口

- `POST /api/rag/generations/build`：构建并自动评测
- `POST /api/rag/generations/{id}/publish`：仅允许发布通过门禁的版本
- `GET /api/rag/generations/{id}/evaluation`：查看该版本的完整报告
- `POST /api/rag/generations/{id}/rollback`：故障时回滚旧版本

## 当前真实基线

- 评测题数：30
- Recall@5：1.0000
- MRR：0.8972
- nDCG@5：0.9231
- Top-1：0.8333
- 关键词覆盖率：0.9889
- 结论：通过

这组数值只代表当前评测集，应随着真实用户问题持续扩充。

# MambaDFuse Agent

面向红外、可见光和双模态成像设备选型，并支持图像融合的 Agentic RAG 项目。

## 目录导航

| 目录 | 作用 | 日常是否需要修改 |
|---|---|---|
| `agent/` | Agent及工具编排 | 增加工具时修改 |
| `backend/` | FastAPI接口和Web入口 | 增加接口时修改 |
| `catalog/` | MySQL产品库、审核与查询 | 调整产品字段时修改 |
| `rag/` | PDF入库、BM25、Milvus、RRF、BGE | 调整知识检索时修改 |
| `fusion/` | MambaDFuse推理与质量指标 | 调整融合流程时修改 |
| `task_queue/` | Redis/Celery异步任务与状态管理 | 增加耗时任务时修改 |
| `config/` | LLM、RAG、融合参数 | 日常调参入口 |
| `data/` | 原始文档、产品种子和文档图片 | 添加资料时使用 |
| `deploy/` | MySQL、Milvus和Redis Docker | 部署时使用 |
| `web/` | 企业Web前端 | 修改页面时使用 |
| `tests/` | 自动化测试 | 每次升级同步补充 |
| `docs/agent_learning/` | 分步骤学习记录 | 学习和复盘入口 |
| `storage/` | SQLite知识库、评测集和运行记录 | 运行时数据，不手工修改 |
| `third_party/MambaDFuse/` | 官方模型代码和红外可见光权重 | 通常不修改 |

## 当前检索链

```text
用户需求
→ 查询理解（模态/分辨率/帧率/接口/预算/场景）
→ MySQL硬条件门禁
→ BM25 + Milvus召回产品证据
→ 产品级RRF融合
→ BGE产品卡片重排
→ 父块证据与产品参数一起交给大模型
```

## 启动

```powershell
.\scripts\mysql.ps1 start
.\scripts\milvus.ps1 start
.\scripts\redis.ps1 start
.\scripts\task-worker.ps1 start
E:\anaconda\envs\mamba\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

- Web：`http://127.0.0.1:8001`
- API文档：`http://127.0.0.1:8001/docs`
- 产品混合检索：`POST /api/products/hybrid-search`
- 任务队列状态：`GET /api/tasks/status`
- 异步图像融合：`POST /api/tasks/fusion`

## 测试

```powershell
E:\anaconda\envs\mamba\python.exe -m pytest -q
```

部署和各阶段设计说明见 `docs/agent_learning/`。

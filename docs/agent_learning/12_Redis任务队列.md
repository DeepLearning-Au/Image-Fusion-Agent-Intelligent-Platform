# 第十二步：Redis任务队列

## 为什么加入

图像融合、知识库重建和向量同步耗时较长。它们如果直接占用 FastAPI 请求，会导致页面等待、超时，也无法排队。

## 当前结构

```text
Web / API
  → FastAPI接收请求并返回task_id
  → Redis保存任务消息和短期结果
  → gpu Worker：串行执行MambaDFuse
  → maintenance Worker：执行知识库构建、Milvus同步
  → 前端按task_id查询进度和结果
```

## 企业级设计点

- 两条队列隔离：模型推理不会阻塞知识库维护。
- GPU Worker预取数为1：避免一个Worker提前占用多个大任务。
- Redis启用AOF与RDB：Docker重启后仍可恢复队列数据。
- `noeviction`：内存不足时明确报错，不静默删除任务。
- 任务有软/硬超时，知识库网络故障会指数退避重试。
- 任务结果保留24小时；上传的临时源图在任务结束后自动删除。

## 启动顺序

```powershell
.\scripts\redis.ps1 start
.\scripts\task-worker.ps1 start
E:\anaconda\envs\mamba\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

查看状态：

```powershell
.\scripts\redis.ps1 status
.\scripts\task-worker.ps1 status
```

## 异步接口

| 接口 | 作用 |
|---|---|
| `GET /api/tasks/status` | Redis和Worker健康状态 |
| `POST /api/tasks/fusion` | 提交单次融合任务 |
| `POST /api/tasks/fusion/batch` | 提交批量融合任务 |
| `POST /api/tasks/rag/generation` | 构建并发布知识库版本 |
| `POST /api/tasks/rag/vector-sync` | 同步Milvus向量 |
| `GET /api/tasks/{task_id}` | 查询排队、进度和结果 |
| `POST /api/tasks/{task_id}/cancel` | 请求取消尚未开始的任务 |

状态流转：`queued → running → succeeded / failed`。

## 本机与生产环境差异

当前Windows学习环境使用Celery `solo` Worker，稳定且便于调试。正式部署建议把Worker放到Linux容器，API、Redis和Worker分别部署，并接入监控告警。

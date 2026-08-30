# 10 Milvus 与高级向量索引

## 当前架构

- SQLite：文档、父子块、产品数据、generation 和评测报告。
- Milvus：子块向量、向量索引和近似最近邻检索。
- 检索链：`SQL/FTS + Milvus → RRF → BGE → 父块回表`。

SQLite 中的 `chunk_embeddings` 作为嵌入缓存与迁移暂存区，线上向量召回由 Milvus 执行。

## 默认索引：HNSW

配置位置：`config/rag.yaml`

- `M=32`：每个节点的图连接数。
- `efConstruction=200`：建图质量；越大构建越慢、召回通常越高。
- `ef=128`：查询候选范围；越大查询越慢、召回通常越高。
- 距离：`COSINE`。

HNSW适合低延迟、高召回的在线问答，但需要更多内存。

## 切换 IVF_FLAT

把 `milvus.index_type` 改为 `IVF_FLAT`，重启后调用：

`POST /api/rag/vector/index/rebuild`

- `nlist=64`：向量聚类数量。
- `nprobe=16`：每次搜索的聚类数量。

IVF_FLAT更节省图索引内存，适合更大的批量数据；参数必须通过评测集调优。

## 服务与数据位置

- Docker Desktop：`F:\Docker\DockerDesktop`
- Docker WSL数据：`F:\Docker\wsl-data`
- Milvus Compose：`deploy/milvus/docker-compose.yml`
- Milvus持久化数据：`deploy/milvus/volumes`
- Milvus地址：`http://127.0.0.1:19530`
- WebUI：`http://127.0.0.1:9091/webui/`

## 日常命令

- 启动：`powershell -File scripts/milvus.ps1 start`
- 状态：`powershell -File scripts/milvus.ps1 status`
- 日志：`powershell -File scripts/milvus.ps1 logs`
- 停止：`powershell -File scripts/milvus.ps1 stop`

发布新 generation 时，只有 SQLite 分块完整、Milvus 向量数量一致、检索评测和回答评测全部通过，版本才能上线。

# 第11课：MySQL产品库迁移

## 1. 为什么只迁移产品表

系统按职责拆分存储：

```text
MySQL：产品、厂商、参数、价格、图片关系、审核候选
SQLite FTS5：文档块与BM25关键词索引
Milvus：Embedding向量与HNSW索引
文件系统：PDF、产品图片和解析结果
```

MySQL负责强关系和事务，Milvus负责向量近似检索，两者不能互相替代。

## 2. 本次迁移

从 `storage/kb.sqlite3` 迁移10张产品表，共387行：

- 2个厂商、3个正式产品；
- 69项参数定义、109条产品参数；
- 3条价格、3张产品图片、8个场景；
- 3条融合适配信息、181条待审核参数。

迁移按外键顺序在一个事务中执行，并核对逐表行数与孤儿记录。原SQLite保留，
因此可以随时回滚。

## 3. 企业级配置

- MySQL 8.4.11；
- `utf8mb4_0900_ai_ci`；
- InnoDB外键与事务；
- JSON产品扩展参数；
- 应用账号与root账号分离；
- 10连接池；
- 慢查询记录、100最大连接；
- 仅绑定本机 `127.0.0.1:3307`。

本机3306已有其他MySQL，因此Docker实例使用3307。

## 4. 常用操作

```powershell
.\scripts\mysql.ps1 status
.\scripts\mysql.ps1 stop
.\scripts\mysql.ps1 start
.\scripts\mysql.ps1 logs
```

部署文件：`deploy/mysql/docker-compose.yml`；产品建表脚本：
`deploy/mysql/init/001_product_catalog.sql`。

## 5. 应用切换

`catalog/schema.py`读取 `PRODUCT_DB_BACKEND=mysql` 和
`PRODUCT_DATABASE_URL`。显式传入测试数据库路径时仍使用SQLite，保证测试隔离；
正常运行时产品API使用MySQL。


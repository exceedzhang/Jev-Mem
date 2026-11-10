# Jev-Mem Dify Sidecar Service

把 Jev-Mem 的长期记忆(写入路径 + 图记忆平面 + 读取路径)暴露为 HTTP API,
供 Dify tool 插件(`integrations/dify/plugin`)调用。Dify 的 workflow / agent
通过插件的三个工具(`query_memory` / `ingest_memory` / `consolidate_memory`)
访问这里的 `/v1/memory/*` 端点。

## 安装

```bash
cd <Jev-Mem repo root>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r integrations/dify/service/requirements.txt
```

## 配置

服务启动时自动加载 `.env`(真实环境变量始终优先):

1. `integrations/dify/service/.env`(本服务专属,优先级最高)
2. 仓库根目录 `.env`(补齐未定义的变量)

参照 `.env.example` 创建即可,例如:

```bash
cd integrations/dify/service
cp .env.example .env    # 填入实际值
```

## 运行

### 离线 mock 模式(无需任何 API key,用于本地联调 Dify 插件)

```bash
JEV_MEM_MOCK=1 .venv/bin/python integrations/dify/service/app.py
```

### 真实模式:任意 OpenAI 兼容模型 + 本地部署的 Jev 决策模型

```bash
# --- System-Two / 抽取模型:任意 OpenAI 兼容端点(可与 Dify 共用同一 provider) ---
export OPENAI_BASE_URL=        # 如 https://api.deepseek.com/v1 或 http://localhost:11434/v1
export OPENAI_API_KEY=
export LLM_MODEL=              # 该端点上的模型名,如 deepseek-chat / qwen2.5

# --- 向量编码:走 API,不走本地模型(与 LLM 共用上面同一个端点) ---
export OPENAI_EMBEDDING_MODEL= # 该端点上的 embedding 模型名(设置即启用 API 向量)
# export OPENAI_EMBEDDING_DIMENSIONS=1536   # 可选,默认 1536,须与端点实际输出一致

# --- System-One 决策模型:本地部署的 Jev 兼容端点(TypeSafe 协议) ---
export JEV_BASE_URL=           # 如 http://127.0.0.1:8900
export JEV_API_KEY=
export JEV_MODEL=              # 如 jev-latest(默认)

.venv/bin/python integrations/dify/service/app.py
```

> 注意:embedding 模型从本地 minilm(384 维)切换到 API 模型(如 1536 维)
> 会改变向量维度,切换前请清理 `JEV_MEM_CACHE_DIR` 下各 namespace 的旧
> `vectors/` 目录,避免维度不匹配。

`JEV_*` 三个变量是 `TYPESAFE_*` 的新名字(指向本地/自托管端点更直观);
两者都设置时 `JEV_*` 优先,只设置旧的 `TYPESAFE_BASE_URL` /
`TYPESAFE_DEFAULT_MODEL` / `TYPESAFE_API_KEY` 也仍然有效。

默认监听 `0.0.0.0:8700`。

### Docker 容器化运行

镜像内同时运行两个服务(一个容器):sidecar(8700)+ MCP 服务(8701)。

```bash
# 构建(在仓库根目录执行)
docker build -f integrations/dify/service/Dockerfile -t jev-mem-sidecar .

# 运行:挂载真实 .env(只读)+ 记忆数据卷,暴露两个端口
docker run -d --name jev-mem \
  -v /path/to/repo/integrations/dify/service/.env:/app/integrations/dify/service/.env:ro \
  -v jev-mem-data:/data \
  -p 8700:8700 -p 8701:8701 jev-mem-sidecar
```

- 记忆数据落 `/data`(镜像内 `JEV_MEM_CACHE_DIR=/data` 为默认值),
  用卷持久化,容器重建不丢记忆。
- **存储后端默认 PostgreSQL**(`.env` / `.env.example` 里
  `JEV_MEM_STORAGE=postgres`):图 / 向量 / 关键词 / namespace 全部进同一个
  数据库,容器里的两个进程不再共享按 namespace 拆分的文件目录。启动前需先有
  可连接的 pgvector 实例(`JEV_MEM_PG_DSN`),连不上或维度不匹配会启动即失败;
  想回到文件布局就设 `JEV_MEM_STORAGE=file`。
- 容器内 `localhost` 是容器自身:访问宿主机上的模型服务须用
  `http://host.docker.internal:8000`(或把模型服务放进同一 docker
  network 后用服务名互访)。
- `.env` 不会被打进镜像(`.dockerignore` + 选择性 COPY 双重保证),
  镜像可安全分发。
- MCP 服务(`entrypoint.py` 启动)默认把 `JEV_MEM_MCP_BASE_URL` 设为
  `http://127.0.0.1:8700`、`JEV_MEM_MCP_API_KEY` 复用 `JEV_MEM_API_KEY`,
  同一份 `.env` 同时服务两个进程;`.env` 中显式写的 `JEV_MEM_MCP_*`
  优先。对外只暴露 8701:智能体连 `http://<host>:8701/sse`。
- 两个进程由同一 entrypoint 托管:任一退出则另一个被一起停掉,
  `docker stop` 同时优雅关闭。
- 只有 sidecar 进程写 `/data` 下的记忆文件,MCP 服务是纯 HTTP 瘦客户端,
  从不直接读写磁盘——所以多个智能体同时使用 8701 是安全的。
  但注意:对同一个 `/data` 卷运行*第二个 sidecar 实例*在 file 后端下是
  不安全的(每个进程各自持有内存索引,并发写入是 last-writer-wins)。
  多实例部署请改用 `JEV_MEM_STORAGE=postgres`(见 docs/postgresql_storage.md)。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `JEV_MEM_HOME` | 仓库根(自动推导) | Jev-Mem 仓库根目录 |
| `JEV_MEM_PROFILE` | `$JEV_MEM_HOME/config/jev_mem.json` | 论文默认配置 profile |
| `JEV_MEM_CACHE_DIR` | `$JEV_MEM_HOME/cache/dify` | 记忆持久化根目录(每个 namespace 一个子目录) |
| `JEV_MEM_MOCK` | 未设置 | `1` = 离线 mock(Jev 与向量编码全部 mock) |
| `JEV_MEM_API_KEY` | 未设置 | 设置后 `/v1/*` 需要 `Authorization: Bearer <key>` |
| `LLM_MODEL` | `gpt-4o-mini` | System-Two 答案合成 / MAGMA 路径抽取模型 |
| `OPENAI_BASE_URL` | 官方端点 | System-Two 模型所在的 OpenAI 兼容端点(DeepSeek/Moonshot/Ollama/vLLM/one-api 等) |
| `OPENAI_API_KEY` | 未设置 | 该端点的 API key(mock 模式可省略) |
| `JEV_BASE_URL` | `https://api.typesafe.ai` | System-One 决策模型的 Jev 兼容端点(本地部署时填本机地址);`TYPESAFE_BASE_URL` 为旧名,`JEV_*` 优先 |
| `JEV_API_KEY` | 未设置 | 该端点的 API key;`TYPESAFE_API_KEY` 为旧名,`JEV_*` 优先 |
| `JEV_MODEL` | `jev-latest` | 该端点上的决策模型名;`TYPESAFE_DEFAULT_MODEL` 为旧名,`JEV_*` 优先 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI 兼容端点上的 embedding 模型名;设置后向量编码走 API(共用 `OPENAI_BASE_URL`/`OPENAI_API_KEY`),不再用本地 minilm |
| `OPENAI_EMBEDDING_DIMENSIONS` | 自动 | 未设置时自动探测端点的原生维度且不发送 `dimensions` 参数(vLLM/Qwen 模型会拒绝该参数);仅 Matryoshka 模型(如 text-embedding-3-small)需显式指定,须与端点实际输出一致 |
| `JEV_MEM_EMBEDDING_MODEL` | 自动 | 显式指定向量后端:`openai`(API)或 `minilm`(本地 sentence-transformers);未设置时 = 设置了 `OPENAI_EMBEDDING_MODEL` 则走 API,否则本地 minilm |
| `JEV_MEM_PORT` | `8700` | 监听端口 |
| `JEV_MEM_MAX_NAMESPACES` | `8` | 同时驻留的 namespace 数(LRU 驱逐前自动保存) |
| `JEV_MEM_STORAGE` | `file`(库默认;容器 `.env` 默认 `postgres`) | 存储后端:`file`(文件布局)或 `postgres`(见 `docs/postgresql_storage.md`) |
| `JEV_MEM_PG_DSN` | 未设置 | `JEV_MEM_STORAGE=postgres` 时必填,未设置则启动即失败 |
| `JEV_MEM_PG_VECTOR_DIM` | `2560` | pgvector 列维度;必须与 embedding 模型维度一致,否则启动即失败 |
| `JEV_MEM_GRAPH_STORAGE` | `table` | 图存储实现;`age` 为保留值,暂未实现 |

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查(无鉴权) |
| POST | `/v1/memory/ingest` | 写入一条观察 `{text, namespace?, timestamp?(ISO 8601), metadata?}` → `{stored, memory_id, node}`;准入拒绝时 `stored=false` |
| POST | `/v1/memory/ingest/batch` | 批量写入 `{items: [str \| {content\|text, timestamp?, metadata?}], namespace?}` → `{admitted, rejected}` |
| POST | `/v1/memory/query` | 检索 `{question, namespace?, top_k?=15, answer?=true}` → `{answer, evidence, trace}`;`trace` 含 Jev 决策全量审计(graph_needs / graph_budgets / stopping_decision / nodes_visited / edges_examined / jev_calls / latency 等) |
| POST | `/v1/memory/consolidate` | 对指定记忆做巩固 `{memory_id, namespace?}` → `{decisions}`(非破坏性) |
| POST | `/v1/memory/save` | 强制持久化 `{namespace?}` |
| GET | `/v1/memory/stats?namespace=default` | 节点/边统计(按边类型分组) |

## 多租户(namespace)

Dify 侧插件的 `namespace` 凭据会透传到每个请求;服务端为每个 namespace
维护独立的 `JevMemSystem`(独立的图、向量索引、审计日志),LRU 上限
`JEV_MEM_MAX_NAMESPACES`,驱逐前自动 `save_memory()`。

## 冒烟测试(mock 模式)

```bash
curl -s localhost:8700/health
curl -s -X POST localhost:8700/v1/memory/ingest \
  -H 'content-type: application/json' \
  -d '{"text": "Alice prefers concise explanations about the Jev-Mem project.", "metadata": {"source": "smoke"}}'
curl -s -X POST localhost:8700/v1/memory/query \
  -H 'content-type: application/json' \
  -d '{"question": "What does Alice prefer?", "top_k": 5}'
curl -s "localhost:8700/v1/memory/stats?namespace=default"
```

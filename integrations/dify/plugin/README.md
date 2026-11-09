# Jev-Mem Dify Tool Plugin

把 [Jev-Mem](https://arxiv.org/abs/2609.23986) 长期记忆接入 Dify 的 tool 插件。
插件本身是薄客户端,真正的记忆系统运行在 Jev-Mem 仓库的
[sidecar 服务](../service/README.md) 中。

## 工具

| 工具 | 作用 |
|---|---|
| `query_memory` | 有界多跳图检索 + System-Two 答案合成,返回答案、证据与检索 trace(stopping_decision / jev_calls / nodes_visited 等) |
| `ingest_memory` | 写入一条观察:Jev typing/准入 → 候选检索 → 关系判断 → 事务插入,返回 `memory_id` |
| `consolidate_memory` | 对指定 `memory_id` 执行非破坏性巩固(冗余/矛盾/过时判断 + 子类型边) |

## 凭据

| 字段 | 必填 | 说明 |
|---|---|---|
| `base_url` | 是 | sidecar 服务地址,如 `http://localhost:8700` |
| `api_key` | 否 | 服务端设置 `JEV_MEM_API_KEY` 时填写 |
| `namespace` | 否 | 记忆命名空间,按 Dify 应用/租户隔离记忆库 |

## 前置:启动 sidecar 服务

```bash
cd <Jev-Mem repo root>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r integrations/dify/service/requirements.txt
JEV_MEM_MOCK=1 .venv/bin/python integrations/dify/service/app.py   # 离线联调
```

## 本地调试

1. Dify 控制台 → 插件页 → 调试图标,复制 **Host Address** 与 **API Key**;
2. `cp .env.example .env`,填入 `REMOTE_INSTALL_URL` 与 `REMOTE_INSTALL_KEY`;
3. `pip install -r requirements.txt && python -m main`;
4. Dify 插件页会出现处于 *debugging* 状态的 Jev-Mem 插件,配置凭据(服务可达即通过校验)。

## 打包安装

需先安装 CLI(`brew install dify`,或从 langgenius/dify-cli releases 下载)。
在**插件目录的上级目录**执行:

```bash
cd integrations/dify            # 插件项目须作为子目录
dify plugin package ./plugin -o ./jev_mem.difypkg
```

把 `.difypkg` 拖入 Dify 插件页安装(查看权限 → 安装);或在 agent 工具列表中添加
`query_memory` / `ingest_memory` / `consolidate_memory`。

**注意**:
- `manifest.yaml` 与 `provider/*.yaml` 的 `author`(当前为 `exceedzhang`)必须与
  **安装者的 GitHub 用户名**一致,否则报 `plugin_unique_identifier not valid`;
  换人安装前改 5 处 author 再重新打包。
- 自部署 Dify 默认强制签名校验,未签名包会被拒:开发/内网可在 docker `.env`
  加 `FORCE_VERIFYING_SIGNATURE=false` 后重启;正式分发请按官方文档签名。

## 使用建议

- 在 agent 的系统提示中约定:用户透露持久偏好/事实/决定 → 调 `ingest_memory`;
  问题可能依赖历史上下文 → 先调 `query_memory`。
- 批量灌入历史对话后,可对关键记忆调用 `consolidate_memory`。
- 不同 Dify 应用建议用不同 `namespace`,避免记忆串号。

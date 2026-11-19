# 后端接入：OpenAI-compatible 与自定义 Adapter

> 本文最初随 v0.2.16 发布，介绍如何接入未知的生产 API。

## 定位与边界

后端层只负责传输和协议适配，不涉及模型选型或判决算法：

- 引入 `app/backends.py`：`LLMBackend` Protocol、统一 `BackendResponse`、默认 `OpenAIChatBackend`、可插拔非 OpenAI Adapter。
- 主模型与复核模型可用不同 Base URL、密钥和 Adapter。
- JSON 输出分为 `legacy`、`on`、`off`、`auto`：不管哪一种，**结果都必须通过原有 Pydantic 校验和原文引用核验**。
- 对没有 `GET /models` 的 API 可选择 `LM_MODEL_DISCOVERY=optional` 或 `off`。`off` 跳过预检，**绝不代表已经连接成功**。
- 支持的 Pipeline：`extraction`（默认）和 `direct_hardened`（实验）。其余历史 direct 模式已在 v0.3.0 移除。
- 本地 Demo 推荐 **Qwen `qwen/qwen3.5-9b` + Gemma `google/gemma-4-12b-qat`**。这个模型组合仅用于开发；未知的生产模型需另行兼容性测试。

**不保证**任何未知云端模型都能产生质量合格的答案，也不保证非 OpenAI HTTP API 无需开发适配器即可接入；离线测试不调用真实 LM Studio 或云 API，不能据此报告真实准确率。

## 最简单：保持原 LM Studio 配置

Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env  # 仅在首次安装时运行；已有 .env 不要覆盖
```

在 `.env` 里：

```dotenv
LM_BASE_URL=http://127.0.0.1:1234/v1
LM_MODEL=qwen/qwen3.5-9b
LM_VERIFIER_MODEL=google/gemma-4-12b-qat
LM_JSON_SCHEMA=true
LM_SCHEMA_MODE=legacy
LM_MODEL_DISCOVERY=required
LM_PIPELINE=extraction
```

### 一条命令：真实 LM Studio 的 11 条回归

确保两个模型已加载且 LM Studio API Server 已启动：

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --model qwen/qwen3.5-9b \
  --verifier-model google/gemma-4-12b-qat \
  --pipeline direct_hardened \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_direct_hardened_11.json
```

这是**调用真实本地模型的命令**；仓库附带的离线 `pytest` 不会替代它。v0.2.16 用 Qwen + Gemma 跑出的结果见 `examples/reference_v0216_direct_hardened_11.json`（11 条中 9 条正确、2 条 REVIEW_REQUIRED、0 误判）。`--pipeline` 的显式选择不改变 `.env` 中默认的 `extraction`。

## 部署场景 A：OpenAI-compatible、支持 JSON Schema

```dotenv
LM_BACKEND=openai
LM_BASE_URL=https://provider.example/v1
LM_API_KEY=PRIMARY_SECRET
LM_MODEL=actual-primary-id
LM_VERIFIER_MODEL=actual-reviewer-id
LM_SCHEMA_MODE=on
LM_MODEL_DISCOVERY=optional
```

`on` 表示强制原生 Schema。API 明确拒绝时返回执行级错误；**不会将 API 失败记作语义 FAIL**。

## 场景 B：OpenAI-compatible，但没有原生 Structured Output

```dotenv
LM_BACKEND=openai
LM_BASE_URL=https://provider.example/v1
LM_API_KEY=PRIMARY_SECRET
LM_MODEL=actual-primary-id
LM_VERIFIER_MODEL=actual-reviewer-id
LM_SCHEMA_MODE=off
LM_MODEL_DISCOVERY=off
```

Prompt 要求 JSON，主机用 `parse_json_object()` + Pydantic + 原文引用检查；格式或引用始终不合格时返回 `ERROR`，不伪造 PASS/FAIL。

### 能力未知：`LM_SCHEMA_MODE=auto`

`auto` 会对每个实际模型优先使用原生 Schema；**仅**在 HTTP 400/415/422 且响应明确称 `response_format/json_schema` 不受支持时，取消 `response_format` 并进行一次 Prompt-only 重试。后续同一模型使用缓存的能力结果。认证失败、模型不存在、限流、服务端错误及普通无效 JSON **不会触发此降级**。对特定第三方 API，推荐显式设置 `on`/`off`，而不是依赖返回错误文本猜测能力。

`legacy` 完全沿用旧版 `LM_JSON_SCHEMA` 约定，以免升级时隐式改变行为。

## 场景 C：主模型与复核模型是两个不同的 API

```dotenv
LM_BACKEND=openai
LM_BASE_URL=https://primary.example/v1
LM_API_KEY=PRIMARY_SECRET
LM_MODEL=primary-model-id

LM_VERIFIER_BACKEND=openai
LM_VERIFIER_BASE_URL=https://secondary.example/v1
LM_VERIFIER_API_KEY=SECONDARY_SECRET
LM_VERIFIER_MODEL=reviewer-model-id

LM_SCHEMA_MODE=auto
LM_MODEL_DISCOVERY=off
```

`LM_VERIFIER_*` 连接配置留空时复用主服务。两个角色的模型 ID 必须不同（双模型 Pipeline 的要求）。不同服务的模型发现按各自适配器执行；`off` 时都不调用 `/models`。

**当前限制：** Schema 模式策略由一个 `LM_SCHEMA_MODE` 控制，但 `auto` 的能力缓存按具体模型隔离；两个后端需要完全不同请求字段时为对应角色配置不同 Adapter。

## 场景 D：非 OpenAI-compatible API

预留 Adapter 接口；提供可执行、但接口**纯属示例**的 `examples/custom_backend_example.py`。真实部署必须按服务商文档改写请求和响应转换。

```dotenv
LM_BACKEND=examples.custom_backend_example:ExampleNonOpenAIBackend
LM_BASE_URL=https://demo-provider.example
LM_API_KEY=YOUR_SECRET
LM_SCHEMA_MODE=off
LM_MODEL_DISCOVERY=off
```

自定义适配器必须以 `__init__(config: BackendConfig, http_client: httpx.Client)` 构造，并实现：

```python
class MyBackend:
    def __init__(self, config, http_client): ...
    def models(self) -> list[str] | None: ...   # None = 不支持查询模型
    def complete(self, payload: dict) -> BackendResponse: ...
```

输入 `payload` 含 `model`, `messages`, `temperature`, `max_tokens`, `stream=False`，在 `on`/`auto` 下还可含 `response_format`；适配器应翻译并返回 `BackendResponse(content=..., finish_reason=..., token usage...)`。**切勿**在 Adapter 中做语义裁决、伪造 `PASS` 或跳过 `ChatClient.structured`。自定义 Adapter 由本地 Python 配置显式导入，**只允许导入可信代码**。

## 场景 E：DeepSeek 等第三方 OpenAI-compatible API（推荐的生产接法）

这是项目最终面向的部署方式。这类服务一般支持 `/chat/completions` 和 JSON 模式（`response_format: {"type": "json_object"}`），不一定支持完整的 JSON Schema。

```bash
LM_BACKEND=openai
LM_BASE_URL=https://<服务商的 OpenAI 兼容地址>/v1
LM_API_KEY=<主模型的 key>
LM_MODEL=<主模型 ID>
LM_VERIFIER_MODEL=<复核模型 ID，必须与主模型不同>
# 复核模型在另一家服务商时再设置：
# LM_VERIFIER_BASE_URL=...  LM_VERIFIER_API_KEY=...
LM_SCHEMA_MODE=json_object     # 服务商支持 JSON 模式时；不支持就用 off
LM_MODEL_DISCOVERY=optional    # /models 不可用时不报错
LM_TIMEOUT_SECONDS=180
LM_MAX_TOKENS=4096             # 推理模型思考较长时适当调大
LM_MAX_CONCURRENT_CHECKS=4     # 按服务商的并发限额设置
```

要点：

- **模型 ID、是否支持 JSON 模式、并发限额，以服务商的最新文档为准**，这里不写死。
- **`dual` 模式要求主模型和复核模型的 ID 不同。** 可以用同一家的两个模型（如一个对话模型加一个推理模型），也可以把复核放在另一家服务商（`LM_VERIFIER_*`），后者独立性更好。
- **三种结构化输出模式**：`on` 要求服务商支持 JSON Schema；`json_object` 只保证输出是合法 JSON，结构写在提示词里；`off` 只靠提示词。三种模式下，结果都要经过 Pydantic 和原文 ID 校验，不合格会带着错误信息重试，最多 3 次。服务商拒绝 `json_object` 时，改用 `off`。
- **推理模型**：思考内容放在 `reasoning_content` 里的，客户端只读取正式的 `content`。截断时会用 `LM_MAX_TOKENS` 做一次恢复。
- **每次检查的调用次数**：`dual` 模式下，正常 2～4 次，有分歧时再加 1～2 次（见 [architecture.md](architecture.md) 5.12 节）。

### 接入验收

换用新服务商后，按顺序跑下面三步。每一步都没有误判、没有 ERROR，再进行下一步：

```bash
python -m examples.evaluate --cases examples/v2_regression_11.json --limit 3 --out eval_smoke.json
python -m examples.evaluate --cases examples/validation_v030_gen_dev_60.json --workers 4 --out eval_dev60.json
python -m examples.evaluate --cases examples/validation_v030_blind_99.json --workers 4 --out eval_blind99.json
```

最后一步是盲测集，只看汇总指标，并把结果记入 [docs/eval_protocol/README.md](eval_protocol/README.md) 的运行记录。参考基线：gpt-6-luna 在盲测集上 99/99，开发集上 60/60。

## Codex CLI backend (evaluation only)

本地安装 Codex CLI 并使用 ChatGPT 账号登录后，可以用此后端运行评测。主模型和复核模型的 ID 必须不同，`@effort` 后缀允许同一模型用不同推理强度参与评测。

```bash
LM_BACKEND=examples.codex_backend:CodexExecBackend LM_MODEL_DISCOVERY=off \
LM_MODEL=gpt-6-luna@medium LM_VERIFIER_MODEL=gpt-6-luna@high LM_TIMEOUT_SECONDS=600 \
python -m examples.evaluate --cases examples/v2_regression_11.json --workers 4 --out eval_x.json
```

`CODEX_BIN` 可指定 CLI 路径。推理强度可取 `low`、`medium`、`high` 或 `xhigh`。省略后缀时使用 `CODEX_REASONING_EFFORT`，默认 `medium`。此后端仅供评测使用。

## 失败/可靠性约定

| 场景 | 输出行为 |
|---|---|
| HTTP 401 / 403 / 429 / 5xx | 执行 ERROR，绝不自动 PASS/FAIL |
| 明确不支持 JSON Schema，`auto` | 一次 Prompt-only 降级，原 Schema 校验仍保留 |
| 没有 `/models`，`optional` | 略过不可用的发现，首次请求仍可能失败 |
| 没有 `/models`，`off` | 不进行远程预检；状态页显示“未验证” |
| 错误 JSON、缺字段、无效原文证据 | 既有有限次数重试；最终仍不通过则 ERROR |
| 单个模型可用 | 使用 `--mode primary`；独立复核将关闭，质量保障较弱 |

所有 Trace 仅保存耗时、token 用量、状态、schema 模式等元数据，适配器不得将 API Key、完整提示词或供应商原始响应明文写入诊断报告。`response_source=reasoning_json` 仅兼容旧 LM Studio 在原生 Schema 模式下把**完整 JSON**放进 reasoning 字段的特殊情况；不能把自由形式的思考过程当作答案。

## 测试说明

离线测试只验证模拟的 HTTP 与业务逻辑，不代表真实模型的判决准确率。真实结果请用上方的 11 条端到端命令跑 LM Studio 回归，再查看 `false_pass`、`false_fail`、`review_required`、`errors_by_stage`、`schema_or_grounding_invalid` 与耗时。

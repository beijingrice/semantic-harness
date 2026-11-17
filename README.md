# Bilingual Keyword Checker

中英文学术规范的关键事实一致性核查工具：判断一段英文要求和对应的中文要求是否表达了同样的关键事实（数值边界、列举项、例外条件、比较方向等），并给出可追溯到原文段落的证据。

提供 Web UI、JSON API（FastAPI）和命令行三种用法；模型通过 OpenAI-compatible API 接入（本地开发用 LM Studio），也可以接自定义后端 Adapter。Prompt 中没有任何模型品牌或服务商专用指令。

> **当前版本 v0.3.0。** 各版本的改动见 [CHANGELOG.md](CHANGELOG.md)，历史版本的详细说明在 [docs/history/](docs/history/)。

## 判决结果的含义

| 结果 | 含义 |
|---|---|
| `PASS` | 关键事实一致 |
| `FAIL` | 有缺失或冲突，并附原文段落证据 |
| `REVIEW_REQUIRED` | 模型意见分歧且无法安全自动裁定，需要人工判断；**不是** FAIL，也不能算作 PASS |
| `ERROR` | API、模型响应、Schema 或校验流程执行失败（API 返回 502，CLI 退出码 2）；**不是**业务判定 |

CLI 退出码：PASS = 0，FAIL = 1，ERROR / 配置错误 = 2，REVIEW_REQUIRED = 3。

## 快速开始

需要 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env      # 首次安装时运行；已有 .env 不要覆盖
python -m pytest -q       # 离线测试，不会调用模型
```

在 LM Studio 中加载两个**不同的**模型并启动 API Server（常用端口 1234），用下面的命令确认模型 ID，再填到 `.env` 的 `LM_MODEL` 和 `LM_VERIFIER_MODEL`：

```bash
curl -s http://127.0.0.1:1234/v1/models | python3 -m json.tool
```

启动 Web UI（API 文档在 `/docs`）：

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8005
```

命令行：

```bash
bilingual-check --en 'The abstract must include purpose, methods, and results.' \
  --zh '摘要须介绍研究目的和研究方法。' --mode dual
```

`--mode primary` 只用主模型，`--mode dual`（默认）会再加上独立复核和必要的仲裁。

## 两种 Pipeline

用 `LM_PIPELINE` 选择：

- **`extraction`（默认）**：程序按空行把原文切成段落并分配不可变 ID → 模型分别从中、英文中提取事实（`claim` + 段落 ID，不生成证据文本）→ 匹配 → 独立模型只看原文复核 → 有分歧时仲裁。数值换算（如 `6 weeks = 42 days`）由程序做形式化校验。短句满足严格条件时会走 Fast Path，跳过生成式提取。
- **`direct_hardened`（实验）**：两个模型分别直接比较完整段落，不做提取；程序检测到数值边界、包含/排除、例外范围、比较方向等风险时，再提出固定问题，由模型选择原文片段，程序自己核对。

v0.2.8 用 `extraction` 在真实模型上跑过：历史回归 11/11，36 条 holdout 34/36。v0.2.16 用 `direct_hardened` 跑 11 条回归：9 条正确、2 条 REVIEW_REQUIRED、0 误判（[结果文件](examples/reference_v0216_direct_hardened_11.json)）。这些都是开发集上的结果，不代表在未知样本上的准确率。

`direct`、`direct_grounded`、`direct_verified`、`direct_questions`、`direct_relations` 这五种早期实验模式已在 v0.3.0 移除，需要时可以从 git tag `v0.2.16` 找回。

## 主要配置

完整注释见 [.env.example](.env.example)。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LM_BASE_URL` | `http://127.0.0.1:1234/v1` | 主模型 API 地址 |
| `LM_MODEL` / `LM_VERIFIER_MODEL` | — | 主模型 / 复核模型 ID，dual 模式下两者必须不同 |
| `LM_API_KEY` | `lm-studio` | API 密钥 |
| `LM_PIPELINE` | `extraction` | `extraction` 或 `direct_hardened` |
| `LM_SCHEMA_MODE` | `legacy` | `legacy` 沿用 `LM_JSON_SCHEMA`；`on` / `off` / `auto` 见 [后端接入](docs/backends.md) |
| `LM_MODEL_DISCOVERY` | `required` | `required` / `optional` / `off`：是否调用 `GET /models` |
| `LM_MAX_TOKENS` | `3072` | 单次生成上限 |
| `LM_TIMEOUT_SECONDS` | `180` | 单次请求超时 |
| `LM_VERIFIER_BASE_URL` 等 | 空 | 复核模型接另一个 API 时填写，见 [后端接入](docs/backends.md) |

配置在每次创建 `Settings()` 时从环境变量读取，填错（比如 `LM_MAX_TOKENS=abc`）会给出指明变量名的配置错误，而不是让服务直接崩溃。

`LM_EVIDENCE_MODE=quote`、`LM_EXTRACTION_PROMPT=legacy|proposition`、`LM_COVERAGE_AUDIT`、`LM_ATOMIC_REFINEMENT` 是复现历史 A/B 实验用的开关，日常使用保持默认即可。

接入云端 API、没有原生 Structured Output 的 API、主次模型分别接不同服务、或者非 OpenAI 协议的服务，见 **[docs/backends.md](docs/backends.md)**。

## 评测

```bash
python -m examples.evaluate --cases examples/v2_regression_11.json \
  --mode dual --stop-after-errors 1 --out eval_regression_11.json
```

评测会分别统计 PASS / FAIL / REVIEW_REQUIRED / ERROR、False PASS、False FAIL、端到端正确率、按类别的结果、耗时和 token 用量。加 `--pipeline direct_hardened` 可以评测另一种 Pipeline；加 `--audit-csv` 可以导出人工核查表。

`examples/` 中的主要数据集：

| 数据集 | 条数 | 用途 |
|---|---:|---|
| `v2_regression_11.json` | 11 | 已知薄弱案例回归，**每次改动后先跑这个** |
| `validation_v028_holdout_36.json` | 36 | v0.2.8 冻结的泛化验证集（附 sha256） |
| `validation_v2_120.json` | 120 | 主要复杂回归集，覆盖 8 类语言与条款结构 |
| `validation_v027_paragraph_16.json` | 16 | 段落切分、跨段落、弯引号等结构问题 |
| `validation_v2_pilot_20.json` | 20 | 120 条集的快速预检 |
| `v2_false_pass_4.json` | 4 | 专门观察关键差异被错误放行的情况 |

注意：11 条回归集和 60 条旧开发集都被反复用于调试，不是盲测集。

## 项目结构

```
app/
  api.py, cli.py          Web/JSON API 与命令行入口
  config.py               配置（环境变量 → Settings）
  backends.py, llm.py     后端 Adapter；结构化生成、校验与重试
  models.py, prompts.py   Pydantic 输出协议；模型无关的 Prompt
  paragraphs.py           原文段落与不可变 ID / offset
  pipeline.py             Checker 入口与 Pipeline 分发
  paragraph_pipeline.py   extraction 默认链路
  extraction_fast_path.py, dispute_resolution.py, validators.py
  direct_pipeline.py, relation_direct.py, source_risks.py   direct_hardened 链路
  static/                 Web UI
examples/                 评测脚本与数据集
tests/                    离线测试（Mock HTTP，不调用真实模型）
docs/                     后端接入说明与历史版本文档
```

## 开发

```bash
python -m pytest -q        # 离线测试
ruff check .               # Lint
```

CI（GitHub Actions）会在 Python 3.10 和 3.13 上跑 ruff 和 pytest，配置在 [.github/workflows/ci.yml](.github/workflows/ci.yml)。测试通过 `tests/conftest.py` 与本地 `.env` 隔离。

## 已知边界

- 段落 ID 只能证明证据片段真实存在于原文，不能证明 `claim` 的含义正确或覆盖了所有关键条件。
- 纯文本以**空行**分段；DOCX、PDF、HTML 的逻辑段落需要在上游解析时恢复。
- 数值校验只处理明确的单位换算和比较边界，不是通用的逻辑求解器。
- 离线测试保证结构、协议和分支安全，不代表真实模型的准确率或耗时。

## 许可证

见 [LICENSE](LICENSE)。

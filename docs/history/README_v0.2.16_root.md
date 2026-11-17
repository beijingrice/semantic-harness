## v0.2.16 — Backend Compatibility（当前版本）

重点是让 Harness 可接**未知生产 API**，而不是继续比较本地模型。
支持 OpenAI-compatible / 可插拔非 OpenAI Adapter、主次模型分离 API 地址与密钥、可选 `/models` 发现，以及可控的 JSON Schema 降级。
**开发示例：Qwen3.5-9B + Gemma 4 12B QAT；默认仍为 `extraction`。**
完整安装、API 配置、唯一真实 11 条回归命令与边界说明见 [README_V0216_BACKENDS.md](README_V0216_BACKENDS.md)。
历史版本说明如下，原始判决 Pipeline 未改动。

## v0.2.15 research preview

This release adds opt-in `direct_hardened`, retaining the original complete paragraphs,
independent Qwen and Ministral judgments, and the v0.2.14 source relation checks.
Hardening is deliberately surgical: mathematical Unicode operand normalization,
explicit entity-specific exception citations, enumerated evidence IDs in the JSON
Schema, and a narrowly bounded source-only recovery of one comparison direction
when a model produces an invalid citation ID. No extra LM requests or sentence
alignment are introduced. See [README_V0215_HARDENED.md](README_V0215_HARDENED.md).

The `extraction` default and ALL previous experimental modes remain available.
The 11 repeated development cases are not a new independent validation set.

## v0.2.14 research preview

New experimental `direct_relations` pipeline: full-paragraph dual-model comparison + fixed risk-specific questions + source-grounded relation checks. The host now separates **same event/referent** from **same decision/value**; `same_event=no` is not an early veto when the host can independently establish that the same entity and condition have opposite properties. A deliberately narrow scalar-equality certificate also protects semantically identical duration restatements. This is not a claim of live LM accuracy. See **[README_V0214_RELATIONS.md](README_V0214_RELATIONS.md)**. Previous `extraction`, `direct`, `direct_grounded`, `direct_verified` and `direct_questions` modes remain available; `extraction` stays default.

## v0.2.13 research preview

New experimental `direct_questions` pipeline: whole-paragraph Qwen + Ministral judgments, followed by program-assigned fixed, source-checkable questions for numeric bounds, inclusion, exceptions, and comparison direction. Existing modes remain available; `extraction` is still the default. See **[README_V0213_QUESTIONS.md](README_V0213_QUESTIONS.md)** for precise guarantees, limits and the single regression command.

**v0.2.12 experimental direct_verified:** See `README_V0212_VERIFIED.md` for installation, source-anchored proposition verification, safety boundaries and evaluation. Default `extraction` remains unchanged.

# v0.2.11: see README_V0211_GROUNDED.md for Paragraph-Direct source-grounded evaluation.

> **v0.2.10 Direct architecture preview:** existing Extraction remains default. > Run `--pipeline direct` to compare whole original paragraphs using two models. > See [README_V0210_DIRECT.md](README_V0210_DIRECT.md) for one-command validation and limitations.

# v0.2.9.2 — Bounded Clause Anchors & Dispute Safety（当前版本）

**本次目标：** 修复 v0.2.9.1 真实回归中 `V2-ENU-13` 的同义提取膨胀
（中文提取 29 秒）与 Qwen 错误仲裁（本应 PASS，却输出 FAIL）。
新增**保守双子句原文锚点**，符合严格结构条件时在生成前跳过两次 LLM Extraction；
保留上一版的单句 Fast Path、Matching、独立复核与 Numeric Guard。
对“初判 FAIL／独立复核 PASS／仲裁 FAIL”这一证据冲突组合返回人工复核，
而不武断采用 Qwen 仲裁的 FAIL。完整说明见 [README_V0292.md](README_V0292.md)，
上一版说明仍保留在 [README_V0291.md](README_V0291.md)。

所有数据和性能声明以实际模型输出为准；离线测试不代表真实模型的耗时与准确率。

---

# v0.2.9 — Numeric Semantics & Bounded Extraction（历史版本）

详见 [README_V029.md](README_V029.md)，包含改动说明、验证边界、Phase 1 与 Exec 命令。

本版在 v0.2.8 基础上新增通用单位和比较边界处理、原文层面的保守数值保护、短句提取去膨胀；保留全部旧版文档与验证集。v0.2.8 既有 11/11 与 34/36 是历史评测，不代表 v0.2.9 已通过真实模型评测。

# v0.2.8 历史文档

新增受限生成恢复、请求级 token/耗时追踪、源引用结构审计、人工审核 CSV、36 条冻结的独立泛化验证集。详见 [README_V028.md](README_V028.md)。

历史说明更新：**v0.2.8 已在用户本机真实评测**，历史回归 11/11、当时全新 36 条测试 34/36；这些结果不代表 v0.2.9 或未知样本的准确率。

# v0.2.7.5 — 1:N Numeric Certificate Recovery & Truncation Diagnostics

See [README_HOTFIX_0275.md](README_HOTFIX_0275.md) for this release's changes and low-cost validation commands. The defaults still use the v0.2.7.4 paragraph-first dual-model pipeline. Offline pytest never calls LM Studio. The bundled evaluation datasets are diagnostic/dev tests, not an independent holdout.

**Historical release: v0.2.7.4**. See [v0.2.7.4 disputed-review resolution](README_HOTFIX_0274.md), [LM Studio Structured Output compatibility](README_HOTFIX_0273.md), and [numeric guard notes](README_RELEASE_0272.md).

> **v0.2.7.1 hotfix:** Paragraph retry instructions and source ID examples corrected. See [README_HOTFIX_0271.md](README_HOTFIX_0271.md).

# Bilingual Keyword Checker v0.2.7 — Paragraph Grounding

中英文论文规范关键事实一致性核查 Demo；FastAPI + Web UI + CLI + LM Studio（OpenAI-compatible HTTP API）。模型可以在 `.env` 或 Web UI 中选择，**Prompt 中无模型品牌、型号或服务商专用指令**。

**当前默认：Paragraph-first Source Grounding**。Python 在原文中按**空行**识别段落并建立不可变 ID / offset；模型生成 `claim + paragraph_ids`，**不生成 `evidence` 字符串**；Python 从原文恢复完整证据段落。这样处理中文、英文、Unicode 弯引号、英文缩写、混排空格和句末标点，无需为不同字符添加越来越多的引文修复特例。

## 一、v0.2.7 改了什么

- **`app/paragraphs.py`**：纯文本按空行切段；单次换行留在段内。保留 Python 原文 `[start,end)` 位置，保证 `original[start:end] == paragraph.text`，不改写原文、不强制分句。不将中英文段落按位置一一绑定。
- **Extraction**：默认 coarse Prompt 从单一语言的原文与段落列表提取事实。一个段落可提取多个事实，一个事实可引用多个段落；完整并列列表允许成为单个 coarse unit。模型只能引用既有 ID，不能凭空生成 offset 或引文。
- **Review / Arbitration**：同样只生成已有 Paragraph ID；Review 只看中英文原文及其段落，不看主模型事实或判断。最终 JSON API 保留向前兼容的 `evidence` 字符串以支持现有 Web UI，增加 `paragraph_ids` 和 `english_paragraphs` / `chinese_paragraphs`。
- **保守仲裁**：初判 FAIL 要改成 PASS 时，即使已提供合法双语 Paragraph ID，也要经过独立模型的语义支持校验；如初判 PASS、独立复核引用了具体冲突，而仲裁仍建议 PASS，则运行一次定向异议核查，只有所有异议均被驳回、并且有来源复算成功的形式化数值等价证书时才允许 PASS；否则返回 **REVIEW_REQUIRED**，不再误报为执行 ERROR。ID 的真实性不等于语义支持证明。
- **模型解耦**：配置决定主模型、复核模型与后端；Prompt 不引用具体品牌或模型。Web UI 不写死备选模型名称；结果摘要显示实际 `verifier_model`。
- **A/B 兼容**：`LM_EVIDENCE_MODE=quote` 仍然使用 v0.2.6 的连续字符串引用和旧流程，供复现历史评测。默认 `paragraph` 路径不调用 `resolve_evidence`；旧的 CJK/ASCII 空格特判仅存于历史 quote 模式。

**注意边界**：Paragraph ID 仅证明一个片段真实存在，无法证明 `claim` 的含义正确或所有关键条件都被覆盖。较长段落导致 evidence 更粗，也可能增加 Token 使用。数值安全网仍只是轻量启发式，并不是单位换算器/逻辑求解器。v0.2.7.2 起能安全处理 `6 weeks = 42 days` 等明确的单位换算，但这只证明一条形式化数量约束，不能证明其他语义一致。这次不以修改 Prompt 的方式专门过拟合 V2。

## 二、安装和配置

Python 3.10+，macOS / Linux：

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.7.4.zip
cd bilingual-keyword-checker-v0.2.7.4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
python -m pytest -q
```

启动 LM Studio 的本地 OpenAI-compatible API（常用端口 1234），加载两个不同模型，并通过下述接口核实模型 ID：

```bash
curl -s http://127.0.0.1:1234/v1/models | python3 -m json.tool
```

编辑 `.env` 中的 `LM_MODEL`、`LM_VERIFIER_MODEL`，令其与服务器实际返回的 ID 一致。主要设置：

```dotenv
LM_BASE_URL=http://127.0.0.1:1234/v1
LM_MODEL=qwen/qwen3.5-9b
LM_VERIFIER_MODEL=ministral-3-8b-instruct-2512
LM_JSON_SCHEMA=true
LM_MAX_TOKENS=4096
LM_EVIDENCE_MODE=paragraph
LM_EXTRACTION_PROMPT=coarse
LM_COVERAGE_AUDIT=false
LM_ATOMIC_REFINEMENT=false
```

模型 ID 只是配置示例，可以切换到其他支持相同 Schema/API 能力的模型；若后端不支持原生 `response_format=json_schema`，可把 `LM_JSON_SCHEMA=false`，但 Pydantic 校验和失败重试依然执行。

运行网页：

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8005
```

浏览器打开 `http://127.0.0.1:8005/`，API 文档 `http://127.0.0.1:8005/docs`。CLI：

```bash
bilingual-check --en 'The abstract must include purpose, methods, and results.' \
  --zh '摘要须介绍研究目的和研究方法。' --mode dual
```

## 三、随包提供的 Validation Sets

所有文件均位于 `examples/`，与 `python -m examples.evaluate` 完全兼容：

| 数据集 | 数量 | 作用 |
|---|---:|---|
| `validation_v027_paragraph_16.json` | 16（8 PASS / 8 FAIL） | **建议首先测试**：空行、单换行、跨段落、弯引号、隐蔽约束反转 |
| `validation_v2_pilot_20.json` | 20（10/10） | V2 预检集，先看复杂样例是否能完成 |
| `validation_v2_120.json` | 120（60/60） | **主要复杂回归集**：8 类语言与条款结构 |
| `v2_regression_11.json` | 11（含原 5 个误判与 6 个 ERROR） | 已知薄弱案例与协议问题回归 |
| `v2_false_pass_4.json` | 4（全部 FAIL） | 单独观测关键差异被错误放行的情况 |
| `validation_60.json` | 60（30/30） | 旧开发集（多次用于调试，不是盲测） |

**建议的验证顺序**：

```bash
# 1. 新 Paragraph 结构集
python -m examples.evaluate --cases examples/validation_v027_paragraph_16.json \
  --mode dual --out eval_v027_paragraph_16.json

# 2. 20 条复杂预检
python -m examples.evaluate --cases examples/validation_v2_pilot_20.json \
  --mode dual --out eval_v027_pilot_20.json

# 3. 全部 120 条
python -m examples.evaluate --cases examples/validation_v2_120.json \
  --mode dual --out eval_v027_v2_120.json

# 4. 11 条已知问题
python -m examples.evaluate --cases examples/v2_regression_11.json \
  --mode dual --out eval_v027_regression_11.json
```

`examples.evaluate` 会分别汇报 **PASS/FAIL/REVIEW_REQUIRED/ERROR**、False PASS、False FAIL、正确率、按类别和仲裁次数。建议主要看完整测试的**端到端正确率**（正确数 / 总数），并将 REVIEW_REQUIRED（业务上待人工裁定）与 ERROR（请求或协议执行故障）分别统计。两者均不进入已裁决样例准确率的分母；仍需关注 correct / total 的端到端完成比例。

如需严格的旧版 Evidence A/B，修改 `.env`：

```dotenv
LM_EVIDENCE_MODE=quote
```

然后用相同模型、样本、其他参数再跑一遍。别同时修改 Prompt、模型和 Evidence 模式，否则不好归因。

## 四、测试范围与真实运行状态

- 本包离线 pytest（含 Mock API/Mock Schema）的目的是保证结构、协议与分支安全性，不是证明模型语义准确。
- 你先前用 v0.2.6 + Qwen3.5-9B + Ministral 8B 跑 V2 120 条，得到了 **109/114 有效正确，6 ERROR、4 False PASS、1 False FAIL**；这些是旧版基线。
- **v0.2.7 尚未连接到你的 LM Studio 做真实模型 120 条评测**。不能把离线测试通过当作新版本的模型准确率。
- 新版普通纯文本以**空行**定义段落。DOCX、PDF、HTML 的逻辑段落应在上游解析层恢复，不会在这里根据中英文标点猜句子边界。含手工换行但没有空行的输入会被当作一个段落。
- `ERROR` 表示 API、模型响应、Schema 或验证流程执行失败；`REVIEW_REQUIRED` 表示模型意见分歧且无法安全自动裁定，**不是**自动 FAIL 或可计入准确率的 PASS。v0.2.7.4 已在 Web UI、JSON API、CLI 与离线统计中单独表示。

## 五、主要源码

`app/paragraphs.py` — 原文段落与位置；`app/paragraph_pipeline.py` — 默认 Paragraph-ID 多模型执行链；`app/prompts.py` — 任务策略与模型无关的 Prompt；`app/models.py` — Pydantic 输出协议；`app/pipeline.py` — 公用匹配、覆盖、旧版 quote A/B；`app/validators.py` — 有限 Numeric Guard；`app/llm.py` — 模型 API 适配和三次协议重试；`examples/` — 评测工具与完整数据集。

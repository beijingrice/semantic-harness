# 项目地图：重构之后，代码在哪里、怎么串起来

写于 2026-11-19，对应 main 上的提交 `2dba142`（重构之后）。

这份文档的用途是**重新建立对代码的整体印象**：每个文件管什么、一次检查经过哪些函数、判决是怎么定下来的、想改某样东西该去哪里。设计原则和每一步的详细原理在 [architecture.md](architecture.md)；项目现状和问题清单在 [project_analysis.md](project_analysis.md)。

---

## 1. 一句话版本

输入一段英文要求和一段中文要求，判断两者是否表达同一组要求。结果有四种：

| 结果 | 含义 | 从哪里来 |
|---|---|---|
| `PASS` | 一致 | `CheckResult.verdict` |
| `FAIL` | 有缺失或冲突 | `CheckResult.verdict` |
| `REVIEW_REQUIRED` | 程序拿不准，交给人 | `CheckResult.verdict` |
| ERROR | 检查本身失败（模型挂了、输出不合格、配置错） | 抛出异常，**不是** verdict |

整个项目围绕一条底线：**宁可转人工，也不误判**。几乎所有复杂逻辑，都是为了在证据不足时退回 REVIEW_REQUIRED，而不是猜一个 PASS 或 FAIL。

---

## 2. 代码分三层

先记住这三层，再看具体文件。

```
            入口：api.py / cli.py / examples/evaluate.py
                              │
                              ▼
            checker.py ── 按配置选一条 Pipeline
                              │
      ┌───────────────────────┼─────────────────────────┐
      ▼                       ▼                         ▼
 ① 编排层 app/pipelines/   ② 纯规则层（不调用模型）    ③ 模型调用层
 决定按什么顺序问模型、    paragraphs / validators /    llm.py（ChatClient）
 怎样合成判决              source_risks / fast_path /    backends.py
                          dispute_resolution /
                          pipelines/hardened_rules
```

- **编排层**：只有它知道 PASS/FAIL 的含义。它调用纯规则，也通过 `StageRunner` 调用模型。
- **纯规则层**：输入原文，输出确定的结果，不碰网络。单元测试最好写，也最容易藏着按样例写死的规则。
- **模型调用层**：只负责"拿到一个合格的结构化回答"，包括 JSON 解析、Schema 校验、重试和截断恢复。这一层不知道判决是什么。

---

## 3. 文件清单

### app/

| 文件 | 作用 | 最常看的函数或类 |
|---|---|---|
| `api.py` | FastAPI：`/`、`/api/status`、`/api/check`；把异常映射成 HTTP 状态码 | `check`, `status` |
| `cli.py` | 命令行 `bilingual-check`；PASS 退出码 0，FAIL 1，ERROR/配置错误 2，REVIEW_REQUIRED 3 | `main` |
| `config.py` | `Settings`：每次创建时读取环境变量；枚举类配置在 `__post_init__` 里统一校验 | `Settings`, `ConfigError`, `CHOICES` |
| `checker.py` | **对外入口**。`Checker(client, settings).check(...)`；还有只跑抽取的 `extract(...)` | `Checker` |
| `llm.py` | `ChatClient`：结构化生成、重试、截断恢复、请求遥测 | `structured`, `_completion`, `_recover_generation` |
| `backends.py` | 后端协议和 OpenAI-compatible 实现；自定义 Adapter 加载 | `OpenAIChatBackend.complete`, `build_backend` |
| `models.py` | 所有 Pydantic 模型：模型输出格式（`*Draft`）和 API 数据（`CheckResult`） | 见第 6 节 |
| `prompts.py` | 所有提示词（system 和 user 构造函数） | `*_SYSTEM`, `*_user` |
| `paragraphs.py` | 按空行切段，分配不可变 ID（`en_p1`…），根据 ID 取回原文 | `build_paragraphs`, `verify_ids`, `paragraph_evidence` |
| `validators.py` | 数值解析（单位、比较符、区间）和数值/计入守卫 | `parse_quantity`, `parse_range`, `clear_numeric_conflicts`, `source_isolated_quantity_conflicts` |
| `extraction_fast_path.py` | 两条"跳过模型抽取"的结构性捷径 | `source_anchor_pair`, `bounded_clause_anchor_pair` |
| `dispute_resolution.py` | 数值等价证书；判断一条复核异议能否被形式化地驳回 | `numeric_equivalence_certificates`, `narrow_exactness_refutation` |
| `source_risks.py` | 原文风险扫描、原文片段切分（direct_hardened 用） | `source_risks`, `source_spans`, `source_quantitative_issue_only` |
| `evidence.py` | quote 模式：把模型重打的引文定位回原文 | `resolve_evidence` |
| `telemetry.py` | 评测报告里的可靠性统计 | `reliability_summary` |
| `static/` | Web UI（HTML + JS） | |

### app/pipelines/

| 文件 | 作用 | 入口 |
|---|---|---|
| `__init__.py` | 按配置选择 Pipeline | `select(settings)` |
| `base.py` | `StageRunner`：按阶段名调用模型、选择模型；`PipelineStageError` | `StageRunner.call`, `choose_models` |
| `common.py` | 两种 extraction 共用的步骤 | `clean_unsupported_claim`, `match_coverage`, `dispute_snapshot`, `extraction_summary` |
| `paragraph.py` | **默认流程**：extraction + 段落 ID 证据 | `run`, `extract`, `_Dispute` |
| `paragraph_checks.py` | 段落 ID 类回答的结构校验，以及转换成带原文的公开格式 | `verify_review`, `verify_arbitration`, `verify_support`, `verify_objection_recheck` |
| `quote.py` | extraction + quote 证据（A/B 用） | `run`, `extract` |
| `direct.py` | 两个模型直接判整段（direct_hardened 的第一步） | `run_direct`, `direct_numeric_conflicts` |
| `hardened.py` | direct_hardened：选题、追问、汇总 | `run`, `_answer_task` |
| `hardened_rules.py` | direct_hardened 的纯原文规则：选题和证明 | `choose_questions`, `prove` |

**所有 Pipeline 用同一个签名**：

```python
run(runner: StageRunner, english, chinese, requested_model, requested_verifier, mode) -> CheckResult
```

### 其他目录

| 位置 | 内容 |
|---|---|
| `tests/` | 405 个离线测试（mock 模型），`helpers.py` 放共用的 Fake 和样例回答；见第 9 节 |
| `examples/` | 评测脚本（`evaluate.py` 最常用）和全部数据集；数据集索引见 [examples/DATASETS.md](../examples/DATASETS.md) |
| `docs/` | 本文、架构说明、后端接入、标注规范草稿、项目分析；`docs/history/` 是旧版本文档 |
| `scripts/run_local_regressions.sh` | 用真实模型跑回归集和约束集的快捷脚本（输出文件名还是旧的 v0275 命名） |

---

## 4. 一次检查经过的函数（默认配置）

默认配置是 `LM_PIPELINE=extraction` + `LM_EVIDENCE_MODE=paragraph` + `mode=dual`。

```
api.check / cli.main
└─ Settings()                         配置有误时这里就抛 ConfigError
└─ ChatClient(settings)
└─ Checker.check(english, chinese, model, verifier, mode)
   └─ pipelines.select(settings)  →  paragraph.run
      ├─ runner.choose_models()                  主模型 + 复核模型，两者不能相同
      ├─ build_paragraphs()  ×2                  切段，分配 en_p1… / zh_p1…
      ├─ _units()                                得到两边的"事实单元"（KeyUnit）
      │   ├─ source_anchor_pair()                捷径 1：两边都是单句 → 原句直接当单元
      │   ├─ bounded_clause_anchor_pair()        捷径 2：两边都是两个清晰分句
      │   └─ extract() ×2                        否则让主模型抽取（模型调用）
      │        ├─ clean_unsupported_claim()      回滚"添油加醋"的改写
      │        ├─ _dedupe()                      去掉完全重复的单元
      │        └─ _compact_short_source()        短句不许膨胀成很多单元
      ├─ runner.call("keyword_matching")         主模型给英中单元配对（模型调用）
      ├─ match_coverage()                        → 中文缺失 / 英文缺失 / 冲突
      ├─ _guard_issues()                         确定性数值、计入守卫（一票否决）
      ├─ initial = FAIL（有任何问题）或 PASS
      └─ _Dispute.resolve()                      仅 dual 模式
          ├─ runner.call("independent_review")   复核模型只看原文，独立判断
          └─ 意见不一致 → _arbitrate()
              ├─ runner.call("dispute_arbitration")
              ├─ _confirm_support()              仲裁要把 FAIL 翻成 PASS 时
              └─ _recheck_objections()           初判 PASS、复核 FAIL、仲裁 PASS 时
   └─ CheckResult
```

每次 `runner.call(...)` 的内部：

```
StageRunner.call(stage, …)
└─ client.stage_label = stage                  遥测里记阶段名
└─ ChatClient.structured(model, system, user, Schema, schema_name, validate)
   ├─ _payload()                               含 JSON Schema（若启用）
   ├─ _completion()  → backend.complete()      一次 HTTP 请求，记录遥测
   ├─ _validate()                              JSON 解析 → Pydantic → 程序校验函数
   ├─ 不合格 → 带错误提示重试，最多 3 次        _retry_feedback()
   ├─ 截断或空回答 → _recover_generation()     只恢复一次，用 LM_MAX_TOKENS 预算
   └─ 最终失败 → LLMError → 包装成 PipelineStageError(stage)
```

**调用次数**（每步都一次成功的情况）：dual 模式走捷径时 2 次（配对、复核），正常抽取时 4 次，发生分歧再加 1～2 次。

---

## 5. 判决是怎么定的

这是最需要记住的部分。

### 5.1 默认流程（`paragraph.py`）

**第一步：初判**（主模型 + 程序）

`initial = FAIL`，只要以下任一项非空：中文缺失、英文缺失、配对冲突、确定性守卫问题。否则 `initial = PASS`。

`mode=primary` 时，`initial` 就是最终结果。

**第二步：复核**（`mode=dual`）

| 情况 | 最终结果 |
|---|---|
| 复核与初判一致 | `initial` |
| 初判 FAIL 且有守卫问题 | FAIL（不仲裁） |
| 复核与初判不一致 | 进入仲裁，见下表 |

**第三步：仲裁**（`_Dispute._arbitrate`，按顺序判断）

| 初判 | 复核 | 仲裁 | 结果 |
|---|---|---|---|
| FAIL | 任意 | PASS | 复核模型再确认支持（`_confirm_support`）：支持 → PASS；不支持 → **ERROR** |
| PASS | FAIL（带问题） | PASS | 逐条反驳复核异议（`_recheck_objections`）：每条都被驳回，且有数值等价证书 → PASS；否则 → REVIEW_REQUIRED |
| PASS | 任意 | UNSURE | REVIEW_REQUIRED |
| FAIL | 任意 | UNSURE | FAIL |
| FAIL | PASS | FAIL | REVIEW_REQUIRED（仲裁和初判都是主模型，不算独立佐证） |
| 其他 | | | 仲裁结论 |

**最后**：只要守卫有发现，无论前面怎么判，一律 FAIL。

### 5.2 quote 模式（`quote.py`）的不同之处

结构相同，但有三处不同。改代码时容易踩坑：

- **"不一致"的判定更宽**：初判 PASS、复核也 PASS 但附带了问题，也算不一致。
- **初判 PASS、仲裁 UNSURE 时抛 LLMError（ERROR）**，段落模式在同样情况下给 REVIEW_REQUIRED。
- **没有"初判 FAIL / 复核 PASS / 仲裁 FAIL → 转人工"这条规则**，也没有逐条反驳复核异议这一步。

### 5.3 direct_hardened（`direct.py` + `hardened.py`）

```
run_direct：主模型和复核模型各自判整段（不抽取）
├─ direct_numeric_conflicts 发现原文数值冲突 → FAIL
├─ 任一方 UNSURE → REVIEW_REQUIRED
├─ 两者一致 → 该结论
└─ 两者不一致 → REVIEW_REQUIRED

hardened.run：在上面的基础上
├─ 已有数值冲突 → 保持 FAIL
├─ primary 模式：有原文风险且结果是 PASS → REVIEW_REQUIRED
├─ choose_questions 选出 0～3 个固定问题；一个都没有 → 保持 direct 的结果
└─ 每个问题：模型引用原文片段作答 → prove() 用原文核对
   ├─ 任一问题证明 FAIL → FAIL
   ├─ 所有问题都证明 PASS → PASS（只有一种很窄的数量换算情形能证明 PASS）
   └─ 其他 → REVIEW_REQUIRED
```

`hardened_rules.py` 里标着 **CASE-SPECIFIC** 的规则，是照着具体测试样例写的，换一种说法就不会触发，结果退回 REVIEW_REQUIRED。详见 [project_analysis.md](project_analysis.md) 4.1 节。

---

## 6. 关键数据对象（`models.py`）

记住一个区分：**`*Draft` 是模型输出的格式**，**其他是程序内部和 API 用的格式**。

| 对象 | 是什么 | 谁产生 |
|---|---|---|
| `Settings` | 全部配置（`config.py`） | 环境变量 / 关键字参数 |
| `SourceParagraph` | 一个原文段落：`id`、`start`、`end`、`text` | `build_paragraphs` |
| `ParagraphExtractedDraft` | 模型抽取结果：`claim` + `paragraph_ids`，**没有原文** | 模型 |
| `KeyUnit` | 程序整理后的事实单元：`id`（`en_1`…）、`kind`、`claim`、`evidence`（程序取回的原文）、`paragraph_ids` | `extract` / 捷径 |
| `Link` / `MatchDraft` | 配对：`en_ids`、`zh_ids`、`relation`（equivalent / plausible / related_only / conflict） | 模型 |
| `ParagraphReviewDraft` → `ReviewDraft` | 复核意见：前者只有段落 ID，后者由 `review_public` 补上原文 | 模型 → 程序 |
| `ParagraphArbitrationDraft` → `ArbitrationDraft` | 仲裁意见，同上 | 模型 → 程序 |
| `ParagraphDisputeRecheckDraft` | 逐条反驳复核异议 | 模型 |
| `NumericEquivalenceCertificate` | 程序证明"两边是同一个数量" | `dispute_resolution` |
| `RelationVerificationDraft` | direct_hardened 的追问回答 | 模型 |
| `CheckResult` | 最终输出：verdict、各类问题清单、过程记录、diagnostics | Pipeline |

`CheckResult` 各字段的含义见 [architecture.md](architecture.md) 第 8 节。其中 `focused_verification` 字段已经没有任何代码会填写，始终是 null，保留它只是为了不改变 API 格式。

`diagnostics` 是排查问题最有用的字段：每一步都会追加一行，比如 `extract_english: OK (qwen/…)`、`source_anchor_fast_path: …`、`dispute_arbitration: …`。

---

## 7. 配置开关影响哪条路径

| 环境变量 | 默认 | 影响 |
|---|---|---|
| `LM_PIPELINE` | `extraction` | `extraction` 或 `direct_hardened`（要求 paragraph 证据模式） |
| `LM_EVIDENCE_MODE` | `paragraph` | `paragraph.py` 或 `quote.py` |
| `LM_EXTRACTION_PROMPT` | `coarse` | 抽取提示词；只有 `coarse` 时才可能走捷径 |
| `LM_EXTRACTION_FAST_PATH` | `true` | 是否尝试两条捷径；同时决定短句抽取是否用较小的 token 上限 |
| `LM_EXTRACTION_MAX_TOKENS` | `1536` | 短单段抽取的 token 上限 |
| `LM_COVERAGE_AUDIT` / `LM_ATOMIC_REFINEMENT` | `false` | quote 模式的补充审计和拆分；任一开启都会关闭捷径 |
| `LM_MAX_TOKENS` | `3072` | 默认生成上限；截断恢复也用它 |
| `LM_GENERATION_RECOVERY` | `true` | 截断或空回答时是否恢复一次 |
| `LM_SCHEMA_MODE` / `LM_JSON_SCHEMA` | `legacy` / `false` | 是否让后端强制 JSON Schema |
| `LM_MODEL_DISCOVERY` | `required` | 是否调用 `/models` 检查模型 |
| `LM_VERIFIER_*` | 空 | 复核模型使用单独的服务地址和密钥 |

取值有限的选项都列在 `config.py` 的 `CHOICES` 里。填错时，`Settings()` 一创建就会报出具体是哪个变量。

---

## 8. 想改某样东西，去哪里

| 想做的事 | 去哪里 | 注意 |
|---|---|---|
| 改判决规则 | `pipelines/paragraph.py` 的 `_Dispute` | quote 模式在 `quote.py`，规则有差异（5.2 节） |
| 改提示词 | `prompts.py` | 改了要跑真实模型评测，离线测试测不出效果 |
| 支持新单位、新的比较说法 | `validators.py` 的 `parse_quantity` 等 | 加完在 290 条样例上检查有没有误伤，测试放 `test_numeric_bounds.py` |
| 调整捷径条件 | `extraction_fast_path.py` | 捷径只跳过抽取，不跳过配对和复核 |
| 新增一种模型输出 | `models.py` 加 `*Draft`，`prompts.py` 加提示词，`llm.py` 的 `_FORMAT_HINTS` 加重试提示 | 校验函数通过 `runner.call(..., validate=…)` 传入 |
| 新增一条 Pipeline | `pipelines/` 新建模块实现 `run(...)`，在 `pipelines/__init__.py` 的 `select` 里接上，在 `config.py` 的 `CHOICES` 里加选项 | 用 `runner.call` 调模型，用 `runner.choose_models` 选模型 |
| 接入新的模型服务 | `backends.py` 或自定义 Adapter | 见 [backends.md](backends.md) |
| 改 API 或 CLI 的错误映射 | `api.py` / `cli.py` | |
| 改评测指标 | `examples/evaluate.py` 的汇总部分，`telemetry.py` | 注意 `accuracy` 不计转人工的样例 |

---

## 9. 测试地图

全部离线运行，0.7 秒左右跑完：`python -m pytest -q`。

| 测什么 | 文件 |
|---|---|
| 配置、错误映射、API、CLI | `test_config_and_errors.py`, `test_api.py` |
| 模型调用层（重试、截断恢复、reasoning_content） | `test_llm.py`, `test_reliability_and_telemetry.py`, `test_reasoning_json_recovery.py`, `test_retry_format_hints.py` |
| 后端协议、Schema 协商 | `test_backends.py`, `test_paragraph_http.py`, `test_end_to_end.py` |
| 段落证据流程 | `test_paragraph_grounding.py`, `test_paragraph_dispute_recheck.py`, `test_numeric_certificates.py` |
| 捷径 | `test_fast_path.py`, `test_bounded_clause_fast_path.py` |
| 数值和计入守卫 | `test_numeric_bounds.py`, `test_validators.py`, `test_constraint_guards.py`, `test_isolated_numeric_and_compaction.py` |
| quote 模式 | `test_quote_*.py`（6 个） |
| direct_hardened | `test_direct_judgment.py`, `test_hardened_replay_*.py`（回放历史真实模型输出）, `test_count_proof.py`, `test_source_risks.py` |
| 评测工具和数据集格式 | `test_eval_dataset.py`, `test_compare_reports.py` |

- 共用的 `Fake`（假模型客户端）、`config()` 和 `extract()` / `match()` / `review()` / `arb()` 等样例回答都在 `tests/helpers.py`。
- `tests/conftest.py` 会清掉 `LM_*` 环境变量，所以本地 `.env` 不影响测试。
- `tests/fixtures/` 是历史真实模型输出的回放数据。

**这些测试能证明什么**：程序逻辑符合预期。**不能证明什么**：判断是对的。判断准不准，只能用真实模型跑评测：

```bash
python -m examples.evaluate --cases examples/v2_regression_11.json --mode dual --out eval_regression_11.json
```

---

## 10. 需要记住的坑

1. **没有可信的泛化成绩**：120 条主集自 v0.2.6 后没完整跑过，也没有干净的盲测集（[project_analysis.md](project_analysis.md) 4.2 节）。
2. **direct_hardened 的 CASE-SPECIFIC 规则**：回归集上的成绩被高估（4.1 节）。
3. **quote 和 paragraph 两种模式的判决规则不完全一样**（本文 5.2 节）。
4. **评测报告里的 `accuracy` 不计转人工的样例**，看 `end_to_end_correct_rate` 更准确。
5. **`.venv` 是从别处复制来的**，路径指向原位置，建议重建。
6. **main 还没推送**，CI 还没在远端跑过这次重构。
7. **重构前的代码**：tag `v0.2.16` 是 Codex 交付的原始版本；重构之前的最后一个提交是 `6fbf1bc`。

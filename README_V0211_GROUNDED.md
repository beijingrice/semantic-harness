# v0.2.11 — Paragraph-Direct + Source-Grounded Verification

**实验版本；未在真实 LM Studio 后端上验证准确率。** 旧 `extraction` 和 v0.2.10 `direct` 实现保持不变。新的 `direct_grounded` 模式保留完整段落输入，不进行强制句子／Claim 拆分。适用 Python >=3.10。

## 为什么做这一版

已上传的真实 v0.2.10 Direct 11 条评测：6 条自动正确、2 条自动判错、3 条 `REVIEW_REQUIRED`，22 次请求、总耗时约 80.7s。v0.2.8 历史 Extraction 在相同 11 条上的记录为 11/11、49 次请求、188.34s（**历史比较，非同期随机对照**）。特别是 `V2-NUM-15` 两模型同时漏掉计入／不计入的反转；`V2-NUM-03` 两模型都知道单位换算却误判 FAIL。这里只用于识别故障模式；并不是独立未见过的新测试。

## 流程

1. Qwen 和 Ministral 对**完整原文段落**进行两次彼此独立的 Direct 审查，结果保留到 `direct_primary` 与 `review`；输入不要求对齐段落中的句子或顺序。
2. Python 的 `source_risks` 只负责发现**可疑信号**，不自作主张翻译实体或宣告对齐。例如：在独立句子中只有一处 `%` 边界且同值但开闭条件不同；学术页数计入的肯定/否定表达数量不一致。扫描句子仅用于查找数值作用域，不分割或替换给模型的原文，也不尝试按句号配对中英内容。现有直接的、严格单一约束 Numeric Guard 保留。
3. 当两模型意见不同、都 PASS 但 source-only 风险存在、或都 FAIL 但存在**严格的单一可证数值等价**时，才请求一次 `direct_evidence_audit`。它不是简单投第三票；请求中给出具体待验证争议，要求输出 `CONFLICT / EQUIVALENT / UNCERTAIN` 和**双语原文精确摘录**，Python 校验摘录必须真实存在于各自不可变原文。
4. 数值开闭边界自动 FAIL 还要求审核**所引证的两段文本自身**可解析成同值、同单位、不同 comparator。两模型同判 PASS 时，如果发现未能形成形式证明的“计入/不计入”风险，**不能维持 PASS**，保守输出 `REVIEW_REQUIRED`；目前不会凭数量直方图强制宣告语义冲突。
5. 若一方先判 FAIL、另一方先判 PASS，最初不支持 FAIL 的模型必须在定向审核中给出原文有效引文且确认真实冲突，才允许最终 FAIL；否则需人工审查。两模型同时 FAIL 的单一数值等价只有在**严格单一事实、没有其他明显独立义务，并且定向审核支持相同单位数值与精度**时才可恢复 PASS；不是普遍的数字相等 → 整段等价。
6. 复查引文伪造、空缺、结构失效或超预算时，结果为 `REVIEW_REQUIRED`；不会将执行问题变为 PASS。`grounded_evidence_audit` 与 `grounded_risks` 保存在逐条 JSON 报告中。引文来源正确**不意味着**模型的语义推理必然正确，必须看人工审核与新数据实测。

## 可重现实验

### Phase 1（新目录）

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.11.zip
cd bilingual-keyword-checker-v0.2.11
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

`cp .env.example .env` 内含默认主模型 `qwen/qwen3.5-9b`、复核模型 `ministral-3-8b-instruct-2512`、`LM_JSON_SCHEMA=true`。如已有自定义本地配置，可以自行迁移。

### Exec（唯一推荐的真实模型评测）

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --pipeline direct_grounded \
  --mode dual \
  --baseline-report examples/reference_v0210_direct_11.json \
  --stop-after-errors 1 \
  --out eval_v0211_grounded_11.json
```

预期基础请求 22 次；具体争议触发审核额外请求，最长实际执行时间要以用户模型为准。`--baseline-report` 只比对打包的**真实历史 v0.2.10 Direct 运行记录**，不调用旧模型。旧版 v0.2.8 Extraction 结果也保留于 `examples/reference_v028_regression_11.json`。

建议对比三项：正确自动裁决（全体 11 条为分母）、错判率（False PASS 优先）、人工审查数，以及生成 token/秒和每阶段耗时。只看排除 REVIEW_REQUIRED 后的准确率会高估有效性能。本版本提供小样本工程验证，不能当作新泛化集的统计估计。

## 离线测试范围（不能替代 LM Studio）

`tests/test_v0211_grounded_direct.py` 使用精确旧模型输出作为前两次模拟 HTTP 响应，再注入模拟的焦点审核回复，以检查判决逻辑、证据来源验证、自动止损、数值证明限制、延迟请求条件及报告结构。任何“审计状态”为预设数据，**不代表真实模型一定会作出这个判断**。另使用已有 11+36 个样例对 source risk 假警报作离线扫描；它们已参与开发，不是独立的盲测数据。

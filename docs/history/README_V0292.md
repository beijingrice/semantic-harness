# v0.2.9.2 — Bounded Clause Anchors & Dispute Safety

基于 v0.2.9.1 的局部可靠性修复。不更换模型，不更改原有 Numeric Guard 的数学判断，不提高 max_tokens，也不要求在每个版本前跑 Smoke Test。

## 真实问题及范围

用户 v0.2.9.1 真实 11-case 回归：10/11，唯一错误 `V2-ENU-13` 本应 PASS 却被判 FAIL，耗时 54.9s。该例：英文提取 2 个事实，中文生成 8 个同义事实并删除 21 条精确重复。Matching 把第二句的 `may` 错绑到第一句的内容清单；Ministral Reviewer 对原文判 PASS，Qwen 仲裁仍错误判 FAIL。该问题不由原单句 Fast Path 触发，原单句 Fast Path 在这个样例上不适用。

## 生成前防膨胀策略（Pre-generation）

原有 `source_anchor_pair` 仅覆盖双语双方都是单一完整短句。新增 `bounded_clause_anchor_pair`：

1. Python 在**发送任何提取请求前**读取不可变原文段落。只接受两侧均恰好一个段落、文本较短的情况。
2. 英文须刚好两句（严格的结束标点）；中文须刚好一个可用于拆分的逗号（完整句号在末尾），可以是“两句↔两子句”的双语非对称结构。
3. 遇到数字、多个逗号、分号、多段落、多于两句、条件/例外/转折、方向否定等复杂结构，**立即放弃 Fast Path**；改回现有 LLM Extraction，保留复杂规则的拆分能力。
4. 满足条件时按标点把源文本分成两段。只引用**原文切片**，不进行翻译、去模态词或模型改写。每个原文切片绑定所在的原始 `paragraph_id`，完整原文仍作为 evidence。
5. 跳过两次提取请求；原 `MATCH_SYSTEM`、Ministral 只读原文复核、确定性 Guard 和争议机制都照常运行。Fast Path 不是等价证明，不负责直接决定 PASS。
6. 不满足 Fast Path 的短段落仍继承原有 `LM_EXTRACTION_MAX_TOKENS=1536`；复杂长文依旧使用 `LM_MAX_TOKENS`。单个模型有可能再次陷入同义输出；未声称已完全解决所有生成膨胀。

**实例：**

English:
- `The review addresses data quality, annotation consistency, and evaluation reproducibility.`
- `These topics may be discussed in any order.`

中文：
- `综述需要涉及数据质量、标注一致性以及评测可复现性`
- `三者的叙述顺序不作限制。`

旧版：中文提取模型自由生成 8 个类似事实；新版：直接得到 `en_1,en_2,zh_1,zh_2`，没有提取的模型调用，不会在该阶段生成 29 秒的同义文本。

## 仲裁安全改变

当初判 FAIL、独立 Reviewer 对完整原文判 PASS、而 Qwen 仲裁依然 FAIL 时：如果没有独立、确定性证据支持冲突，最终为 `REVIEW_REQUIRED`，**不能直接把重复的 Qwen 判决当成独立证据**。这是阻止未核实 FALSE FAIL 的保护措施，**不是自动提升为 PASS**。

已有明确确定性数值冲突时，Guard 依然强制 FAIL；当初判 FAIL、Reviewer FAIL，无须仲裁，继续 FAIL。初判 PASS、Reviewer FAIL 的原有严格核查逻辑保持不变。

## 验收与局限

- 新测试验证：完整两子句来源保留；HTTP 实际只调用 Matching + Reviewer；原文仍送独立 Reviewer；多个复杂反例不会进入新 Fast Path；真假义务冲突不会轻率 PASS；未验证的仲裁冲突变为人工复核；数值边界 Guard 没有放宽。
- 交付前跑全部离线单元/HTTP 模拟测试并从正式 ZIP 全新解压复验。
- **未连接用户的 LM Studio 实测**。真实 11 条回归准确率和耗时需要用户的一次完整 Exec。不能由离线测试推断真实 11/11。

## Phase 1

```bash
cd ~/PycharmProjects
unzip ~/Downloads/bilingual-keyword-checker-v0.2.9.2.zip
cd bilingual-keyword-checker-v0.2.9.2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp ../bilingual-keyword-checker-v0.2.9.1/.env .env
```

## Exec（仅一条完整历史回归，不先跑 Smoke）

```bash
python -m examples.evaluate \
  --cases examples/v2_regression_11.json \
  --mode dual \
  --stop-after-errors 1 \
  --out eval_v0292_regression_11.json
```

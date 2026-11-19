# 评测数据集索引

本目录下所有 `*.json` / `*.jsonl` 数据集的用途、状态和相互关系。标注时依据的判断规则见 [docs/labeling_guidelines.md](../docs/labeling_guidelines.md)。

## 状态说明

| 状态 | 含义 |
|---|---|
| **开发集** | 调试时反复使用过。上面的成绩会偏乐观，不能代表泛化能力 |
| **回归集** | 已知薄弱案例，每次改动后都应该跑；同样不是盲测 |
| **部分污染的 holdout** | 本来是留出集，但其中部分样例后来被用于调试 |
| **冒烟集** | 少量样例，用于快速确认某个修复是否生效 |

> **盲测集**：`validation_v030_blind_99.json` 是 v0.3.0 新建的干净盲测集，使用规则见 [docs/eval_protocol/README.md](../docs/eval_protocol/README.md)：只看汇总指标，不读样例，不据此调试。

## 主要数据集

| 文件 | 条数 | PASS / FAIL | 状态 | 用途 |
|---|---:|---:|---|---|
| `validation_v030_blind_99.json` | 99 | 50 / 49 | **盲测集（冻结）** | 隔离生成、独立复核标签的泛化测试集，附 `.sha256`。只看汇总指标，见 [评测协议](../docs/eval_protocol/README.md) |
| `validation_v030_gen_dev_60.json` | 60 | 30 / 30 | 开发集 | 与盲测集同分布、同方法生成的开发集，用于分析错误和改进 |
| `validation_v030_challenge_claude_32.json` | 32 | 16 / 16 | 挑战集 | Claude 编写的高难度跨模型家族样例；第一次运行后已用于修复数值守卫，不是盲测 |
| `v2_regression_11.json` | 11 | 5 / 6 | 回归集 | 历史上的误判与 ERROR 案例，**每次改动后先跑** |
| `validation_v2_120.json` | 120 | 60 / 60 | 开发集 | 主要复杂集：8 类语言与条款结构，含 41 组最小差异对 |
| `validation_v2_pilot_20.json` | 20 | 10 / 10 | 开发集 | 120 条集的快速预检子集 |
| `v2_false_pass_4.json` | 4 | 0 / 4 | 回归集 | 专门观察关键差异被错误放行的情况 |
| `validation_v028_holdout_36.json` | 36 | 18 / 18 | **部分污染的 holdout** | v0.2.8 时冻结的泛化集，附 `.sha256` 校验文件。见下方说明 |
| `validation_v027_paragraph_16.json` | 16 | 8 / 8 | 开发集 | 段落切分、软换行、跨段落、弯引号 |
| `validation_v0272_constraints_12.json` | 12 | 6 / 6 | 开发集 | 单位换算、数值边界、计入/排除 |
| `validation_v030_numeric_phrasing_22.json` | 22 | 11 / 11 | 开发集 | v0.3.0 新增：数值边界的多种说法（单位后的"以上/or more/or fewer"、exceed/above、时间上的 within/以内、非时间的 within、not exceeding、年龄）。为测试数值守卫修复而写，**不是盲测** |

### holdout 污染说明

`validation_v028_holdout_36.json` 中的 **H28-02、H28-03、H28-04、H28-13** 在 v0.2.8 评测时出错，之后被原样复制进 `validation_v029_smoke_6.json` 和 `validation_v0291_smoke_2.json`，用于开发 v0.2.9。因此：

- v0.2.8 的 34/36 是有效的 holdout 成绩；
- **v0.2.9 及以后**，只有其余 32 条还算干净。报告成绩时请把这 4 条单独列出；
- 这个集合全是短句、单处差异，没有长段混合约束和低重合改写，难度明显低于 120 条集，两者的成绩不能直接比较。

## 冒烟集与历史集

| 文件 | 条数 | 状态 | 说明 |
|---|---:|---|---|
| `validation_v0275_smoke_3.json` | 3 | 冒烟集 | 单位换算；内容来自 120 条集和 constraints_12 |
| `validation_v028_reliability_smoke_4.json` | 4 | 冒烟集 | 协议与恢复机制 |
| `validation_v029_smoke_6.json` | 6 | 冒烟集 | 其中 4 条复制自 holdout（见上） |
| `validation_v0291_smoke_2.json` | 2 | 冒烟集 | 两条都复制自 holdout（H28-02、H28-13） |
| `validation_60.json` | 60 | 开发集 | 最早的开发集，多次用于调试。部分 AMB 类样例按"宽松通过"标注，比新集宽松 |
| `validation_60.jsonl` | 60 | — | 与 `validation_60.json` 内容完全相同，**有意保留**：`tests/test_eval_dataset.py` 用它验证 JSONL 格式的读取 |
| `errors_5.json` | 5 | 开发集 | `validation_60.json` 的子集，早期的 5 个 ERROR 案例 |
| `v024_targeted_14.json` | 14 | 开发集 | 大部分样例与 `validation_60.json` 相同，另加 4 条 POL 类 |
| `v026_smoke_14.json` | 14 | 开发集 | 描述句与规范句、列举、角色等 |
| `manual_cases.jsonl` | 16 | 开发集 | 早期手写样例；没有 `category` 字段 |

## 子集关系

- `v2_false_pass_4` ⊆ `v2_regression_11` ⊆ `validation_v2_120`
- `validation_v2_pilot_20` ⊆ `validation_v2_120`
- `errors_5` ⊆ `validation_60`；`validation_60.json` = `validation_60.jsonl`
- `validation_v0291_smoke_2` ⊆ `validation_v028_holdout_36`，也 ⊆ `validation_v029_smoke_6`

同一个样例 ID 在不同文件中的内容都一致。

## 参考报告

`reference_*.json` 不是数据集，而是历史版本用真实模型跑出的评测结果，可以配合 `examples.evaluate --baseline-report` 逐条对比。其中保留的是当时原样的 `why` 文本。

## 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 样例编号，全局唯一 |
| `en` / `zh` | 是 | 英文 / 中文原文；用空行分隔段落 |
| `expected` | 是 | `PASS` 或 `FAIL` |
| `why` | 是 | 标注理由。最小差异对的 FAIL 样例会写明与哪条 PASS 样例配对 |
| `category` | 建议 | 类别 |
| `difficulty` | 可选 | `easy` / `medium` / `hard` |

## 已知的构造偏差

- 120 条集的 FAIL 中文常常是压缩改写，除了 `why` 中声明的那一处差异，有时还省略了其他细节（如 V2-CRS-15、V2-EXC-10、V2-MIX-10、V2-NUM-13）。标签仍然正确，但如果人工审核模型给出的理由，模型指出的"另一处差异"也可能是真实存在的。
- 41 组最小差异对中，有 22 组的 FAIL 中文明显比 PASS 短（只有 5 组更长），模型有可能利用长度作为线索。
- `validation_v0272_constraints_12.json` 的 C27-10（"样本须少于五天内销毁"）是为测试解析器构造的，不是自然的中文表达。

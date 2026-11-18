# 项目总体分析（v0.3.0，2026-11-19）

本文根据仓库当前代码、测试、数据集和本轮真实模型评测结果，评估项目的实际状态。项目早期由 Codex 开发，第 5 节专门讨论那一阶段留下的问题。

---

## 1. 结论

**核心设计是对的，但目前测出来的成绩不能代表真实水平。**

- 默认的 extraction pipeline 设计扎实：防止模型编造证据、守住数值底线、把"不确定"和"出错"分开。这几点是这个项目最有价值的部分，应当保留。
- 实验性的 direct_hardened pipeline，有相当一部分"程序核验"其实是**按具体测试样例写死的规则**，其中两条规则来自本应保持干净的 holdout 样例。它在回归集上的成绩，很大程度上是在用题目验证为这些题目写的答案。
- 最主要的复杂评测集（120 条）**自 v0.2.6 之后再没有在真实模型上完整跑过**。中间经过了十个版本，所有改进都只在 11 条回归集和几条冒烟样例上验证过。
- 目前**没有干净的盲测集**，所以无法回答"它在新文本上表现如何"。

下一步最值得做的不是继续加规则，而是**先建立一个可信的测量基线**（见第 7 节）。

---

## 2. 现状速览

| 项目 | 现状 |
|---|---|
| 版本 | v0.3.0；git 从 v0.2.16 基线开始，共 13 个提交 |
| 代码 | `app/` 共 19 个模块、约 4,700 行、152 个函数 |
| 测试 | 33 个文件、405 个测试，全部离线（mock 模型），用时约 0.8 秒；ruff 无告警；CI 在 Python 3.10 和 3.14 上运行 |
| 入口 | CLI（`bilingual-check`）、FastAPI JSON API、单页 Web UI |
| Pipeline | `extraction`（默认）和 `direct_hardened`（实验） |
| 后端 | OpenAI-compatible（LM Studio 已实测），可自定义 Adapter |
| 数据集 | 约 290 条样例，分布在 19 个文件；详见 [examples/DATASETS.md](../examples/DATASETS.md) |

### 本轮真实模型成绩（Qwen3.5-9B + Gemma 4 12B QAT）

| 测试 | extraction | direct_hardened |
|---|---|---|
| 11 条回归集 | 11/11 | 9/11，2 条转人工 |
| 22 条数值集（本轮新写） | 21/22，1 条 ERROR | 21/22，1 条转人工 |
| 误判 PASS / 误判 FAIL | 0 / 0 | 0 / 0 |
| 每条耗时（direct_hardened 回归集） | — | p50 22 秒，p95 88 秒 |

**这些数字的局限**：回归集是调试时反复使用的已知样例；数值集是为了验证本轮修复而写的。两者都不是盲测（见第 4.2 节）。

---

## 3. 做得好的地方

这些设计应当保留，后续重构也不要破坏。

### 3.1 不让模型编造证据

在段落模式下，模型只能引用段落 ID（如 `en_p2`），引用的原文由程序从输入中取出；不存在的 ID 会被拒绝。这从结构上避免了"模型引用一段原文里没有的话"这种最常见的幻觉。

### 3.2 判决分四种，而不是两种

`PASS` / `FAIL` / `REVIEW_REQUIRED` / `ERROR` 的含义各不相同：

- 模型调用失败返回 ERROR（API 返回 502），不会被当成业务上的 FAIL；
- 证据不足或模型之间无法裁定时返回 REVIEW_REQUIRED，不会勉强给出 PASS。

对于"放过一处真实差异"代价很高的场景，这种设计是正确的。本轮所有测试中，没判对的样例都落在转人工或 ERROR，而不是误判。

### 3.3 数值守卫有一票否决权

数值边界（"至少 10 页"与"超过 10 页"）是模型最容易看错、后果又最具体的一类差异。`validators.parse_quantity` 用确定性规则解析，发现冲突就直接判 FAIL，不依赖模型判断。

### 3.4 模型输出的工程处理完整

`llm.ChatClient.structured` 支持 JSON Schema、纠错重试和截断恢复，还兼容 LM Studio 把 JSON 放进 `reasoning_content` 的情况。每次请求都记录 token 用量、耗时和处理结果。对于本地小模型，这些处理都是必需的。

### 3.5 有评测工具

`examples/evaluate.py` 可以批量评测、和历史报告逐条对比，并输出误判率、转人工率和延迟分位数。

---

## 4. 主要问题

按严重程度排列。P0 会影响"能否相信这个项目的成绩"；P1 影响后续能否持续改进；P2 是代码质量和体验问题。

### 4.1 【P0】direct_hardened 的部分规则是测试样例查找表

direct_hardened 的思路是：先由两个模型判断，再由程序用规则"核验"。问题在于，很多规则不是通用机制，而是对着具体样例写出来的。

| 代码位置 | 规则内容 | 对应的样例 |
|---|---|---|
| `relation_direct.py` `_focus_exception_evidence` | 只在实体**恰好是** `chinese abstract`，且英文出现 `optional unless` + `program` + `requir…`、中文出现 `无论` + `项目/专业` + `都…可选` 时才生效，其他实体一律返回 None | V2-EXC-03（以及 120 集里的 V2-EXC-02） |
| `relation_direct.py` `choose_questions` | 只有英文含 `unless`、中文含 `无论/不论/除非` 时才生成例外条款问题 | 同上 |
| `relation_direct.py` `_scope_name` / `_host_anchor_direction` | 只认 `first/second/third dataset` 和 `第一组数据`；问题只在出现 `outperformed / higher score` 时触发 | V2-ROL-07、V2-ROL-06 |
| `relation_direct.py` `_BOUND_ANCHORS` | 数值冲突要升级为 FAIL，两边必须各命中词表里至少两个词。11 组词中，`calibration/校准 + error/误差` 和 `font size/字体大小 + table/表格` 分别对应 **holdout 样例 H28-02 和 H28-04**；`invalid + mandatory + response` 对应 V2-NUM-12/13；`table + participant` 对应 V2-MIX-09 | 回归集 + **holdout** |
| `relation_direct.py` `_ENTITY_PAIRS` | 只收录 12 个实体名（cover page、appendix、main report、baseline method……） | V2-NUM-15 等 |
| `dispute_resolution.py` `_COMPOUND_SCOPE_ZH` | 通用连词列表里混入了 `可以删除`、`须归档` | 早期调试样例 |

**后果：**

1. **回归集成绩被高估。** 11 条回归集中，至少 6 条（EXC-03、ROL-07、NUM-12、NUM-13、NUM-15、MIX-09）有专门为它们写的代码路径。9/11 在相当程度上是"规则命中了为它写的题"。
2. **holdout 被污染到了代码层面。** 污染不只是"把 holdout 样例复制进冒烟集"（DATASETS.md 已记录），holdout 的具体措辞也被写进了产品代码。
3. **在新文本上会大量转人工。** 换一个实体名、换一种说法，这些规则就不会触发，结果退回 REVIEW_REQUIRED。这样不会误判，但自动判决的覆盖率会远低于回归集上看到的数字。本轮 NUM30-09F 就是一个实例：同样的计入冲突换了说法，核验就失效了。我已经把这一处改成通用匹配，但上表中的其他规则仍是逐案写法。

注释里写着"Only HIGH-PRECISION bilingual academic terminology"，说明作者本意是建一个高精度小词表。但从词条和样例一一对应的情况看，这个词表实际上是从失败样例里反向摘出来的。

### 4.2 【P0】没有可信的泛化测量

- **120 条主集停测十个版本。** 最后一次完整的真实模型成绩是 v0.2.6（Qwen3.5-9B + Ministral 8B，109/114 有效正确，6 条 ERROR，4 条误判 PASS，1 条误判 FAIL）。直到 v0.2.16，根目录 README 里仍然写着"v0.2.7 尚未连接到你的 LM Studio 做真实模型 120 条评测"（见 [docs/history/README_v0.2.16_root.md](history/README_v0.2.16_root.md)）。此后的段落锚定、fast path、争议复核、五个 direct 实验，都只在 11 条回归集和冒烟集上验证过。
- **没有干净的盲测集。** v0.2.8 的 36 条 holdout 中有 4 条被拿去调试，2 条的措辞进了代码（见 4.1），剩下的又全是短句、单处差异，难度明显低于 120 集。
- **数据集本身有构造偏差。** 最小差异对中，22/41 组的 FAIL 中文明显更短，模型可能把长度当线索；部分 FAIL 样例除了标注的那一处差异，还省略了别的细节。
- **标注规则尚未定稿。** should 与 must、描述句与规范句、充分与必要条件、近义词宽严这四类问题，都还标着"待定"（[labeling_guidelines.md](labeling_guidelines.md)）。规则不统一，同一类样例的标签就可能互相矛盾，准确率的意义也会打折扣。

**结论：现在说不出"这个系统在新文档上的准确率是多少"。**

### 4.3 【P1】报告中的 accuracy 不计转人工样例

`examples/evaluate.py` 的汇总里，`accuracy = correct / evaluated`，而 `evaluated` 不包括转人工和 ERROR 的样例。本轮 direct_hardened 回归集的报告写着 `"accuracy": 1.0`，但 11 条中实际只有 9 条自动判对。

报告中其实也有 `end_to_end_correct_rate`（0.818）和 `review_required_rate`，只是 `accuracy` 排在最前面，最容易被引用。一个系统只要把难题都转人工，accuracy 就能接近 100%。建议把端到端正确率、转人工率、误判率三项并列作为主指标。

### 4.4 【P1】架构耦合：Checker 是上帝对象

- `pipeline.Checker` 同时负责抽取、匹配、覆盖审计、证据校验和仲裁，还是三条执行路径的调度入口。
- 段落模式和 direct_hardened 写成模块级函数 `check_paragraph(checker, …)` 和 `check_hardened(checker, …)`，把 Checker 整个传进去，再调用它的私有方法 `checker._structured`。
- 模块之间导入私有函数：`direct_pipeline` 从 `paragraph_pipeline` 导入 `_review_public` 和 `_verify_review`；`relation_direct` 又依赖 `direct_pipeline.check_direct`（这是已删除的 `direct` 模式留下的内核）。
- `pipeline` 和 `paragraph_pipeline` 互相依赖，只能靠函数内的延迟 import 绕开循环导入。

结果是没有一个清晰的 "Pipeline" 接口。新增或替换一条 pipeline，必须了解 Checker 的内部实现。

### 4.5 【P1】配置开关的组合太多

现在影响判决路径的开关有：`LM_PIPELINE`、`LM_EVIDENCE_MODE`（paragraph/quote）、`LM_EXTRACTION_PROMPT`（coarse/proposition）、`LM_COVERAGE_AUDIT`、`LM_ATOMIC_REFINEMENT`、`LM_EXTRACTION_FAST_PATH`、`LM_SCHEMA_MODE`、`LM_JSON_SCHEMA`、`LM_GENERATION_RECOVERY`，还有单模型/双模型两种模式。

理论上的组合有上百种，真实模型评测过的只有默认组合附近的几种。quote 模式的抽取逻辑（`pipeline.Checker.check`，约 150 行）和段落模式（`paragraph_pipeline.check_paragraph`，约 190 行）是两套平行实现。之前决定保留这些 A/B 开关，这个决定是合理的，但应当给它们设一个期限：在新基线上跑一次对比，表现不好的选项就删掉。

### 4.6 【P1】正则规则越补越多

`relation_direct.py` 有 42 处正则调用，`validators.py` 有 28 处，`extraction_fast_path.py` 有 23 处。本轮修数值守卫时也看到了这种模式的代价：

- 每发现一种新说法（"10 页以上"、"not exceeding"、"18 周岁"），都要加一条正则；
- 每加一条正则，又可能误伤别的说法（"少于五天内"、"within 30 pages" 在修复过程中都曾出错）。

数值解析本身适合用规则来做，因为它的语法有限、结果可以验证。但 direct_hardened 用正则来判断"两句话是不是在说同一件事"，这件事规则做不好，最后只能变成 4.1 里的样例查找表。

### 4.7 【P2】代码质量

- **函数过长**：`ChatClient.structured` 238 行，`check_paragraph` 190 行，`Checker.check` 155 行，`Checker.extract` 154 行，`check_hardened` 127 行。其中嵌套了多层闭包（`validate`、`request`、`validate_refinement`……），很难单独测试。
- **风格不统一**：`relation_direct.py` 大量使用单行压缩写法（`a,b=…`、缺少空格），单双引号混用，和其他模块风格不一致。ruff 只开了错误类规则，不检查这些。
- **注释写成了版本日志**：代码注释里常见 "v0.2.9.2: …"、"catch the V2-NUM-15 failure" 这类叙事。这些内容应该放在 git 历史里，注释只需说明代码现在做什么、为什么这样做。
- **提示词大量使用全大写强调**：`prompts.py` 580 行，全大写的 ONLY 出现 16 次，ORIGINAL 12 次，EXACT 9 次，MUST 7 次。对小模型来说，强调词过多会稀释真正重要的约束，而且没有实验证明这样写更有效。

### 4.8 【P2】测试结构

- 33 个测试文件中有 20 个按版本号命名（`test_v0272_…`、`test_v0291_…`），看文件名无法知道它测的是哪个模块或行为。
- 6 个测试文件从其他测试文件导入辅助函数（例如 `from test_v027_paragraph_grounding import Fake, config, …`），这些辅助函数应当放进 `conftest.py` 或单独的 helper 模块。
- 405 个测试全部使用 mock 模型。它们保证的是"程序逻辑符合预期"，不保证"判断是对的"。真实模型评测是手动执行的，没有固定的流程，也不会进入任何报告，这正是 4.2 中主集停测十个版本却没人发现的原因。

### 4.9 【P2】部署与使用

- **单条检查很慢**：本地模型下 p95 接近 90 秒，最坏情况会有几分钟（例如截断后再恢复）。API 是同步请求，没有任务队列、进度反馈、取消或并发限制；多个用户同时提交，会直接压垮本地 LM Studio。
- **双模型模式要求两个不同的模型**，部署成本翻倍，而单模型模式的效果没有评测过。
- **名称和实际功能不符**：项目叫 keyword checker，配置里还有 `MAX_KEYWORDS`，但实际做的是"双语要求一致性核查"。这是早期设计留下的名字。
- **`.venv` 是从别的目录复制过来的**，其中的路径仍指向原来的位置，建议重建。

---

## 5. 关于 Codex 开发阶段的问题

以下是从代码、文档和 git 基线中能看到的开发方式问题。这些问题不是某一行代码写错，而是工作流程本身的问题，所以单独列出。

### 5.1 逐案打补丁，用回归集当目标

v0.2.10 到 v0.2.15 的六个版本里，先后做了 `direct`、`direct_grounded`、`direct_verified`、`direct_questions`、`direct_relations`、`direct_hardened` 六条实验 pipeline，其中五条已经被废弃。每个版本的发布说明，几乎都以"修复了回归集的某几条"作为成果。

这样做的问题是，回归集从"检查有没有退步的工具"变成了"优化目标"。目标一旦变成了被优化的对象，它就不再能衡量真实效果（Goodhart 定律）。4.1 中的规则就是这个过程的直接产物。

### 5.2 holdout 的使用方式不对

正确的做法是：holdout 冻结之后只用来报告成绩，失败的样例也不能拿去调试。

实际情况是，v0.2.8 holdout 中出错的 4 条被原样复制进冒烟集，用来开发 v0.2.9；其中两条的措辞还被写进了 `_BOUND_ANCHORS`。项目甚至为 holdout 准备了 `.sha256` 校验文件，形式上很严谨，但保护的只是文件内容，没有约束它被怎样使用。

### 5.3 规则和测试循环论证

很多单元测试的结构是：取一个失败样例的原文，断言新规则对它返回 FAIL。这只能证明"规则能命中写规则时参照的那条样例"，不能证明规则在别的说法上也成立。本轮放宽计入核验时，我在全部 290 条样例上逐一检查有没有误伤，这种检查以前没有做过。

### 5.4 文档的说法比实测更强

各版本 README 大量使用"严格""保证""不会""只有在……才会"这类绝对化表述，而对应的真实模型验证往往只有 11 条，甚至没有做。最典型的是：从 v0.2.7 到 v0.2.16，文档一直承认 120 条主集没有在新版本上跑过，但这期间仍然连续发布了九个版本，每个版本都声称有改进。

在本轮整理之前，根目录堆着 19 个版本 README，项目也没有用 git 管理。

### 5.5 过度防御的写法

代码注释和提示词中有大量"绝不""必须""ONLY"之类的措辞，还有很多注释在解释"这里为什么不会出错"。这种写法让代码看上去很谨慎，但是否真的可靠，只能靠测量来证明。

### 5.6 也要肯定的部分

公平地说，Codex 阶段确立了这个项目最有价值的几项设计：段落 ID 证据锚定、四种判决的语义、数值守卫一票否决、截断恢复和 reasoning_content 兼容、请求级遥测。问题主要出在验证和取舍上，核心思路本身没有方向性错误。

---

## 6. 本轮已经完成的改进

| 类别 | 内容 |
|---|---|
| Bug 修复 | 环境变量只在导入时读取一次；配置值无效时报错不清楚；后端返回结构异常时泄出 `AttributeError`；测试受本地 `.env` 影响 |
| 精简 | 删除五条废弃的 direct pipeline 及对应的模块、测试和报告；合并 19 个版本 README |
| 工程 | 引入 git（以 tag `v0.2.16` 为基线）、ruff、GitHub Actions CI |
| 数值守卫 | "以上 / or more / exceed / above / not exceeding"、时间单位上的 within/以内、"周岁"等写法不再被误读成"恰好等于" |
| 可靠性 | 截断恢复请求改用 `LM_MAX_TOKENS` 作为上限 |
| direct_hardened | 计入核验改为通用匹配（已在 290 条样例上检查无误伤）；只能转人工的通用追问不再调用模型，最长从 212 秒降到 17 秒 |
| 数据集 | 新增数据集索引、标注规范草稿和 22 条数值措辞集；修正 15 条指向错误的 `why` |
| 文档 | [README](../README.md)、[architecture.md](architecture.md)、[backends.md](backends.md)、[CHANGELOG](../CHANGELOG.md) |

测试从 363 个增加到 405 个。

---

## 7. 建议路线图

按顺序做。前两步完成之前，不建议再给任何一条 pipeline 加新规则。

### 第一步：建立可信基线（最重要）

1. **定稿标注规范**：先决定 [labeling_guidelines.md](labeling_guidelines.md) 中四个待定问题，否则后面的标签没有统一标准。
2. **新建一个干净的盲测集**：50 到 100 条，最好由没看过现有样例的人编写，或者从真实的课程、项目要求文档中摘取。要包含长段落和低重合度的改写，避免 FAIL 样例系统性地比 PASS 短。冻结后只用来报告成绩，失败样例不拿去调试。
3. **用当前代码跑全量评测**：120 条主集 + 新盲测集，两条 pipeline 都跑。主指标并列报告：端到端正确率、转人工率、误判 PASS 率、误判 FAIL 率、p95 耗时。

### 第二步：根据数据决定 direct_hardened 去留

- 如果它在盲测集上的端到端正确率不比 extraction 高，或者转人工率明显偏高，建议**删除 4.1 中的样例特化规则**，或者干脆把 direct_hardened 退回实验分支。
- 如果它在某类问题上确实更好，就把对应的规则改写成通用机制，并用盲测集验证，不再用回归集。

### 第三步：收敛配置

在新基线上对比各个 A/B 开关，删除表现不好的选项。至少可以评估 quote 证据模式、proposition 抽取提示词、coverage audit、atomic refinement 是否还有保留的必要。

### 第四步：重构

- 定义统一的 `Pipeline` 接口（输入两段原文和配置，输出 `CheckResult`），两条 pipeline 各自实现，不再把 Checker 整个传来传去，也不再跨模块导入私有函数。
- 拆分超长函数，先从 `ChatClient.structured` 开始。
- 按模块和行为重新组织测试文件，把共用的辅助函数放进 `conftest.py`。
- 清理注释里的版本叙事。

### 第五步：使用体验

- API 改成提交任务后轮询结果（或用 SSE 推送进度），并限制同时执行的检查数量。
- 评估单模型模式的效果。如果差距不大，可以降低部署门槛。
- 考虑把项目改名，或者至少在 README 里说明名称的由来。

### 随时可做的小事

- 重建 `.venv`；
- 推送到远端，让 CI 真正跑起来；
- 给当前状态打 `v0.3.0` tag。

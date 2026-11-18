# Changelog

各版本的完整说明在 [docs/history/](docs/history/)。v0.2.16 及之前的代码可以通过 git tag `v0.2.16` 找回。

## v0.3.0

**移除与精简**
- `LM_PIPELINE` 只保留 `extraction`（默认）和 `direct_hardened`。删除 `direct`、`direct_grounded`、`direct_verified`、`direct_questions`、`direct_relations` 五种早期实验模式，以及对应的模块、测试和参考报告。
- `direct_hardened` 始终启用 hardened 检查；`source_spans` 和 `source_quantitative_issue_only` 移到 `app/source_risks.py`。
- 根目录的 19 个版本 README 合并：`README.md` 只介绍当前版本，后端接入说明移到 `docs/backends.md`，历史文档移到 `docs/history/`。

**修复**
- `Settings` 改为每次创建时读取环境变量，不再只在导入时读一次。
- 配置值无效时（如 `LM_MAX_TOKENS=abc`、未知的 `LM_PIPELINE`）抛出指明变量名的 `ConfigError`：API 返回清楚的 500 错误，CLI 以退出码 2 退出。
- OpenAI-compatible 后端在读取 `choices`、`message`、`usage` 之前先检查类型，提供商返回的结构异常时会得到 `BackendError`，不再泄出 `AttributeError`。
- 离线测试通过 `tests/conftest.py` 与本地 `.env` 隔离。
- 数值守卫：修复几种等价写法被误读成"恰好等于"、进而被一票否决判成 FAIL 的问题。单位后的"以上/or more/or fewer"（如"10 页以上"）、exceed、above 现在能正确识别；within、内、以内只对时间单位按上限处理；"not exceeding" 不再被误读成 `>`。

- 数值守卫："18周岁"不再被误读成 18 周。
- direct_hardened 计入核验：不再要求原文字面出现开发样例的措辞（"main report limit"、"N页限制"）。现在只要两边被引用的规则指向同一个已识别的限额即可，例如写出数值、用原文绑定到该数值的范围名，或用"this limit / 该限制"；并能识别 is counted within、counts toward、does not count、不算入 等说法。指向其他限额（如 appendix limit）、一边有多个限额数值、counted separately 时仍然放弃判断。在全部 290 条数据集样例上逐一检验，没有任何 PASS 样例因此变成 FAIL。
- direct_hardened：两个模型意见不一致、但没有可核对的原文风险时，程序生成的通用追问（Q_DISPUTED_CLAIM）无论模型怎么回答，结果都只能是 REVIEW_REQUIRED，现在直接跳过这次模型调用，判决不变。真实测试中，这次调用最多白白耗费约 3 分钟。
- 截断恢复：输出被截断时，唯一一次恢复请求改用 `LM_MAX_TOKENS` 作为上限，不再沿用提取（1536）或 direct_hardened 追问（1536）的较小预算。真实模型测试中，推理把 1536 个 token 全部用完后，用同样上限的恢复请求会在同一位置再次被截断。

**数据集**
- 新增 `examples/DATASETS.md`：每个数据集的用途、状态、子集关系和已知构造偏差；标明 v0.2.8 holdout 中的 H28-02/03/04/13 后来被用于调试。
- 新增 `docs/labeling_guidelines.md`（草稿）：整理现有标注实际遵循的规则，并标出 should、描述句与规范句、充分与必要条件、近义词宽严四处待定问题。
- 新增 `validation_v030_numeric_phrasing_22.json`：数值边界多种说法的 11 组最小差异对。
- 修正 120 条集及其子集中 15 条 `why`："与上一条构成最小差异对"改为写明配对样例的 ID（文件顺序已打乱，"上一条"指向错误）。只改了 `why`，其他字段未动。

**代码结构（行为不变）**
- 新增 `app/pipelines/` 包：每条 Pipeline 都是 `run(runner, english, chinese, model, verifier, mode)`。`Checker`（移到 `app/checker.py`）只负责选择 Pipeline；各 Pipeline 通过 `StageRunner` 调用模型，不再把整个 Checker 传进去、调用它的私有方法。
- 原来的 `pipeline.py`、`paragraph_pipeline.py`、`direct_pipeline.py`、`relation_direct.py` 拆成 `pipelines/` 下的 `paragraph`、`quote`、`direct`、`hardened`、`hardened_rules`、`paragraph_checks`、`common`；模块之间不再导入私有函数，也不再有循环导入。
- 拆分超长函数（`ChatClient.structured`、抽取与争议处理阶段）；枚举类配置统一在 `Settings` 创建时校验。
- direct_hardened 中按具体样例写的规则在 `hardened_rules.py` 中标注为 CASE-SPECIFIC（规则本身未改，见 [docs/project_analysis.md](docs/project_analysis.md) 4.1 节）。
- 测试按被测行为重新命名，共用的 Fake 与样例回答移到 `tests/helpers.py`。
- 全项目使用 `ruff format`，CI 增加格式检查。
- 验证方法：用脚本化假模型在 296 条数据集样例 × 7 种配置上运行（10,952 次），重构前后的判决、调用顺序、提示词与诊断信息逐字节一致。

**工程**
- 加入 git 版本管理、ruff lint 配置和 GitHub Actions CI。

## v0.2.16 — 后端兼容
OpenAI-compatible 后端隔离与自定义 Adapter；主模型和复核模型可以分别配置 API 地址和密钥；可配置的原生 JSON Schema 协商（`LM_SCHEMA_MODE`）；`/models` 发现可设为可选或关闭。见 [docs/backends.md](docs/backends.md)。

## v0.2.15 — direct_hardened
数学 Unicode 比较符归一化；例外条款按具体实体引用；JSON Schema 中枚举允许的证据 ID；模型给出无效引用 ID 时，对比较方向做严格受限的纯原文恢复。

## v0.2.14 — direct_relations
区分"同一事件/指代"和"同一决定/取值"；为语义相同的时长改写加了窄范围的标量等价证明。

## v0.2.13 — direct_questions
整段双模型判断之后，由程序指定固定的、可对照原文核查的问题（数值边界、包含、例外、比较方向）。

## v0.2.12 — direct_verified
原文片段锚定的命题核查，片段由模型选择、程序核对。

## v0.2.11 — direct_grounded
Paragraph-Direct 加上原文证据复核。

## v0.2.10 — direct
新架构预览：两个模型直接比较完整原文段落，不做提取。

## v0.2.9.2
保守的双子句原文锚点 Fast Path；"初判 FAIL / 复核 PASS / 仲裁 FAIL"的冲突组合返回人工复核。

## v0.2.9.1
生成前单句 Fast Path。

## v0.2.9
通用单位和比较边界处理；原文层面的保守数值保护；短句提取去膨胀。

## v0.2.8 — 可靠性与泛化
截断或空回答时最多一次受限的恢复生成；请求级 token / 耗时追踪；源引用结构审计；人工审核 CSV；冻结 36 条独立泛化验证集（真实评测：历史回归 11/11，holdout 34/36）。

## v0.2.7.5
1:N 数值等价证书恢复与截断诊断。

## v0.2.7.4
争议复核；无法安全裁定时返回 `REVIEW_REQUIRED`，不再误报为执行 ERROR。

## v0.2.7.3
兼容 LM Studio 在 Structured Output 模式下把完整 JSON 放进 `reasoning_content` 的情况。

## v0.2.7.2
约束守卫（如 `6 weeks = 42 days`）与本地回归脚本。

## v0.2.7.1
修正段落模式下重试提示和源 ID 示例。

## v0.2.7 — 段落锚定
按空行切分段落并分配不可变 ID；模型只引用段落 ID，证据由程序从原文恢复。

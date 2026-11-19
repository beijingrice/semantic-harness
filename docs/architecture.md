# 项目架构、用法与工作原理

本文完整介绍 Bilingual Keyword Checker：它解决什么问题、怎么用、内部由哪些部分组成，以及一次检查从输入到判决的每一步是怎么做的。

想马上跑起来，看第 2 节；想了解原理，从第 3 节开始读；想改代码，第 9、10 节是模块地图和扩展指南。

## 目录

1. [它解决什么问题](#1-它解决什么问题)
2. [怎么用](#2-怎么用)
3. [整体架构](#3-整体架构)
4. [核心设计原则](#4-核心设计原则)
5. [extraction 流程详解（默认）](#5-extraction-流程详解默认)
6. [direct_hardened 流程详解（实验）](#6-direct_hardened-流程详解实验)
7. [模型调用层：结构化生成、校验与重试](#7-模型调用层结构化生成校验与重试)
8. [怎么读检查结果](#8-怎么读检查结果)
9. [模块地图](#9-模块地图)
10. [扩展与修改指南](#10-扩展与修改指南)
11. [局限与注意事项](#11-局限与注意事项)

---

## 1. 它解决什么问题

学术规范、投稿须知、学位论文要求等文本常常同时有中文版和英文版。两个版本一旦不一致，读者就可能被误导，比如：

| 类型 | 英文 | 中文 | 问题 |
|---|---|---|---|
| 列举缺项 | The abstract must include purpose, methods, and results. | 摘要须介绍研究目的和研究方法。 | 中文漏了 "results" |
| 数值边界 | The response rate must be more than 10%. | 回复率须不低于 10%。 | `>` 和 `≥` 不同 |
| 例外范围 | A Chinese abstract is optional unless the program requires it. | 无论项目是否另有规定，中文摘要都只是可选材料。 | 英文有例外，中文没有 |
| 比较方向 | The baseline outperformed the new system. | 新系统表现优于基线方法。 | 胜者反了 |
| 等价改写 | The survey ran for six weeks. | 调查持续 42 天。 | **没有问题**，6 周 = 42 天 |

这个工具读入一段英文和对应的中文，判断两者的**关键事实**是否一致，并给出能追溯到原文段落的证据。它关心的是要求、数值、条件、范围这类事实，不评判翻译文笔。

最后一行很重要：好的检查工具既要抓出真正的差异，也不能把合理的改写误报成错误。项目里很多设计都是为了同时做到这两点。

---

## 2. 怎么用

### 2.1 安装

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env      # 首次安装时运行；已有 .env 不要覆盖
python -m pytest -q       # 离线测试，不会调用模型
```

### 2.2 准备模型

工具通过 OpenAI-compatible 接口调用大模型。本地开发通常用 LM Studio：

1. 在 LM Studio 中加载**两个不同的模型**：一个做主模型，一个做独立复核模型。开发时用的组合是 Qwen3.5-9B + Gemma 4 12B QAT。
2. 打开 Developer → Start Server，默认端口 1234。
3. 查出模型 ID，填进 `.env`：

```bash
curl -s http://127.0.0.1:1234/v1/models | python3 -m json.tool
```

```dotenv
LM_BASE_URL=http://127.0.0.1:1234/v1
LM_MODEL=qwen/qwen3.5-9b
LM_VERIFIER_MODEL=google/gemma-4-12b-qat
LM_JSON_SCHEMA=true
```

接云端 API、主次模型分别接不同服务、或者服务不支持 OpenAI 协议，见 [backends.md](backends.md)。

### 2.3 三种使用方式

**Web 界面**：适合人工逐条检查。

```bash
uvicorn app.api:app --host 127.0.0.1 --port 8005
```

浏览器打开 `http://127.0.0.1:8005/`，贴入中英文、选好模型，点"开始检测"。结果页会列出双方提取出的关键内容、对应关系、复核和仲裁意见，以及执行诊断，还可以导出 JSON。

**命令行**：适合脚本和批处理。

```bash
bilingual-check --en 'The abstract must include purpose, methods, and results.' \
  --zh '摘要须介绍研究目的和研究方法。' --mode dual

# 文本较长时从文件读取
bilingual-check --en-file en.txt --zh-file zh.txt
```

输出完整的 JSON 结果。退出码：PASS = 0，FAIL = 1，执行错误或配置错误 = 2，REVIEW_REQUIRED = 3。

**JSON API**：适合集成到其他系统。接口文档在 `http://127.0.0.1:8005/docs`。

```bash
curl -s http://127.0.0.1:8005/api/check \
  -H 'Content-Type: application/json' \
  -d '{"english": "The survey ran for six weeks.", "chinese": "调查持续 42 天。", "mode": "dual"}'
```

请求字段：`english`、`chinese`（必填，各自最长 6000 字符），`model`、`verifier_model`（可选，覆盖 `.env`），`mode`（`dual` 或 `primary`，默认 `dual`）。

| HTTP 状态 | 含义 |
|---|---|
| 200 | 正常返回判决：PASS / FAIL / REVIEW_REQUIRED |
| 422 | 输入不合法，比如文本为空 |
| 500 | 服务端配置错误，`detail` 会写明是哪个配置项 |
| 502 | 模型调用失败或模型输出始终不合格，即 ERROR，**不是**业务判定 |

### 2.4 两种模式

- **`dual`（默认，推荐）**：主模型完成初判，另一个模型只看原文、独立复核；两者意见不一致时再进入仲裁。
- **`primary`**：只用主模型。速度快、调用次数少，但没有独立复核，质量保障明显更弱。只有一个模型可用时才用它。

### 2.5 两种 Pipeline

`.env` 里的 `LM_PIPELINE` 决定用哪条检查流程：

- **`extraction`（默认）**：先分别从中英文中提取事实，再逐条配对、比较。原理见第 5 节。
- **`direct_hardened`（实验）**：两个模型直接比较完整段落；程序发现风险点时，再针对性地提问和核对。原理见第 6 节。

### 2.6 输入格式

- 纯文本。**空行**是段落分隔符，单个换行仍算同一段（这样 PDF 复制出来的软换行不会把一句话切断）。
- 中英文段落不要求一一对应，顺序也可以不同。
- DOCX、PDF、HTML 需要先在上游转成纯文本，并用空行保留段落结构。

### 2.7 批量评测

```bash
python -m examples.evaluate --cases examples/v2_regression_11.json \
  --mode dual --stop-after-errors 1 --out eval_regression_11.json
```

评测报告会统计四种结果的数量、False PASS（漏报）、False FAIL（误报）、端到端正确率、各类别表现、耗时和 token 用量。常用参数：

| 参数 | 作用 |
|---|---|
| `--pipeline direct_hardened` | 评测另一条流程 |
| `--limit N` / `--category X` | 只跑前 N 条，或只跑某一类 |
| `--stop-after-errors 1` | 出现第一个 ERROR 就停，避免浪费时间 |
| `--baseline-report old.json` | 和一份历史报告逐条对比 |
| `--audit-csv audit.csv` | 导出人工核查表 |

各数据集的用途和状态见 [examples/DATASETS.md](../examples/DATASETS.md)，标注规则见 [labeling_guidelines.md](labeling_guidelines.md)（草稿）。

---

## 3. 整体架构

### 3.1 分层

```mermaid
flowchart TB
    subgraph 入口层
        WEB["Web UI<br/>static/index.html + app.js"]
        API["JSON API<br/>api.py"]
        CLI["命令行<br/>cli.py"]
        EVAL["批量评测<br/>examples/evaluate.py"]
    end

    CFG["配置<br/>config.py: Settings"]

    subgraph 检查层
        CHK["Checker<br/>checker.py：选择 Pipeline"]
        EXT["extraction 流程<br/>pipelines/paragraph.py"]
        DIR["direct_hardened 流程<br/>pipelines/hardened.py"]
    end

    subgraph 程序校验层["程序校验层（不调用模型）"]
        PAR["段落与不可变 ID<br/>paragraphs.py"]
        FAST["Fast Path<br/>extraction_fast_path.py"]
        VAL["数值 / 计入守卫<br/>validators.py"]
        DIS["数值等价证书<br/>dispute_resolution.py"]
        RISK["原文风险扫描<br/>source_risks.py"]
    end

    subgraph 模型调用层
        LLM["ChatClient<br/>llm.py：Schema 校验、重试、遥测"]
        BK["Backend Adapter<br/>backends.py"]
    end

    MODEL[("大模型服务<br/>LM Studio / 云 API")]

    WEB --> API
    API --> CHK
    CLI --> CHK
    EVAL --> CHK
    CFG -.-> CHK
    CFG -.-> LLM
    CHK --> EXT
    CHK --> DIR
    EXT --> PAR & FAST & VAL & DIS
    DIR --> PAR & RISK & VAL
    EXT --> LLM
    DIR --> LLM
    LLM --> BK --> MODEL
```

各层职责：

| 层 | 负责 | 不负责 |
|---|---|---|
| 入口层 | 接收输入、调用 `Checker`、把结果转成 HTTP 响应、JSON 或退出码 | 任何判断逻辑 |
| 检查层 | 编排检查步骤，根据各步结果合成最终判决 | 网络通信 |
| 程序校验层 | 纯 Python 的确定性检查：分段、ID 校验、数值解析、风险扫描 | 理解语义（交给模型） |
| 模型调用层 | 发请求、解析 JSON、按 Schema 校验、失败重试、记录遥测 | 判决（这一层不知道 PASS/FAIL 是什么意思） |

### 3.2 一次请求的生命周期

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as api.py / cli.py
    participant C as Checker
    participant P as Pipeline
    participant L as ChatClient
    participant M as 模型服务

    U->>A: 英文 + 中文 + 模式
    A->>A: Settings() 读取并校验配置（填错时报 ConfigError）
    A->>C: Checker(client, settings).check(...)
    C->>P: pipelines.select 按配置选择 extraction 或 direct_hardened
    loop 每个需要模型的步骤
        P->>L: structured(模型, Prompt, Schema, 校验函数)
        L->>M: POST /chat/completions
        M-->>L: JSON 文本
        L->>L: 解析 → Pydantic 校验 → 原文 ID 校验
        L-->>P: 合格的结构化结果（否则重试，最终抛出 LLMError）
    end
    P->>P: 程序校验 + 合成判决
    P-->>A: CheckResult
    A-->>U: 200 + 结果；或 502 / 500 / 422
```

### 3.3 三类错误，三种去向

项目严格区分"检查出了问题"和"检查本身出了问题"：

| 异常 | 定义位置 | 何时出现 | API | CLI |
|---|---|---|---|---|
| `ConfigError` | `config.py` | 配置值无效，比如 `LM_MAX_TOKENS=abc`、未知的 `LM_PIPELINE` | 500 | 退出码 2 |
| `LLMError` 及其子类 | `llm.py` | 模型不可达、输出不是 JSON、多次重试仍不符合 Schema 或原文校验 | 502 | 退出码 2 |
| `BackendError` | `backends.py` | 调用时的传输错误会被包装成 `LLMError`（502）；Adapter 加载失败时按配置错误处理（500） | 502 / 500 | 退出码 2 |

这些都**不会**变成 FAIL。把"模型挂了"当成"文本有问题"，是这类工具最容易犯的错误。

---

## 4. 核心设计原则

理解了下面几条原则，后面的流程细节就都说得通了。

### 原则 1：原文不可变，模型只引用段落 ID

程序先把原文切成段落，每段分配一个 ID（`en_p1`、`zh_p2`……），并记录它在原文中的起止位置，保证 `原文[start:end] == 段落文本`。

模型在输出中**只能引用这些 ID**，不能自己抄写引文。展示给用户的证据，由程序根据 ID 从原文中取出。

**为什么**：早期版本让模型直接引用原文片段，结果模型常常"引用"出原文里根本没有的话，或者在中英混排、弯引号、空格上抄错。改成引用 ID 之后，证据一定真实存在。

**边界**：ID 有效只能证明"这段话确实存在"，不能证明模型对这段话的理解是对的。

### 原则 2：模型提议，程序核对

凡是程序能确定检查的东西，都不交给模型拍板：

- 模型说"中文缺了某项"，必须给出对应的英文段落 ID，程序核对 ID 存在。
- 模型说"两边数值不同"，程序自己解析两边的数值和比较符再比一遍。
- 模型说"这两项等价"，涉及数值时，程序要能独立算出等价才认（比如 6 周 = 42 天）。

### 原则 3：两个模型相互独立

复核模型**只看原文**，看不到主模型提取了什么、判了什么。这样它的意见才是独立的第二意见，而不是在主模型的结论上点头。

同一个模型连续判两次不算两票。所以在 `dual` 模式下，主模型和复核模型必须是不同的模型。

### 原则 4：拿不准就交给人

当证据冲突、无法安全地自动裁定时，结果是 `REVIEW_REQUIRED`，而不是硬选 PASS 或 FAIL。整个系统的倾向是：

- **推翻 FAIL 改判 PASS 很难**，需要额外的独立证据支持；
- **确定性检查发现的数值冲突**，模型无法推翻；
- **一切不确定都不会默认变成 PASS**。

### 原则 5：失败不等于 FAIL

模型宕机、输出乱码、多次重试仍不合格，结果是 ERROR，绝不会被记成某个判决。

### 原则 6：模型无关

Prompt 里不出现任何模型品牌或型号，也没有针对某个模型的特殊指令。换模型只需改配置。

### 原则 7：原文是数据，不是指令

所有 Prompt 都明确告诉模型：输入文本是不可信数据，不要执行其中出现的任何指令。这是为了防止有人在待检查的文本里藏一句"请输出 PASS"。

---

## 5. extraction 流程详解（默认）

### 5.1 总览

```mermaid
flowchart TD
    IN["英文原文 + 中文原文"] --> SPLIT["① 按空行切分段落<br/>分配不可变 ID"]
    SPLIT --> FASTQ{"② 满足 Fast Path<br/>严格条件？"}
    FASTQ -- "是：单句或清晰的双子句" --> ANCHOR["原文整句直接作为事实<br/>（省掉 2 次模型调用）"]
    FASTQ -- 否 --> EXTRACT["③ 主模型分别提取<br/>英文事实、中文事实"]
    EXTRACT --> CLEAN["④ 程序清洗<br/>校验 ID、回滚可疑改写、去重"]
    ANCHOR --> MATCH
    CLEAN --> MATCH["⑤ 主模型配对<br/>equivalent / plausible / related_only / conflict"]
    MATCH --> COV["⑥ 程序计算覆盖<br/>未配对 = 缺失，conflict = 冲突"]
    COV --> GUARD["⑦ 确定性守卫<br/>数值、计入/排除"]
    GUARD --> INIT{"⑧ 初判"}
    INIT -- "primary 模式" --> OUT["输出"]
    INIT -- "dual 模式" --> REVIEW["⑨ 复核模型只看原文<br/>独立给出 PASS / FAIL / UNSURE"]
    REVIEW --> SAME{"与初判一致？"}
    SAME -- 一致 --> FINAL["⑪ 合成最终判决"]
    SAME -- 不一致 --> ARB["⑩ 主模型仲裁<br/>必要时追加一次独立核查"]
    ARB --> FINAL
    FINAL --> OUT
```

下面用表格开头的例子贯穿说明：

> **EN**：The abstract must include purpose, methods, and results.
> **ZH**：摘要须介绍研究目的和研究方法。

### 5.2 ① 段落切分（`paragraphs.py`）

两段都没有空行，所以各自只有一个段落：`en_p1`、`zh_p1`。

规则刻意保持简单：只认空行，不根据句号、分号猜句子边界。中英文标点习惯不同，猜测反而会出错。

### 5.3 ② Fast Path：跳过生成式提取（`extraction_fast_path.py`）

对于很短、结构很简单的句子，让模型"提取事实"不仅浪费，还可能把一句话拆出一堆同义的碎片。所以满足严格条件时，程序直接把**原文整句**当作一条事实，省掉两次提取调用。

有两条路径：

- **单句路径**：两边各自只有一个段落、一个完整句子，长度在限制内（英文 8～180 字符、中文 4～90 字），带有情态词或数字之类的线索，并且**没有**并列连词、多个情态词、列举标志、分号冒号等复杂结构。
- **双子句路径**：两边各自恰好是两个可以清楚分开的子句，子句之间没有条件、例外、转折关系，也不含数字。

任何一条不满足就放弃捷径，走正常提取。**宁可多花一次调用，也不冒漏掉并列要求的风险。**

例子中英文有 "and" 和两个逗号（并列列举），不满足条件，所以走正常提取。

另外，Fast Path 只是省掉提取这一步，后面的配对、独立复核、确定性守卫一步都不少。

### 5.4 ③ 事实提取（主模型）

主模型分别读英文和中文，输出事实清单。每条事实包含：

```json
{"kind": "enumerated_item", "claim": "The abstract must include results", "paragraph_ids": ["en_p1"]}
```

- `kind`：concept / enumerated_item / entity / constraint / relation 五选一。
- `claim`：用一句话概括的事实，最长 220 字符。
- `paragraph_ids`：这条事实出自哪个段落，**只能**从程序给出的 ID 里选。

默认的 coarse 提取策略不强制把列表拆成单项，一个完整的列举可以作为一条事实。这是在多次评测后选定的做法：强制拆分会让事实数量膨胀，配对也更容易出错。

例子的典型结果：英文提取出"摘要须包含目的、方法、结果"相关的事实，中文提取出"摘要须介绍目的和方法"相关的事实。

如果某一边一条事实都没提取出来，会换一个提示词再试一次（`dual` 模式下改用复核模型来做）；短文本还提取不出来时，就把原文整段作为一条事实，交给后面的步骤判断。**空清单永远不会直接导致 PASS。**

### 5.5 ④ 程序清洗

模型返回后，程序在接受结果前会做几件事：

1. **校验 ID**：每条事实引用的段落 ID 必须存在、不重复，否则视为不合格输出并要求模型重试。
2. **回滚可疑改写**（`pipelines/common.py` 的 `clean_unsupported_claim`）：如果 `claim` 里出现了原文中没有的强制语气（must、必须）、排他词（only、仅）、否定词或数字，说明模型可能"添油加醋"了。引用的原文不超过 220 字符时，直接用原文替换这条 `claim`；更长时只在诊断信息里记一笔。
3. **精确去重**：`claim` 和引用段落完全相同的重复项会被丢弃。只做精确去重，不做"看起来意思差不多"的模糊合并，以免把两条不同的要求误合成一条。
4. **防止提取膨胀**：单段短句被提取出 9 条以上事实时，直接退回成一条原文整句。另外，如果已经有一条事实就是原文整句，那些只是从这句话里截出来的名词片段会被去掉。

### 5.6 ⑤ 配对（主模型）

主模型拿到双方的事实清单和两段原文，输出配对关系：

```json
{"links": [
  {"en_ids": ["en_1"], "zh_ids": ["zh_1"], "relation": "conflict", "reason": "English requires results; Chinese omits them"}
]}
```

四种关系：

| relation | 含义 | 计入覆盖？ |
|---|---|---|
| `equivalent` | 意思相同 | 是 |
| `plausible` | 大致对应，表述有差别但不影响事实 | 是 |
| `related_only` | 话题相关，但不能算对应 | 否 |
| `conflict` | 指同一件事，但内容矛盾 | 记为冲突 |

一条配对两边都可以包含多条事实。程序会检查所有 ID 都真实存在，否则要求重试。

### 5.7 ⑥ 覆盖计算（程序）

`pipelines/common.py` 的 `match_coverage` 根据配对结果算出三张清单：

- **中文缺失**：没有被 `equivalent` / `plausible` 配上、也不在冲突里的英文事实；
- **英文缺失**：反过来；
- **冲突**：所有 `conflict` 配对。

### 5.8 ⑦ 确定性守卫（`validators.py`，程序）

这一步不调用模型，只检查程序能明确解析的形式化性质：

**数值守卫**：对一对一配上的事实，各自解析出"比较符 + 数值 + 单位"，再换算成统一单位后比较：

| 维度 | 支持的单位（部分） |
|---|---|
| 时长 | 分钟、小时、天、周（统一换算成天）；年单独一类 |
| 百分比 | %、percent、百分比 |
| 页数 | pages、页 |
| 长度 | mm、cm、m、毫米、厘米、米（统一换算成毫米） |
| 人数 | participants、respondents、名参与者、人 |
| 字号 | points、pt、磅 |

比较符包括 `<`、`<=`、`>`、`>=`、`=`。能识别的写法举例：

- 数字前的比较词：at least、more than、exceed、above、no more than、must not exceed、不少于、不超过、超过；以及中文里数字前隔着动词的写法（如"至少保留五天"）。
- 数字或单位后的界限词：以上、以下、or more、or fewer（如"10%以上""10 页以上""30 pages or more"）。
- 只对**时间**生效的上限：within、no later than、内、以内、之内（如"within 6 weeks""七天内"）。对页数等其他单位，这些词常常表示"计入某个限额"（如"封面计入30页内"），所以不当作上限。

数字前的比较词优先于单位后的界限词，例如"少于五天内"按 `<` 处理。没有任何比较词的陈述（如"持续了 42 天"）按"恰好等于"处理。两端数值和开闭都明确的区间会作为整体比较。

所以：

- `six weeks` 和 `42 天` → 都是 `= 42 天`，**一致**，不报错；
- `more than 10%` 和 `不低于 10%` → `> 10` 对比 `>= 10`，**冲突**。

> **注意**：没有被识别的比较写法会按"恰好等于"处理。如果另一边用的是能识别的写法，守卫就会误报冲突，而守卫有一票否决权，结果会被强制判成 FAIL。目前已知还没覆盖的写法包括：非时间单位后的"以内"（如"10%以内"）。新增写法时请在 `tests/test_numeric_bounds.py` 中同时补上"应当一致"和"应当冲突"的例子。

为了不误报，守卫遇到下面这些情况会**主动放弃判断**：一句话里有多个数字、带"大约"之类的约数词、单位无法识别、出现多个互相矛盾的比较符。放弃判断不等于 PASS，只是不出手。

**计入/排除守卫**：两边都明确谈到"计入/不计入"（页数限制、字数统计等），而一边是肯定、另一边是否定时，报告疑似反转。

**原文兜底**：上面两项都没发现问题，且两边都是单句、并且已被配对为等价时，再直接拿两段原文比一次数值。

守卫发现的问题有**一票否决权**：最终判决一定是 FAIL，模型无法推翻。

### 5.9 ⑧ 初判

有任何缺失、冲突或守卫问题，初判就是 FAIL，否则是 PASS。

例子中，中文缺了 "results"，所以初判是 **FAIL**。`primary` 模式到这里就结束了。

### 5.10 ⑨ 独立复核（复核模型）

复核模型**只拿到两段原文和段落列表**，看不到主模型的事实清单、配对和初判。它独立给出：

```json
{"verdict": "FAIL", "reason": "...",
 "issues": [{"kind": "missing_in_chinese", "en_paragraph_ids": ["en_p1"], "zh_paragraph_ids": [], "reason": "results not mentioned"}]}
```

程序会校验它的输出是否自洽：PASS 不能带问题项；FAIL 必须至少有一个问题项；"中文缺失"必须给出英文来源段落，"冲突"必须两边都给。

例子中复核也判 FAIL，和初判一致，**最终结果 FAIL**，不需要仲裁。

### 5.11 ⑩ 仲裁：意见不一致时怎么办

初判和复核不一致时，由主模型做仲裁（初判 FAIL 且已有守卫问题时跳过这一步，因为结果反正是 FAIL）。仲裁模型能看到原文、初判依据和复核意见。

之后按下表合成结果：

| 初判 | 复核 | 仲裁 | 额外步骤 | 最终结果 |
|---|---|---|---|---|
| FAIL | PASS 或 UNSURE | PASS | 复核模型再做一次**独立支持核查**，必须确认每个争议点都有原文支持 | 确认 → PASS；不确认 → ERROR |
| PASS | FAIL（附具体问题） | PASS | 复核模型**逐条回应**原复核的每个问题，程序同时生成数值等价证书 | 所有问题都被反驳**且**有形式化数值证书 → PASS；否则 REVIEW_REQUIRED |
| FAIL | PASS | FAIL | — | REVIEW_REQUIRED |
| PASS | 任意 | UNSURE | — | REVIEW_REQUIRED |
| FAIL | 任意 | UNSURE | — | FAIL |
| 其他组合 | | | — | 采用仲裁结论 |

最后无论前面结果如何，**只要确定性守卫报了问题，最终就是 FAIL**。

这张表背后的思路：

- **第 1 行：推翻 FAIL 要更多证据。** 仲裁说 PASS 还不够，还要复核模型独立确认，并且引用的段落必须覆盖所有原本有争议的事实。
- **第 2 行：两次 PASS 不能压过一条有依据的异议。** 主模型初判 PASS、仲裁又判 PASS，本质上还是同一个模型。复核模型提出的具体异议，只有被逐条反驳，并且有程序能独立验证的数值等价证书（比如"6 周和 42 天确实相等"）时，才会被驳回。
- **第 3 行：同一个模型判两次 FAIL 不算两票。** 独立复核认为没问题，而主模型初判和仲裁都说有问题，这时宁可交给人判断，也不轻易定为 FAIL（真正的数值冲突由守卫兜底）。

### 5.12 一次检查要调用几次模型

| 场景 | 调用次数 |
|---|---|
| `primary`，走 Fast Path | 1（配对） |
| `primary`，正常提取 | 3（英文提取、中文提取、配对） |
| `dual`，意见一致，走 Fast Path | 2（配对、复核） |
| `dual`，意见一致，正常提取 | 4 |
| `dual`，需要仲裁 | 再加 1～2 次 |

以上是每一步都一次成功的情况。某一步输出不合格时会重试，见第 7 节。

---

## 6. direct_hardened 流程详解（实验）

### 6.1 思路

extraction 流程的风险在于：提取和配对每一步都可能出错，错误还会层层传递。direct 系列换了一种思路：**不提取，让两个模型各自直接比较完整原文**；程序扫描原文，在已知的高风险点上追问模型、再自己核对。

### 6.2 流程

```mermaid
flowchart TD
    IN["英文原文 + 中文原文"] --> SPLIT["切分段落"]
    SPLIT --> J1["主模型：比较完整段落"]
    SPLIT --> J2["复核模型：独立比较完整段落"]
    J1 --> NG{"原文数值守卫<br/>发现冲突？"}
    J2 --> NG
    NG -- 是 --> FAIL1["FAIL"]
    NG -- 否 --> Q{"程序扫描原文风险<br/>+ 两个模型是否分歧"}
    Q -- "没有风险、没有分歧" --> KEEP["保留两个模型的结论<br/>（一致则采用，有 UNSURE 或分歧则 REVIEW_REQUIRED）"]
    Q -- "有风险或分歧" --> ASK["对每个风险点提一个固定问题<br/>（最多 3 个）"]
    ASK --> ANSWER["模型从程序给出的原文片段中选证据<br/>并填写结构化答案"]
    ANSWER --> PROVE["程序自己核对证据和答案"]
    PROVE --> FINAL["汇总：任一项证实冲突 → FAIL<br/>全部证实一致 → PASS<br/>否则 → REVIEW_REQUIRED"]
```

### 6.3 第一步：两个独立判断（`pipelines/direct.py` 的 `run_direct`）

两个模型拿到完全相同的输入（两段原文和段落列表），各自给出 PASS / FAIL / UNSURE 和问题清单。合成规则很简单：

- 原文数值守卫发现冲突 → FAIL；
- 任一方 UNSURE → REVIEW_REQUIRED；
- 两者一致 → 采用；
- 两者不一致 → REVIEW_REQUIRED（**不引入第三个模型投票**，因为第三票往往和其中一方的错误相关）。

### 6.4 第二步：程序选题（`pipelines/hardened_rules.py` 的 `choose_questions`）

程序扫描原文（`source_risks.py`），并结合两个模型是否有分歧，决定要不要追问、问什么：

| 触发条件 | 固定问题 | 程序如何核对 |
|---|---|---|
| 两边各有一个可解析的数值约束，但比较符或数值不同 | 数值边界：两边的比较符、数值、单位分别是什么？ | 自己解析原文数值，模型给的值必须和原文一致；能识别 `≥`、`≧`、全角数字等 Unicode 写法 |
| 两边"计入"类表述的数量对不上 | 计入实体：哪个文档组成部分在页数/字数限制中的计入规则不同？ | 自己在原文中定位实体和计入规则。两边必须指向**同一个**限额：原文各自只有一个限额数值且相同；被引用的英文规则要么直接写出该数值（"25-page limit"），要么用原文绑定到该数值的范围名（"main-report limit"），要么用"this limit"且原文没有其他范围的限额。指向其他限额（如"appendix limit"）时放弃判断 |
| 模型有分歧，英文有 unless，中文有"无论/不论/除非" | 例外范围：同一组成部分，是否一边有条件、一边总是可选？ | 要求模型给出**具体实体名**，程序把证据收窄到提到该实体的那一句；找不到唯一目标就放弃。实体对应目前只覆盖一小张内置表（如 Chinese abstract / 中文摘要、appendix / 附录） |
| 模型有分歧，涉及 outperformed / 优于 等比较 | 比较方向：同一数据集上，两边的胜者分别是谁？ | 自己从原文中判断胜者；模型给的证据 ID 无效时，若原文中能唯一定位，程序可以自行完成核对 |
| 两个模型都判 FAIL，而原文只是同一个量的不同单位写法 | 标量等价：是否是同一个量的不同单位，且没有额外义务？ | 两边都必须是单句、可完整解析，且不含其他义务性表述 |

| 模型有分歧，但不属于以上类型 | 通用问题：找出被指称的具体差异 | 程序无法形式化核对，结果固定为 REVIEW_REQUIRED，所以**不调用模型**，直接记为跳过 |

每次最多提 3 个问题。没有风险也没有分歧时，不额外调用模型，保留第一步的结论。

`primary` 模式下没有复核模型，也不追问：只要原文扫描出风险，PASS 就会被降为 REVIEW_REQUIRED。

### 6.5 第三步：结构化追问与程序核对

每个问题单独调用一次模型（上表最后一行的通用问题除外）。模型必须从程序列出的原文片段中选择证据（`en_all` 表示整段，`en_s1`、`en_s2`… 表示按标点切开的子句），并填写主体、条件、数值、是否同一事件、结论等字段。JSON Schema 里直接枚举了本次允许的证据 ID。

程序**不采信模型的结论字段**，而是根据模型选出的原文片段，用自己的规则重新核对。只有程序能独立证实时才给出 FAIL 或 PASS，其余都是 REVIEW_REQUIRED。

为了控制成本，每个追问都不做修正重试（输出被截断时仍会有一次恢复请求），输出不合格就按 REVIEW_REQUIRED 处理。

### 6.6 适用场景

direct_hardened 在 11 条回归集上表现最好（9 条正确、2 条转人工、0 误判），但它的追问规则是针对已知风险类型写的，覆盖面比 extraction 窄。目前仍是实验模式，建议和 extraction 对照着用。

---

## 7. 模型调用层：结构化生成、校验与重试

所有模型调用都通过 `ChatClient.structured()`（`llm.py`）进行。调用方传入模型、Prompt、Pydantic Schema 和一个可选的校验函数，拿回的要么是**完全合格**的结构化结果，要么是一个 `LLMError`。

### 7.1 一次调用的处理过程

```mermaid
flowchart TD
    REQ["组装请求<br/>temperature=0，按配置决定是否带 JSON Schema"] --> SEND["发送到 Backend"]
    SEND --> RESP{"响应状态"}
    RESP -- "被截断 / 内容为空" --> REC["恢复请求（最多 1 次）<br/>去掉 Schema，要求直接给出完整 JSON<br/>被截断时改用 LM_MAX_TOKENS 预算"]
    RESP -- "明确不支持 JSON Schema<br/>且 LM_SCHEMA_MODE=auto" --> DOWN["降级为纯 Prompt 模式（只降 1 次）<br/>记住该模型不支持"]
    RESP -- 正常 --> PARSE["提取 JSON → Pydantic 校验 → 调用方校验函数"]
    DOWN --> PARSE
    REC --> PARSE2["同样的校验，不再重试"]
    PARSE -- 合格 --> OK["返回结果"]
    PARSE2 -- 合格 --> OK
    PARSE -- "不合格，未满 3 次" --> FIX["把错误原因和 Schema 发回给模型<br/>要求重新生成完整答案"]
    FIX --> SEND
    PARSE -- "第 3 次仍不合格" --> ERR["LLMError → ERROR"]
    PARSE2 -- 不合格 --> ERR
```

要点：

- **三层校验**：能解析成 JSON → 符合 Pydantic Schema（字段、类型、取值、长度）→ 通过调用方的业务校验（ID 存在、FAIL 必须带证据等）。
- **修正重试最多 3 次**：每次都把具体的错误原因发回给模型，并按不同步骤附上对应的格式提示。
- **截断恢复最多 1 次**：输出被 `max_tokens` 截断或内容为空时，换一种问法再要一次，结果同样要通过全部校验。如果是**被截断**，这次恢复请求的上限会提高到 `LM_MAX_TOKENS`：提取（`LM_EXTRACTION_MAX_TOKENS`）和 direct_hardened 追问（1536）用的是更小的预算，推理型模型可能把它全部花在推理上，用同样的上限重试几乎必然再次截断。
- **网络或 HTTP 错误不自动重试**，直接报错。
- **JSON 提取比较宽容**：允许外面包着代码块、`<think>` 标签或少量说明文字，但不会去修补模型写坏的 JSON。
- **LM Studio 兼容**：部分模型在 Schema 模式下把完整 JSON 放进 `reasoning_content`、`content` 却是空的。只有当 `reasoning_content` 本身恰好是一个完整 JSON 对象时，才会取用它。

### 7.2 JSON Schema 模式（`LM_SCHEMA_MODE`）

| 值 | 行为 |
|---|---|
| `legacy`（默认） | 沿用旧开关：`LM_JSON_SCHEMA=true` 时带上 Schema |
| `on` | 一定带 Schema；服务端拒绝时报错 |
| `off` | 从不带 Schema，只在 Prompt 里要求输出 JSON |
| `auto` | 先带 Schema；只有当服务端以 400/415/422 明确回复"不支持 response_format/json_schema"时，才降级一次，并记住这个模型不支持 |

无论哪种模式，返回结果都要经过同样的 Pydantic 和原文校验。Schema 只是让模型更容易输出合格的格式，而不是信任模型的理由。

### 7.3 Backend Adapter（`backends.py`）

`ChatClient` 不直接发 HTTP 请求，而是交给 Backend：

```python
class LLMBackend(Protocol):
    def complete(self, payload: dict) -> BackendResponse: ...
    def models(self) -> list[str] | None: ...   # None 表示不支持查询模型列表
```

- 内置的 `OpenAIChatBackend` 调用 `/chat/completions` 和 `/models`，并在读取每一层响应前先检查类型，提供商返回的结构异常时统一抛出 `BackendError`。
- 主模型和复核模型可以分别用不同的地址、密钥、甚至不同的 Adapter（通过 `LM_VERIFIER_*` 配置）。
- 非 OpenAI 协议的服务，可以写一个类实现上面两个方法，然后用 `LM_BACKEND=包名.模块:类名` 加载。示例见 `examples/custom_backend_example.py`，详见 [backends.md](backends.md)。

### 7.4 遥测

每次模型调用都会记录：所属步骤、模型、第几次尝试、是否为恢复请求、耗时、token 用量、`finish_reason`、结果状态（validated / schema_or_grounding_invalid / truncated / recovered_validated 等）。

**不记录**完整 Prompt、模型原始回复和 API 密钥。评测脚本用这些记录统计各步骤的耗时分布、重试和恢复次数。

---

## 8. 怎么读检查结果

`CheckResult`（`models.py`）字段较多，按用途分组：

**结论**

| 字段 | 说明 |
|---|---|
| `verdict` | 最终判决：PASS / FAIL / REVIEW_REQUIRED |
| `initial_verdict` | 主模型初判 |
| `summary` | 一句话中文总结，包括初判、复核、仲裁情况和各类问题数量 |
| `mode`, `model`, `verifier_model` | 本次用的模式和模型 |

**原文与提取**

| 字段 | 说明 |
|---|---|
| `english_paragraphs`, `chinese_paragraphs` | 切分后的段落：ID、起止位置、原文 |
| `english_keywords`, `chinese_keywords` | 提取出的事实，每条带 `paragraph_ids` 和程序取回的原文证据 `evidence` |
| `links` | 配对关系 |

**问题清单**

| 字段 | 说明 |
|---|---|
| `missing_in_chinese`, `missing_in_english`, `conflicts` | 配对后发现的缺失和冲突 |
| `deterministic_issues` | 确定性守卫发现的问题（有一票否决权） |
| `review_issues` | 复核模型提出的问题，附双方原文证据 |

**过程**

| 字段 | 说明 |
|---|---|
| `review` | 复核模型的完整意见 |
| `arbitration`, `arbitrated` | 仲裁意见、是否发生了仲裁 |
| `dispute_recheck` | 逐条反驳复核异议的结果（只在第 5.11 节表格第 2 行的情况下出现） |
| `diagnostics` | 每一步的执行记录，包括是否走了 Fast Path、是否有回滚和去重、各步骤用的模型 |

**direct_hardened 专用**

| 字段 | 说明 |
|---|---|
| `direct_primary` | 主模型的原始判断 |
| `grounded_risks` | 程序扫描出的原文风险 |
| `verification_questions` | 每个追问的结构化答案和程序核对结论 |
| `grounded_evidence_audit` | 追问时选中的原文证据片段 |

阅读建议：先看 `verdict` 和 `summary`；FAIL 时看 `deterministic_issues`、`conflicts`、`missing_*` 和 `review_issues` 找原因；REVIEW_REQUIRED 时看 `review` 和 `arbitration` 了解分歧在哪里；结果出乎意料时看 `diagnostics`。

---

## 9. 模块地图

```
app/
├── api.py                  FastAPI：/、/api/status、/api/check；错误到 HTTP 状态码的映射
├── cli.py                  命令行入口 bilingual-check；判决到退出码的映射
├── config.py               Settings：每次创建时读取环境变量；ConfigError
├── checker.py              Checker：按配置选择 Pipeline 并运行（对外入口）
├── pipelines/              调用模型的编排部分；每条 Pipeline 都是
│   │                       run(runner, english, chinese, model, verifier, mode) -> CheckResult
│   ├── base.py             StageRunner：按阶段调用模型、选择模型；PipelineStageError
│   ├── common.py           两种 extraction 共用：可疑改写回滚、配对覆盖计算、判决摘要
│   ├── paragraph.py        extraction 默认流程：提取或捷径、配对、守卫、复核、仲裁、判决合成
│   ├── paragraph_checks.py 段落 ID 回答的结构校验，以及转换成带原文的公开格式
│   ├── quote.py            extraction 的 quote 证据模式（LM_EVIDENCE_MODE=quote，A/B 用）
│   ├── direct.py           direct 第一步：两个模型独立比较完整段落
│   ├── hardened.py         direct_hardened：逐题追问与汇总
│   └── hardened_rules.py   direct_hardened 的纯原文规则：选题与证明（含标注为 CASE-SPECIFIC 的样例特化词表）
├── extraction_fast_path.py 跳过提取的两条保守捷径
├── validators.py           数值解析（单位、比较符、区间）和计入/排除守卫
├── dispute_resolution.py   数值等价证书；判断一条异议能否被形式化证据反驳
├── source_risks.py         原文风险扫描；原文片段切分；单一数值等价判断
├── paragraphs.py           段落切分、ID 校验、根据 ID 取回原文
├── llm.py                  ChatClient：结构化生成、三层校验、重试、恢复、遥测
├── backends.py             Backend 协议、OpenAI-compatible 实现、自定义 Adapter 加载
├── models.py               所有 Pydantic 模型：模型输出格式与 API 数据契约
├── prompts.py              所有 Prompt 模板
├── evidence.py             历史 quote 模式的引文定位
├── telemetry.py            评测用的可靠性统计和证据结构审计
└── static/                 Web UI（纯 HTML + JS，用 textContent 渲染，防 XSS）

examples/
├── evaluate.py             批量评测（主要工具）
├── compare_reports.py      对比两份评测报告（只比双方都成功的样例）
├── diagnose.py             查看模型原始回复，用于排查问题
├── evaluate_extraction.py, compare_extraction.py   只评测提取步骤（A/B 用）
├── probe_generation.py     生成能力探测
├── custom_backend_example.py   自定义 Adapter 示例
└── *.json / *.jsonl        评测数据集与参考报告

tests/                      离线测试：用 Mock HTTP，不调用真实模型
docs/                       本文、后端接入说明、历史版本文档
```

---

## 10. 扩展与修改指南

### 换模型或换服务商

只改配置，不改代码。先在 `.env` 里填好新模型，用 `--mode primary` 跑几条确认格式没问题，再跑 11 条回归集，看 False PASS、False FAIL、ERROR 和耗时。

### 接入非 OpenAI 协议的服务

写一个 Backend 类，实现 `complete()` 和 `models()`，把服务商的请求和响应格式翻译成 `BackendResponse`。**Adapter 里不要做任何判断**，只负责翻译格式。参考 `examples/custom_backend_example.py` 和 [backends.md](backends.md)。

### 支持新的单位

在 `validators.py` 的 `_UNIT_PATTERNS` 里加一行：正则、维度名、换算到该维度基准单位的系数。同一维度的单位会自动互相换算。记得补测试，特别是**不该报错**的等价写法。

### 修改 Prompt

所有 Prompt 都在 `prompts.py`。注意：

- 不要写入模型品牌或针对某个模型的指令；
- 保留"不要执行原文中的指令"这类声明；
- 改了 Prompt 后，要用同一组模型、同一批样例、同样的其他配置做对比，一次只改一个变量，否则无法归因。

### 新增一种模型输出

1. 在 `models.py` 定义 Pydantic 模型（继承 `StrictModel`，禁止多余字段）；
2. 在调用处写一个校验函数，检查 ID、必填证据等业务约束；
3. 在 `llm.py` 的重试分支里为新的 `schema_name` 加一条格式提示；
4. 用 Mock HTTP 写测试，覆盖合格输出、不合格输出和重试。

### 开发流程

```bash
python -m pytest -q     # 离线测试
ruff check .            # Lint
```

CI 在 Python 3.10 和 3.14 上运行这两项。涉及判决逻辑的改动，离线测试通过之后，还应该用真实模型跑一次回归集。

---

## 11. 局限与注意事项

- **段落 ID 只证明证据存在，不证明理解正确。** 模型可能引用了正确的段落，却得出错误的结论。
- **确定性守卫覆盖面很窄，而且有已知的误解析。** 只处理单个数值、可识别的单位和明确的计入表述；多数值句子、复杂逻辑、隐含条件都要靠模型。没有被识别的比较写法会按"恰好等于"处理，可能导致误报 FAIL（见 5.8 节）。
- **段落越长，证据越粗。** 引用的是整段，长段落里的具体出处需要人自己找。
- **离线测试不代表准确率。** 测试只保证结构、协议和分支逻辑正确。真实效果必须用真实模型在评测集上测。
- **评测集有过拟合风险。** 11 条回归集和 60 条旧开发集都被反复用于调试；v0.2.8 的 36 条 holdout 中也有 4 条（H28-02/03/04/13）后来被用于调试。目前没有完全干净的盲测集，判断泛化能力需要新建一个从未用于调试的留出集。详见 [examples/DATASETS.md](../examples/DATASETS.md)。
- **REVIEW_REQUIRED 需要人来处理。** 它是系统在拿不准时的诚实回答，不是失败。上线前需要规划好人工复核的流程。
- **API 是同步长请求。** 一次 dual 检查要串行调用多次模型，每次超时上限默认 180 秒。对外提供服务时，需要考虑任务队列、超时设置、鉴权和限流。

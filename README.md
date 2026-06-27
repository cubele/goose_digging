# 神鹅语挖掘 (Goose Digging)

自动挖掘"神鹅语":一句通顺的中文 `S`,经字符映射 `goose(S)` 后得到另一句也通顺、意思截然不同的中文。例:`粪厂 → 实境`。

方法为遗传算法造句 + 多模型 LLM 打分。LLM 只打分(通顺预筛 + 关系评分),不造句;造句由 GA 在"鹅语碎片 + 随机字映射"空间进化。

## 目录

- [快速开始](#快速开始)
- [配置 LLM](#配置-llm)
- [算法](#算法)
- [可调参数](#可调参数)
- [自定义 Prompt](#自定义-prompt)
- [调优指南](#调优指南)
- [输出文件](#输出文件)
- [脚本](#脚本)
- [项目结构](#项目结构)

---

## 快速开始

### 安装

```bash
git clone <本仓库>
cd goose_digging
pip install -e .
```

依赖(`pyproject.toml`):`opencc-python-reimplemented`、`openai`、`wordfreq`。Python ≥ 3.8。

### 配置 LLM

```bash
cp llm_config.toml.example llm_config.toml
# 编辑 llm_config.toml: 填 base_url / api_key / 各阶段模型
```

### 运行

```bash
python -m goose_digging.mining            # 默认: bidict → full → seed
python -m goose_digging.mining bidict     # 只跑双边词典阶段
python -m goose_digging.mining full       # 只跑全量枚举阶段
python -m goose_digging.mining seed       # 只跑 GA 进化 (需先有 scored 种子)
```

看进度:`tail -f mined/logs/run_*.log`

---

## 配置 LLM

所有 LLM 走 OpenAI 兼容 Chat Completions API(`base_url` + `api_key`)。支持 OpenAI / DeepSeek / 智谱 GLM / Together / 自部署 vLLM 等。

`llm_config.toml`(复制自 `llm_config.toml.example`):

```toml
[security]
base_url = "https://llmapi.paratera.com/v1"   # OpenAI 兼容端点
api_key  = "sk-..."                           # 你的 key (别提交这个文件)

[fluency]            # 通顺预筛: 便宜小模型, 关 thinking
model = "DeepSeek-V4-Flash"
thinking = false

[[score]]            # 关系评分: 多模型 cross-check, 取均分, 任一>=6 入 gold
model = "DeepSeek-V4-Flash"
thinking = true

[[score]]
model = "DeepSeek-V4-Pro"
thinking = false

[[score]]
model = "Qwen3.7-Max"
thinking = false

[[score]]
model = "GLM-5.1"
thinking = false
```

优先级:`llm_config.toml` 字段 > `config.py` 占位默认。无配置文件时走 `config.py` 占位(需自行改对)。

api_key 优先级:`[security] api_key` > 环境变量 `GOOSE_API_KEY`。

各阶段模型:
- **通顺预筛**(`[fluency]`):便宜快的小模型,关 thinking。只判两边通不通。
- **关系评分**(`[[score]]`,可列多个):强模型,开 thinking。每个候选被 k 个模型各打一次分,k 越大越准也越贵。建议 2-4 个。

---

## 算法

挖掘分三阶段,固定顺序,`state.json` 记进度,跨 run 续跑:

```
            ┌─────────┐     ┌─────────┐     ┌──────────────────────┐
scored.jsonl│         │     │         │     │                      │
   ↑        │ bidict  │────▶│  full   │────▶│   seed (GA 进化)      │──▶ state.gold_pairs
   │        │ 双边词典 │     │ 全量枚举 │     │  字/词映射积木 → 进化  │   (持久化 gold)
   └────────┤ 跳过预筛 │     │预筛+评分 │     │  → 全模型打分 → gold  │
   (种子源) └─────────┘     └─────────┘     └──────────────────────┘
```

### 阶段一、二:枚举 (bidict / full)

两阶段用枚举驱动:一次性枚举所有候选(固定 seed shuffle,跨 run 顺序一致),按 cursor 分批送预筛/评分。

- **bidict**:`S` 与 `goose(S)` 都在词典里的 pair。两边都是真词,跳过预筛,直接进评分。
- **full**:单边枚举(`S` 在词典),`goose(S)` 可能不在词典。走通顺预筛 + 评分。覆盖右边不在词典但讲得通的情况(赶班/撬松)。

两阶段产出 `scored.jsonl`(全量打分记录,含低分),作为 GA 的种子源。

#### 通顺预筛

判 left/right 两边是否都讲得通,砍明显乱码,省评分 token。满足以下任一即判通顺:

- (a) 在词表里 / 真实词 —— 词典命中直接判通顺,不调 LLM
- (b) 解释得通的词语(谐音/口语/俚语,如 赶班=赶去上班)
- (c) 可作为句子的一部分

只杀字硬拼、纯随机字组合、明显乱码。词典外的串才送模型判 (b)/(c)。两边都通顺才进评分。GA 阶段用更宽的 `GA_FLUENCY_SYSTEM`(能产生联想即通顺,长句尤其宽松)。

#### 批大小

| 阶段 | 机制 | 每批 |
|---|---|---|
| 预筛 (full) | 每轮预取 `FLUENCY_PARALLEL(5)` 个 `FLUENCY_FETCH(200)` 批,并行判通顺 | 1000 候选/轮 |
| 评分 (枚举) | 预筛存活攒满 `SCORE_BATCH(20)` 才进评分 | 20 条/批 |
| 评分 (GA) | offspring 分 `GA_SCORE_BATCH(10)` 一批喂全模型 | 10 条/批 |

bidict 跳过预筛,直接取 20 个未见的候选。预筛模型漏回的重试 `FLUENCY_RETRY_MAX(2)` 次,仍漏的按不通处理并报警。

#### 断点续跑

随时可中断,重启接着跑,不重烧 LLM:

- **枚举**:`state.json` 记游标(bidict/full 各一个)+ 已评 pair 集合(`seen_pairs`),重启从断点继续,跳过已评。
- **GA**:种群、epoch、已评的 S 全部持久化,重启续种群不重判。
- **评分记录**:`scored.jsonl` 每批追加写,断电前的评分不丢。

### 阶段三:GA 进化 (seed)

长句(3-9 字)组合空间爆炸,无法枚举。用稳态遗传算法 + 小生境在"鹅语碎片"空间进化。

**数据流**:
```
scored.jsonl ──▶ 字映射 (goose(a)=b 的单字对, 按频次)
              ──▶ 词映射 (词级 pair, 按 score)
                    │  近均匀采样 (高低分都采)
                    ▼
            GA 进化 (mining/ga/evolve.py)
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
     crossover  mutation   immigrant
     (段拼接)  (字映射替换  (近均匀采样
      父A+父B   /词积木/删/重复) 全新拼)
                    │
                    ▼  通顺预筛 (两边都通) + 全模型 score_pairs 打分
                    ▼  fitness = 全模型均分
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
     crowding 入种群      state.gold_pairs
     + aging/evict/seen   (max>=6 持久化)
```

**基因型**:一条候选中文串 S(3-9 字)。`goose(S)` 由 oracle 确定性算出。

**算子**:
- **crossover(段拼接)**:父 A 前缀 + 父 B 后缀,裁剪到合法长度。
- **mutation**:字映射替换 / 词积木插值 / 删字 (不造叠词 —— 旧版曾有重复字模式, 但 GA 的通顺预筛正好杀叠词堆砌, 造了也是白造)。
- **immigrant(随机移民)**:近均匀采样从零拼一条全新 S 注入种群。

**近均匀采样**:权重 = `1 + signal*boost_alpha`。高分积木权重微高,低分的也采,因为低分 pair 塞进长句后可能撞出高分反差,只采高分会漏。`boost_alpha=0` 退化为纯均匀随机。

**适应度**= 全模型 `score_pairs` 的 k 模型 cross-check 均分。

**防同化**(防止种群坍缩到一类套路):
- **aging**:年龄超 `GA_MAX_AGE` epoch 强制淘汰。
- **evict gold**:max≥6 的个体存 gold 后移出种群。
- **全局 seen_s**:评过的 S 记入 `state.ga_seen_s`,不重判。

**多样性**:fitness sharing(Levenshtein 邻域稀释聚集套路分)+ crowding 替换(新生填相似生态位)+ 随机移民 + 早熟震荡(diversity<2 时抬移民率)。

### 评分标尺

评分阶段全模型对每个 pair 打 0-10 分,`max(各模型分) >= SCORE_THRESH(6)` 入 gold:

- 9-10:两边反差强烈且都自然
- 6-8:明显的双关/反讽/荒诞
- 3-5:有点意思但牵强
- 0-2:强行编故事或毫无关联

---

## 可调参数

全部在 `goose_digging/mining/config.py`,改代码。模型走 `llm_config.toml`。

### 挖掘核心

| 参数 | 默认 | 说明 |
|---|---|---|
| `SCORE_THRESH` | 6 | 入 gold 阈值:max(各模型分)≥此值入 gold_pairs |
| `SCORE_BATCH` | 20 | 评分单批 pair 数(过大易被服务端 length 截断) |
| _(无分数门槛)_ | — | GA 积木池=全部枚举 pair (不卡分数); 高低分权重由 `SEED_BOOST_ALPHA` 近均匀采样区分, 低分 pair 成句后可能反而高分 |

### GA 进化

| 参数 | 默认 | 调大 | 调小 |
|---|---|---|---|
| `GA_POP_SIZE` | 60 | 种群大,探索广 | 省内存/算力 |
| `GA_OFFSPRING_PER_EPOCH` | 20 | 每 epoch 多产候选 | 省 LLM(每候选 k× 成本) |
| `GA_IMMIGRANT_RATE` | 0.40 | 更探索 | 更收敛 |
| `GA_CROSSOVER_RATE` | 0.50 | 更多碎片拼接 | — |
| `GA_MUTATION_RATE` | 0.20 | 更多变异 | — |
| `GA_TOURNAMENT_K` | 3 | 偏开发(选更强父) | 偏探索 |
| `GA_ELITE` | 1 | 收敛快 | 防种群坍缩 |
| `GA_SHARING_SIGMA` | 2.5 | 多样性惩罚放宽 | 加强多样性约束 |
| `GA_SCORE_BATCH` | 10 | 单批喂更多 | 不易截断 |
| `GA_MAX_AGE` | 4 | 好个体多留几代 | 更激进防同化 |
| `SEED_BOOST_ALPHA` | 0.05 | 高分积木权重更高 | 趋近纯均匀采样(0=纯随机) |
| `GA_IMMIGRANT_BOOST_ALPHA` | 0.05 | 移民采高分积木更多 | 移民采样更均匀 |

### 防同化

| 参数 | 默认 | 说明 |
|---|---|---|
| `GA_EVICT_PROMOTED` | True | gold 个体存档后移出种群 |
| `GA_EVICT_SCORE_THRESH` | 6.0 | max(scores)≥此值视为"已挖到",移出种群 |

---

## 自定义 Prompt

所有 prompt 在 `goose_digging/mining/prompts.py`。模型走 `llm_config.toml`,prompt 走代码。参数只能调"挖多少/多激进",prompt 决定"什么叫神、什么叫废",是微调语料风格的主要手段。

### 6 个 prompt

LLM 只在两处被调用:通顺预筛(便宜模型,砍乱码)和关系评分(全模型 cross-check)。每处分枚举阶段和 GA 阶段两条路径,共 4 条 system prompt,加 2 个 user prompt 构造函数。

| prompt | 用在哪 | 喂给 | 作用 |
|---|---|---|---|
| `FLUENCY_SYSTEM` | 预筛 枚举阶段 | `FLUENCY_MODEL` | 判短语讲不讲得通,三选一即通顺,只杀乱码 |
| `GA_FLUENCY_SYSTEM` | 预筛 GA 阶段 | `FLUENCY_MODEL` | 同上但更宽松,长句不误杀 |
| `SCORE_SYSTEM` | 评分 枚举阶段 | `SCORE_MODELS` | 定 pair 神不神,0-10 分,带反幻觉例 |
| `SEED_SCORE_SYSTEM` | 评分 GA 阶段 | `SCORE_MODELS` | 同上但通顺权重更高,防造句被双关带跑 |
| `fluency_prompt()` | 预筛 user(枚举+GA) | — | 拼待判短语,要回 `{"items":[...]}` |
| `score_prompt()` | 评分 user(枚举+GA) | — | 拼待评 pair,要回 `{"scores":[...]}` |

调用链:
- 枚举 full:`fluency_filter`(`FLUENCY_SYSTEM`)→ `score_pairs`(`SCORE_SYSTEM`)
- 枚举 bidict:跳过预筛 → `score_pairs`(`SCORE_SYSTEM`)
- GA:`fluency_filter`(`GA_FLUENCY_SYSTEM`)→ `score_offspring` → `score_pairs`(`SEED_SCORE_SYSTEM`)

### 各 prompt 说明

**通顺预筛**:任务是砍明显乱码,判据故意很宽(三选一即通顺),因为神鹅语的金子常在"单看不是标准词、但放进句子能读通"的边角(赶班、撬松)。"神不神"的判断完全交给评分,预筛只杀字硬拼。`GA_FLUENCY_SYSTEM` 比枚举版更宽,因为 GA 候选是字映射/词积木拼出的谐音联想串,严格判会误杀(吊挺吗=屌挺嘛)。

**关系评分**:全模型 cross-check 取均分,任一 ≥6 入 gold。`SCORE_SYSTEM` 核心是"读者能不能一眼品到",强调多种读法(整句/谐音/荤段/拆字)、常用度门槛、一眼品到门槛,及反幻觉自检(防给不相关两词脑补反差)。`SEED_SCORE_SYSTEM` 把通顺一票否决提到最前(每个字都认识 ≠ 通顺,如 奶奶瞪鸡冲奶位),防造句被双关带跑给高分。

### 改 prompt 的约束

1. **JSON 契约不能破**。user prompt 要求模型回:
   - `fluency_prompt` → `{"items": [{"text": "原词", "ok": true/false}]}`
   - `score_prompt` → `{"scores": [{"left": "原左", "score": 0-10, "why": "..."}]}`

   解析端(`jsonx.parse_json_list` + `shared.parse_scores`)取最外层数组,按 key 名 `text`/`ok`/`left`/`score`/`why` 取值。改文案可以,别改结构和字段语义:`score` 必须 0-10,`left`/`text`+`ok` 做一一对应(缺了这条候选就丢)。`why` 可删。

2. **正反例别删**。`SCORE_SYSTEM` 里 `软件→起凤=1`(防脑补)、`奶奶瞪鸡冲奶位→≤2`(防字硬拼)等反例是模型不跑偏的主要约束。改文案时保留。

3. **批量数量必须对齐**。两个 user prompt 都要求"必须全部回,N 条一一对应"。防模型漏回(输出截断会少条)。解析端有校验/重试(预筛)/报警(评分),但 prompt 这层要先卡住。

### 验证

```bash
pytest goose_digging/tests/                # mock LLM, 确认没改崩契约 (80 测)
python scripts/verify_full_e2e_mock.py     # mock 端到端, 三阶段串接 + GA
# 真实 API 跑一小批看效果:
python -m goose_digging.mining full
python -m goose_digging.mining seed
```

改 prompt 后看 `mined/logs/model_<ts>_<model>.log` 的思维链判断文案有没有起效;低分候选看 `scored.jsonl` 的 `whys`。

---

## 调优指南

目标 → 改哪里(参数去 `config.py`,prompt 去 `prompts.py`)。

**调语料风格(太荤/太正经/不够妙)**:
- 改 `SCORE_SYSTEM` / `SEED_SCORE_SYSTEM` 的标尺和正反例。这是调风格的主要手段。

**想挖得更快(省 LLM)**:
- 降 `GA_OFFSPRING_PER_EPOCH`(如 12-15)
- 降 `llm_config.toml` 的 `[[score]]` 模型数(k=2 够)
- 增大 `GA_SCORE_BATCH`(如 15-20)

**想覆盖更广**:
- 调大 `GA_IMMIGRANT_RATE`(如 0.5)
- 调大 `GA_SHARING_SIGMA`(如 3.0)
- 调小 `GA_ELITE`(可到 0)
- 调大 `GA_POP_SIZE`(如 80-100)
- 调小 `SEED_BOOST_ALPHA`(如 0.02,采样更均匀)

**想偏开发已验证的好积木**:
- 调大 `SEED_BOOST_ALPHA` / `GA_IMMIGRANT_BOOST_ALPHA`(如 0.1-0.2)

**想让好句子多留几代**:
- 调大 `GA_MAX_AGE`(如 8-10)

**想更激进防同化**:
- 调小 `GA_MAX_AGE`(如 2-3)

**换模型 / 换 API 服务商**:
- 改 `llm_config.toml` 的 `base_url` / `api_key` / `[[score]]`,不动代码。

**调 gold 阈值**:
- 改 `SCORE_THRESH`(如 7 更严)。

**调积木采样偏向**:
- 改 `SEED_BOOST_ALPHA`(高分积木权重更高; 调小→趋近纯均匀, 低分积木更容易采到)。积木池本身包含全部枚举 pair, 不再卡分数门槛。

**觉得好句子被预筛误杀**:
- 放宽 `GA_FLUENCY_SYSTEM` 的通顺判据(尤其长句)。

**模型总脑补不相关的双关**:
- 在 `SCORE_SYSTEM` 反幻觉自检段加反例。

---

## 输出文件

所有产出在 `mined/`(gitignore):

| 文件/目录 | 说明 |
|---|---|
| `scored.jsonl` | 全量评分记录(跨阶段累积)。GA 种子源 + split_gold 输入 |
| `state.json` | 挖掘状态:seen_pairs/gold_pairs/游标/GA 种群。gold_pairs 持久化全部 max≥6 |
| `logs/run_<ts>.log` | 主日志(进度+结果,不含思维链) |
| `logs/model_<ts>_<model>.log` | 各评分模型的思维链日志 |
| `gold_split/` | `split_gold.py` 生成的四类 gold 文件 |

### 数据 schema

**scored.jsonl**(一行一记录):
```json
{"score": 8.0, "left": "义父", "right": "盗摄", "why": "...",
 "dir": "seed_ga", "round": 1, "scores": [8,9,7,8], "whys": ["..."]}
```
- `score`:均分(float);`scores`:各模型分(int 列表)
- `dir`:`fwd`/`rev`/`fixed`/`seed_ga`
- `whys`:各模型打分理由(对应 scores)

**gold_split/*.jsonl**(split_gold.py 筛分生成):
```json
{"pair": "粪厂→实境", "score": 10.0, "scores": [10,10,10,10], "whys": ["..."]}
```
- `gold_fixed.jsonl`:不动点(left==right,max≥6)
- `gold_short.jsonl`:非不动点 ≤2 字,max≥6
- `gold_medium.jsonl`:非不动点 3-4 字,max≥5
- `gold_long.jsonl`:非不动点 ≥5 字,max≥5

**state.json**:
- `seen_pairs` `[[left,right],...]` 已挖 pair(去重)
- `gold_pairs` `[{S, real_goose_S, why, round, score, direction},...]` gold 库
- `n_iter` / `phase` / `cursor_bidict` / `cursor_full`:枚举进度
- `ga_population` `[{s, scores, born},...]` GA 种群(跨 run 续)
- `ga_epoch` GA 已跑 epoch 数
- `ga_seen_s` `[str,...]` GA 评过的 S

---

## 脚本

`scripts/`(独立工具,消费 `mined/scored.jsonl`):

| 脚本 | 用途 |
|---|---|
| `split_gold.py` | 从 scored.jsonl 筛分生成四个 gold 文件到 `mined/gold_split/` |
| `verify_full_e2e_mock.py` | mock LLM 完整端到端验证:真枚举 + 同一 state 贯穿三阶段 + 1000 轮 GA + 断电续传 |

```bash
python scripts/split_gold.py             # 筛分生成四个 gold 文件
python scripts/split_gold.py 20          # 同上 + 每类预览 top 20
python scripts/verify_full_e2e_mock.py   # mock e2e (真枚举+1000轮GA, ~7min, 不调真实 API)
```

---

## 项目结构

```
goose_digging/
├── llm_config.toml.example   # LLM 配置模板 (复制改名用)
├── llm_config.toml           # 真实配置 (gitignore)
├── README.md
├── LICENSE                   # MIT
├── pyproject.toml
└── goose_digging/
    ├── oracle/               # goose 字符变换 (确定性查表)
    │   ├── oracle.json       # 字符映射表 (21391 字)
    │   └── __init__.py       # goose()/goose_char()/build_inverse()
    └── mining/
        ├── __main__.py       # CLI 入口
        ├── config.py         # 可调参数 (knob)
        ├── llm_config.py     # TOML 配置读取 + 覆盖默认
        ├── llm.py            # OpenAI 客户端 + 流式调用
        ├── prompts.py        # 所有 prompt (4 system + 2 user), 见「自定义 Prompt」
        ├── fluency.py        # 通顺预筛
        ├── scoring.py        # 多模型 cross-check 评分
        ├── enumerate.py      # 字典枚举 (bidict/full)
        ├── pipeline.py       # 三阶段编排
        ├── seed.py           # GA 积木源 (字/词映射提取 + 近均匀采样)
        ├── state.py          # state.json 持久化 (断点续跑)
        ├── finding.py        # Finding 数据类 + 导出
        ├── wordlist.py       # 静态词典 (7 源合并)
        ├── readability.py    # 字级 zipf 频率 (预筛用)
        ├── ga/               # GA 进化
        │   ├── genome.py     # 基因型 + 算子 (crossover/mutation/immigrant)
        │   ├── population.py # 种群 + fitness sharing + crowding + aging/evict
        │   ├── fitness.py    # 全模型 score_pairs 打分
        │   └── evolve.py     # 稳态 GA 主循环
        └── tests/            # pytest (80 测)
            ├── test_goose.py   # oracle 表自洽 (goose_char/ungoose/roundtrip)
            ├── test_mining.py  # 纯单测: prompt 格式 / 采样数学 / 惰性枚举 / state 持久化
            ├── test_ga.py      # 纯单测: GA 算子 / 种群多样性 / 防同化 / 采样覆盖
            └── test_e2e.py     # mock e2e: 三阶段串接 + GA + 断点续传 (秒级, 不调真实 API)
```

## 测试

```bash
pytest goose_digging/tests/    # 80 测 (不调真实 API)
#  test_goose.py  oracle 表自洽 (确定性)
#  test_mining.py 纯单测: prompt 格式 / 采样数学 / 惰性枚举 / state 往返
#  test_ga.py      纯单测: GA 算子 / 种群多样性 / 防同化 / 采样覆盖
#  test_e2e.py     mock e2e: 三阶段串接 + GA + 断点续传 (秒级, mock LLM)
python scripts/verify_full_e2e_mock.py   # 大规模 mock e2e (真枚举 + 1000 轮 GA, ~7min)
```


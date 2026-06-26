# 神鹅语挖掘 (Goose Digging)

自动挖掘"神鹅语"——一种因字符编码错乱(GBK/EUC-JP 误读)产生的、两边都通顺却形成反差的双关中文。

> **神鹅语**:一句通顺的中文 `S`,经过字符映射 `goose(S)` 后,恰好变成**另一句也通顺、但意思截然不同**的中文。妙处全在两边的反差与谐音。
>
> 例子:`粪厂 → 实境`

本项目用 **遗传算法 + 多模型 LLM 评分** 自动发现这样的句子,**LLM 只用于打分(通顺预筛 + 关系质量),不参与造句**——造句靠 GA 在"鹅语碎片 + 随机字映射"空间进化。

## 目录

- [快速开始](#快速开始)
- [配置 LLM](#配置-llm)
- [算法](#算法)
- [可调参数 (knob)](#可调参数-knob)
- [改进方式](#改进方式)
- [输出文件与数据格式](#输出文件与数据格式)
- [辅助脚本](#辅助脚本)

---

## 快速开始

### 安装

```bash
git clone <本仓库>
cd goose_digging
pip install -e .
```

依赖(见 `pyproject.toml`):`opencc-python-reimplemented`、`openai`、`wordfreq`。Python ≥ 3.8。

### 配置 LLM

复制示例配置,填入你自己的 API:

```bash
cp llm_config.toml.example llm_config.toml
# 编辑 llm_config.toml: 填 base_url / api_key / 各阶段模型
```

详细字段见 [配置 LLM](#配置-llm)。

### 跑

```bash
python -m goose_digging.mining            # 默认: 按顺序 bidict → full → seed
python -m goose_digging.mining bidict     # 只跑双边词典阶段
python -m goose_digging.mining full       # 只跑全量枚举阶段
python -m goose_digging.mining seed       # 只跑 GA 神鹅语进化 (需先有 scored 种子)
```

实时看进度:`tail -f mined/logs/run_*.log`

---

## 配置 LLM

所有 LLM 走 **OpenAI 兼容 Chat Completions API**(`base_url` + `api_key`)。支持 OpenAI 官方 / DeepSeek / 智谱 GLM / Together / 自部署 vLLM 等。

`llm_config.toml`(复制自 `llm_config.toml.example`):

```toml
[security]
base_url = "https://llmapi.paratera.com/v1"   # OpenAI 兼容端点
api_key  = "sk-..."                           # 你的 key (别提交这个文件!)

[fluency]            # T1 通顺预筛: 便宜小模型, 关 thinking
model = "DeepSeek-V4-Flash"
thinking = false

[[score]]            # T2 关系评分: 多模型 cross-check, 取均分, 任一>=6 入 gold
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

**优先级**:`llm_config.toml` 里的字段 > `config.py` 占位默认。没有 `llm_config.toml` 时走 `config.py` 的 OpenAI 占位(需自行改对)。

**api_key 优先级**:TOML `[security] api_key` > 环境变量 `GOOSE_API_KEY`。(明文 `API_KEY` 文件回退已移除:易误提交泄露密钥。)

各阶段模型建议:
- **T1 通顺预筛**(`[fluency]`):便宜快的小模型,关 thinking。只判两边通不通,省 T2 的 token。
- **T2 关系评分**(`[[score]]`,可列多个):强模型,开 thinking(关系判断是难任务)。**模型越多越准但越贵**——每个候选要被 k 个模型各打一次分。建议 2-4 个。

默认配置(无 `llm_config.toml` 时)用的是 `https://llmapi.paratera.com/v1` 端点 + DeepSeek-V4-Flash/Pro + Qwen3.7-Max + GLM-5.1 这套,可按需改成自己的 OpenAI 兼容服务。

---

## 算法

挖掘分三阶段(固定顺序,`state.json` 记进度,跨 run 续跑):

```
            ┌─────────┐     ┌─────────┐     ┌──────────────────────┐
scored.jsonl│         │     │         │     │                      │
   ↑        │ bidict  │────▶│  full   │────▶│   seed (GA 进化)      │──▶ state.gold_pairs
   │        │ 双边词典 │     │ 全量枚举 │     │  字/词映射积木 → 进化  │   (持久化 gold)
   └────────┤ 跳过T1  │     │ T1+T2   │     │  → 全模型打分 → gold  │
   (种子源) └─────────┘     └─────────┘     └──────────────────────┘
```

### 阶段一、二:枚举 (bidict / full)

两阶段都用**枚举驱动**:`all_pairs` 一次性枚举(固定 seed shuffle,跨 run 顺序一致),按 cursor 分批送 T1/T2,**state.json 记 cursor/seen_pairs,断电续跑**。

- **bidict**:S 和 `goose(S)` 都在词典里的 pair。零噪声(两边都是真词)。**跳过 T1** —— 两边都在词典已天然满足 T1 的判据(a),字典短路零 LLM,直接进 T2。
- **full**:单边枚举(S 在词典),`goose(S)` 可能不在词典。走 **T1 通顺预筛**(三选一判据)+ T2 评分。覆盖"右边不在词典但讲得通"的(如 赶班/撬松)。

这两个阶段产出 `scored.jsonl`(**所有打分 pair 全量落盘,含低分**),作为下一阶段 GA 的**种子源**。

#### T1 通顺预筛 (三选一判据)

T1 = **两边都讲得通才打分**的预筛,砍明显乱码/字硬拼,省 T2 全模型 token。对每个 left/right,满足以下任一即判通顺:

- **(a) 在词表里 / 真实词** → 字典短路,**零 LLM** (`fluency_filter` 词典命中直接 ok=true)
- **(b) 解释得通的词语** —— 谐音/口语/俚语/略不规范但讲得通(如 赶班=赶去上班)
- **(c) 可作为句子的一部分** —— 能塞进一句话说得通

只杀"字硬拼/纯随机字组合/明显乱码"。词典外的串才送 Flash(关 thinking)判 (b)/(c)。**只有 left 和 right 都判通顺的 pair 才进 T2 打分**。GA 阶段用更宽的 `GA_FLUENCY_SYSTEM`(能解释得通产生联想即可,长句尤其宽松)。

#### 枚举打包 & LLM 批大小控制

| 阶段 | 机制 | 每批大小 |
|---|---|---|
| **T1**(full) | 每轮预取 `T1_PARALLEL(5)` 个 `T1_FETCH(200)` 批,并行判通顺 | 5×200=1000 候选/轮 |
| **T2**(枚举) | T1 存活攒满 `SCORE_BATCH(20)` 才进 T2 | 恰好 20 条/批 |
| **GA T2** | offspring 分 `GA_SCORE_BATCH(10)` 一批喂全模型 | 恰好 10 条/批 |

- **full**:`while len(survivors) < SCORE_BATCH` 一直预取+判通顺,直到攒满 20 条存活才喂 T2 → T2 每批必然 20 条(除非候选耗尽)。
- **bidict**:跳 T1,直接取 20 个未见的候选。
- **T1 批大小硬编码**:`T1_FETCH=200` 控制喂 Flash 的批,`T1_PARALLEL=5` 路并行,合计每轮 1000 候选。Flash 漏回的重试(`FLUENCY_RETRY_MAX=2`),仍漏的才按不通处理(报警,不静默误杀)。

#### 断电续跑 (三重保护)

1. **cursor 落盘**:`_next_batch` 前进 cursor 后,迭代末尾 `state.save()` **原子写** state.json(`.tmp`→`replace`)。重启从保存的 cursor 继续,不重判已前进的候选。bidict/full 各独立 cursor。
2. **seen_pairs 兜底**:即使 cursor 落盘失败,`seen_pairs`(已评 pair 集合)也持久化,`_next_batch` 跳过已见 pair → **幂等,不重烧 LLM**。
3. **GA 续跑**:`state.ga_population`(种群)+ `ga_epoch` + `ga_seen_s`(GA 评过的 S)全部持久化,重启后 `evolve` 用 `existing_population` 续种群,`ga_seen_s` 防重烧。
4. **scored.jsonl 追加写**:每批 T2 打分后立即追加落盘(不覆盖),断电前的评分记录不丢。

### 阶段三:GA 神鹅语进化 (seed)

长句(3-9 字)组合空间爆炸,无法枚举。用**稳态遗传算法 + 小生境**在"鹅语碎片"空间进化,**LLM 只打分**。

**数据流**:
```
scored.jsonl 已评 pair ──▶ 字映射 (goose(a)=b 的单字对, 按频次)
                      ──▶ 词映射 (词级 pair, 按 score)
                            │  近均匀 boost 采样 (高低分都采, 保证覆盖)
                            ▼
            GA 进化 (mining/ga/evolve.py)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         crossover      mutation      immigrant
         (段拼接)      (字映射替换    (近均匀采样
          父A+父B       /词积木/删/重复) 全新拼)
                            │
                            ▼  预筛 (T1 两边都通) + 全模型 score_pairs 打分
                            │
                            ▼  fitness = 全模型 cross-check 均分
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       crowding 入种群               state.gold_pairs
       + 防同化三件套                  (max>=6 持久化)
       (aging/evict/seen)
```

**基因型**:一条候选中文串 S(3-9 字)。`goose(S)` 由 oracle 确定性算出。

**算子**(全部从现有积木构造,不引入新数据源):
- **crossover(段拼接)**:父 A 前缀 + 父 B 后缀,裁剪到合法长度。已有鹅语碎片的拼接。
- **mutation**:字映射替换 / 词积木插值 / 删字 / 重复字(产生叠词)。直接驱动 `goose(S)` 变化。
- **immigrant(随机移民)**:近均匀 boost 采样从零拼一条全新 S 注入种群。探索源,防早熟。

**近均匀 boost 采样**(取代旧温度采样):权重 = `1 + signal*boost_alpha`(`SEED_BOOST_ALPHA=0.05`)。高分 9→1.45,低分 1→1.05,比值 ~1.4:1,**几乎一视同仁**。动机:低分词映射成句后可能反而出现高分长句(低分 pair 单看无巧,但塞进长句可能撞出神级反差),故需尽量完整覆盖,不能只采高分积木。`boost_alpha=0` 时退化为纯均匀随机。

**适应度(fitness)**= 全模型 `score_pairs` 的 k 模型 cross-check **均分**。这是真·神鹅语审美信号。

> **为什么不用单模型 surrogate 省成本?** 早期版本用过单模型(Flash)驱动内层进化,实测单模型系统性高估"两边勉强通顺但无巧思"的候选,导致 GA 朝**该单模型的偏好**收敛(进化方向被带偏,promote→gold 转化率仅 40%)。改为全模型直接打分后,fitness 真实可靠。代价是 LLM 成本 ~k×,代数减少,但每代质量真实。

**T1 通顺预筛**:全模型打分前,先用便宜小模型(关 thinking)判候选**两边是否都讲得通**(三选一判据见上)。两边都不通的(如 哪哪墩墩、墙松垮你活活)直接杀掉,不浪费全模型 token。GA 阶段用更宽判据(`GA_FLUENCY_SYSTEM`,能解释得通产生联想即可,长句尤其宽松)——因为 GA 候选本就是谐音/联想串,严格判会误杀(吊挺吗=屌挺嘛)。

**防同化三件套**(防止种群坍缩到一类套路):
- **aging**:个体年龄超 `GA_MAX_AGE` epoch 强制淘汰(即使满分),防高分个体永久存活主导种群。
- **evict gold**:全模型 max≥6 的个体(已存 gold)从种群移除——既存档就不该再占种群当进化锚点。
- **全局 seen_s**:评过的 S 记入 `state.ga_seen_s`,永不重烧 LLM。

**多样性机制**:fitness sharing(Levenshtein 邻域稀释聚集套路分)+ crowding 替换(新生填相似生态位而非总替最差)+ 40% 随机移民 + 早熟震荡(diversity<2 时抬移民率)。**GA 参数已调宽**(种群 60/移民 0.40/精英 1/aging 4/sharing 2.5),配合近均匀采样最大化搜索空间覆盖,长时间运行持久探索广泛的神长句。

### 评分标准

T2 全模型对每个 pair 打 0-10 分,`max(各模型分) >= SCORE_THRESH(6)` 入 gold:
- **9-10** 拍案叫绝,两边反差强烈且都自然
- **6-8** 明显的双关/反讽/荒诞
- **3-5** 有点意思但牵强
- **0-2** 强行编故事或毫无关联

---

## 可调参数 (knob)

全部在 `goose_digging/mining/config.py`,hardcode 改这。LLM 模型走 `llm_config.toml`。

### 挖掘核心

| 参数 | 默认 | 说明 |
|---|---|---|
| `SCORE_THRESH` | 6 | 入 gold 阈值:max(各模型分)≥此值入 state.gold_pairs |
| `SCORE_BATCH` | 20 | T2 单批喂多少 pair(过大易被服务端 length 截断) |
| `SEED_MIN_SCORE` | 4.5 | GA 从 scored.jsonl 选 score≥此值的 pair 提取字/词映射积木 |

### GA 进化

| 参数 | 默认 | 调大效果 | 调小效果 |
|---|---|---|---|
| `GA_POP_SIZE` | 60 | 种群大,探索广 | 省内存/算力 |
| `GA_OFFSPRING_PER_EPOCH` | 20 | 每 epoch 多产候选 | 省 LLM(每候选 k× 成本) |
| `GA_IMMIGRANT_RATE` | 0.40 | 更多随机移民=更探索 | 更收敛(靠 crossover/mutation) |
| `GA_CROSSOVER_RATE` | 0.50 | 更多碎片拼接 | — |
| `GA_MUTATION_RATE` | 0.20 | 更多变异 | — |
| `GA_TOURNAMENT_K` | 3 | 偏开发(选更强父) | 偏探索(更随机) |
| `GA_ELITE` | 1 | 保留更多精英(收敛快) | 防种群坍缩 |
| `GA_SHARING_SIGMA` | 2.5 | 更大邻域=多样性惩罚适度放宽 | 加强多样性约束 |
| `GA_SCORE_BATCH` | 10 | 全模型单批喂更多 | 单批不易截断 |
| `GA_MAX_AGE` | 4 | 好个体多留几代进化 | 更激进防同化 |
| `SEED_BOOST_ALPHA` | 0.05 | 高分积木权重更高 | 趋近纯均匀采样(0=纯随机) |
| `GA_IMMIGRANT_BOOST_ALPHA` | 0.05 | 移民采高分积木更多 | 移民采样更均匀 |

### 防同化

| 参数 | 默认 | 说明 |
|---|---|---|
| `GA_MAX_AGE` | 4 | 个体年龄上限(epoch)。超龄淘汰,防高分永久主导 |
| `GA_EVICT_PROMOTED` | True | gold 个体(max≥thresh)存档后移出种群 |
| `GA_EVICT_SCORE_THRESH` | 6.0 | 配合 evict:max(scores)≥此值视为"已挖到" |

---

## 改进方式

常见调优目标 → 改哪个参数:

**想挖得更快(省 LLM)**:
- 降 `GA_OFFSPRING_PER_EPOCH`(如 12-15)
- 降 `llm_config.toml` 的 `[[score]]` 模型数(k=2 就够)
- 增大 `GA_SCORE_BATCH`(如 15-20),减少全模型调用批次数

**想要更多元(覆盖更广的神鹅语)**:
- 调大 `GA_IMMIGRANT_RATE`(如 0.5)
- 调大 `GA_SHARING_SIGMA`(如 3.0,邻域更宽,相似套路惩罚更弱)
- 调小 `GA_ELITE`(已是 1,可保持或调到 0)
- 调大 `GA_POP_SIZE`(如 80-100)
- 调小 `SEED_BOOST_ALPHA`(如 0.02,采样更均匀,低分积木也覆盖)

**想让采样更偏高分积木(开发已验证的好积木)**:
- 调大 `SEED_BOOST_ALPHA`/`GA_IMMIGRANT_BOOST_ALPHA`(如 0.1-0.2,高分权重提升)

**想让好句子多留几代(挖得更深)**:
- 调大 `GA_MAX_AGE`(如 8-10)

**想更激进防同化**:
- 调小 `GA_MAX_AGE`(如 2-3)

**换模型 / 换 API 服务商**:
- 改 `llm_config.toml` 的 `base_url` / `api_key` / `[[score]]` 列表
- 不用动代码

**调 gold 阈值(想更严/更松)**:
- 改 `SCORE_THRESH`(如 7 更严,gold 更精)

**调种子源(想用更高/更低的积木)**:
- 改 `SEED_MIN_SCORE`(如 6 只用高质量 pair 提取积木)

---

## 输出文件与数据格式

所有产出在 `mined/`(gitignore,各自重挖):

| 文件/目录 | 说明 |
|---|---|
| `scored.jsonl` | 所有评分记录(全量,跨阶段累积)。GA 的种子源 + split_gold 的输入 |
| `state.json` | 挖掘状态(断点续跑):seen_pairs/gold_pairs/游标/GA 种群(gold_pairs 持久化全部 max≥6 的金鹅语) |
| `logs/run_<ts>.log` | 主日志(进度+结果,不含思维链) |
| `logs/model_<ts>_<model>.log` | 各评分模型的思维链日志(tail -f 盯单模型) |
| `gold_split/` | `split_gold.py` 生成的四类 gold 文件(手动跑脚本生成,不在挖矿时实时写) |

### 数据 schema

**scored.jsonl**(一行一记录,全量已打分,是所有 gold 的源):
```json
{"score": 8.0, "left": "义父", "right": "盗摄", "why": "...",
 "dir": "seed_ga", "round": 1, "scores": [8,9,7,8], "whys": ["..."]}
```
- `score`:均分(float);`scores`:各模型分(int 列表)
- `dir`:`fwd`/`rev`/`fixed`/`seed_ga`
- `whys`:各模型的打分理由(对应 scores)

**gold_split/*.jsonl**(split_gold.py 从 scored.jsonl 筛分生成,gold 格式):
```json
{"pair": "粪厂→实境", "score": 10.0, "scores": [10,10,10,10], "whys": ["..."]}
```
- `gold_fixed.jsonl`:不动点神鹅语(left==right,max≥6)
- `gold_short.jsonl`:短神鹅语(非不动点,≤2字,max≥6)
- `gold_medium.jsonl`:中神鹅语(非不动点,3-4字,max≥5)
- `gold_long.jsonl`:长神鹅语(非不动点,≥5字,max≥5)

**state.json**(断点续跑):
- `seen_pairs` `[[left,right],...]` 已挖 pair(去重)
- `gold_pairs` `[{S, real_goose_S, why, round, score, direction},...]` gold 库(全部 max≥6,持久化)
- `n_iter` / `phase` / `cursor_bidict` / `cursor_full`:枚举阶段进度
- `ga_population` `[{s, scores, born},...]` GA 种群(跨 run 续)
- `ga_epoch` GA 已跑 epoch 数
- `ga_seen_s` `[str,...]` GA 评过的所有 S(防重烧 LLM)

---

## 辅助脚本

`scripts/`(独立工具,消费 `mined/scored.jsonl`):

| 脚本 | 用途 |
|---|---|
| `split_gold.py` | 从 scored.jsonl 全量筛分生成四个 gold 文件(按均分降序)到 `mined/gold_split/`:`gold_fixed.jsonl`(不动点,max≥6)、`gold_short.jsonl`(非不动点 ≤2字,max≥6)、`gold_medium.jsonl`(非不动点 3-4字,max≥5)、`gold_long.jsonl`(非不动点 ≥5字,max≥5) |
| `verify_e2e_mock.py` | **mock LLM 端到端验证**:不调真实 API,在 tmp 目标跑一遍 bidict→full→seed→采样覆盖→断点续传,验证改进后流程正确。开发改完算法后跑一遍回归 |

```bash
python scripts/split_gold.py        # 筛分生成四个 gold 文件到 mined/gold_split/
python scripts/split_gold.py 20     # 同上 + 每类预览 top 20 到终端
python scripts/verify_e2e_mock.py   # mock LLM 端到端验证 (不调真实 API, 不碰真实 mined/)
```

---

## 项目结构

```
goose_digging/
├── llm_config.toml.example   # 用户 LLM 配置模板 (复制改名用)
├── llm_config.toml           # 你的真实配置 (gitignore, 自建)
├── README.md
├── LICENSE                   # MIT
├── pyproject.toml
└── goose_digging/
    ├── oracle/               # goose 字符变换 (确定性查表)
    │   ├── oracle.json       # 字符映射表 (21391 字)
    │   └── __init__.py       # goose()/goose_char()/build_inverse()
    └── mining/
        ├── __main__.py       # CLI 入口
        ├── config.py         # 全部可调参数 (knob)
        ├── llm_config.py     # TOML 配置读取 + 覆盖默认
        ├── llm.py            # OpenAI 客户端 + 流式调用
        ├── prompts.py        # 所有 system/user prompt
        ├── fluency.py        # T1 通顺预筛 (两边都通才留)
        ├── scoring.py        # T2 多模型 cross-check 评分
        ├── enumerate.py      # 字典枚举 (bidict/full)
        ├── pipeline.py       # 三阶段编排
        ├── seed.py           # GA 积木源 (字/词映射提取 + 近均匀 boost 采样)
        ├── state.py          # state.json 持久化 (断点续跑)
        ├── finding.py        # Finding 数据类 + 导出
        ├── wordlist.py       # 静态词典 (7 源合并)
        ├── readability.py    # 字级 zipf 频率 (预筛用)
        ├── ga/               # GA 神鹅语进化
        │   ├── genome.py     # 基因型 + 算子 (crossover/mutation/immigrant)
        │   ├── population.py # 种群 + fitness sharing + crowding + aging/evict
        │   ├── fitness.py    # 全模型 score_pairs 打分
        │   └── evolve.py     # 稳态 GA 主循环
        └── tests/            # pytest (86 测, 全 mock LLM)
```

## 测试

```bash
pytest goose_digging/tests/    # 86 测全过 (全 mock LLM, 不调真实 API)
```

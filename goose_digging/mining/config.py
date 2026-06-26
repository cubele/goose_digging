# -*- coding: utf-8 -*-
"""集中配置: 路径/模型/挖掘参数.

改 prompt 文案去 prompts.py, 改参数(阈值/batch/GA)在这.
LLM 模型/api_key/base_url 走 llm_config.toml (用户自定义), 见 llm_config.toml.example.

目录布局:
  项目根 (PROJ_DIR): llm_config.toml, mined/
    goose_digging/   (包 PKG_DIR): oracle/, mining/
      mining/        (本文件)
"""
from __future__ import annotations

from pathlib import Path

_THIS = Path(__file__).resolve()
_PKG_DIR = _THIS.parents[1]    # goose_digging/ (包)
PROJ_DIR = _THIS.parents[2]    # 项目根

OUT_DIR = PROJ_DIR / "mined"                  # 数据输出 (scored/gold/state)
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR = OUT_DIR / "logs"                     # 日志 (run/model/llm_debug)
LOG_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 挖掘参数 (adhoc 改这里)
# ===========================================================================

# --- 枚举/T1 通顺预筛 ---
T1_FETCH = 200       # T1 单批喂的候选数 (大 batch 省 Flash 调用)
T1_PARALLEL = 5      # T1 并行度: 同时判多少个 FETCH 批 (5 = 一次 1000 候选并行判)
FLUENCY_RETRY_MAX = 2   # T1 Flash 漏回时最大重试次数 (首判 + 此值次重试)

# --- T2 评分 ---
SCORE_BATCH = 20     # T2 评分批大小 (减小可降低单批 prompt 体积, 避免服务端 length 截断)
SCORE_THRESH = 6     # 入 gold 阈值: 任一模型打分>=此值的 pair 入 gold.jsonl 供人工看
STAR_TOP_OFFSET = 2  # 极致档(✨)阈值 = SCORE_THRESH + 此偏移 (日志显示用, ≥此=极致神鹅语)

# --- seed/GA 字积木参数 (extract_char_maps/word_maps + 近均匀采样移民算子用) ---
SEED_MIN_LEN = 3     # 候选最短字数 (短于此丢弃)
SEED_MAX_LEN = 9     # 候选最长字数 (GA crossover/immigrant 长度上界)
SEED_MIN_SCORE = 4.5   # 从 scored.jsonl 选 score>=此值的 pair 提取字/词映射 (GA 积木源)
SEED_N_CHARS = 200   # 每次 iter 采样的字映射数 (GA 移民/变异算子用)
SEED_N_WORDS = 200   # 每次 iter 采样的词映射数
# 近均匀采样: 权重 = 1 + signal*boost_alpha. 高分/高频只微弱加成, 低分低频也尽量采,
# 保证完整覆盖 (低分词映射成句后可能反而高分). boost_alpha 小=近均匀, 0=纯均匀随机.
SEED_BOOST_ALPHA = 0.05    # 字/词映射加性 boost 系数
WORD_MAP_LEN = 2     # 词映射只采几字的 pair (2=只采2字词当积木)
# (废弃, 保留兼容旧引用) 旧温度采样参数, 已被 SEED_BOOST_ALPHA 取代.
SEED_SAMPLE_T_MIN = 1.1   # DEPRECATED: 旧采样温度下限 (已改用加性 boost)
SEED_SAMPLE_T_MAX = 2.0   # DEPRECATED: 旧采样温度上限

# ===========================================================================
# GA 神鹅语进化 (取代 LLM 造句: 稳态 GA + 小生境多样性, 全模型直接打分)
#   fitness = 全模型 score_pairs 的 k 模型 cross-check 均分 (真·神鹅语审美信号).
#   不再用单模型 surrogate 驱动进化 (单模型曾导致 GA 朝其偏好收敛).
#   算子原料 = 字映射/词映射近均匀 boost 采样, 即"鹅语碎片+随机字映射". 详见 mining/ga/.
#
# 调宽参数 (目标: 广泛覆盖、持久探索, 挖出尽可能多的神长句):
#   种群更大、移民更多、精英更少、aging 更激进、sharing 邻域略放宽.
#   配合近均匀采样 (低分积木也能进), 最大化搜索空间覆盖.
# ===========================================================================
GA_POP_SIZE = 60          # 种群大小 (稳态: 每次只替换少数个体). 调大→探索更广
GA_OFFSPRING_PER_EPOCH = 20  # 每 epoch 产 offspring 数. 全模型打分成本 k×, 别设太大
GA_IMMIGRANT_RATE = 0.40  # offspring 中随机移民占比 (近均匀采样全新拼, 探索源防早熟). 调大→更探索
GA_CROSSOVER_RATE = 0.50  # offspring 中 crossover(段拼接) 占比
GA_MUTATION_RATE = 0.10   # offspring 中 mutation 占比. 注意 IMMIGRANT+CROSSOVER+MUTATION=1.0
                          #   (evolve 里 mutation 吃前两者之后的余数; 三者和应=1)
GA_TOURNAMENT_K = 3       # 锦标赛选父大小 (小=多探索, 大=偏开发)
GA_ELITE = 1              # 精英保留数. 调小→防收敛, 给新生更多机会
GA_SHARING_SIGMA = 2.5    # fitness sharing 邻域半径 (Levenshtein 距离); 调大→相似套路惩罚适度放宽
                          #   配合均匀采样让高分长句即使相似也能多留几代
GA_SCORE_BATCH = 10       # 全模型 score_pairs 一次喂多少 offspring (单批过大易截断)
GA_IMMIGRANT_BOOST_ALPHA = 0.05  # 移民算子近均匀采样 boost 系数 (复用 seed 加性 boost 思路)
GA_GEN_TAG = "seed_ga"    # GA 产物落 scored.jsonl 的 dir 标签 (load_seed_pairs 按 seed 前缀排除, 不喂回)
# (废弃, 保留兼容旧引用) 旧温度采样参数, 已被 GA_IMMIGRANT_BOOST_ALPHA 取代.
GA_IMMIGRANT_T_MIN = 1.1  # DEPRECATED: 旧移民温度采样下限
GA_IMMIGRANT_T_MAX = 2.0  # DEPRECATED: 旧移民温度采样上限
# --- 防同化 / 防重复 (核心: 已挖鹅语存档后移出种群, 让种群永远留给探索中的个体) ---
GA_MAX_AGE = 4           # 个体年龄上限 (epoch). 超龄强制淘汰, 防高分个体永久存活主导种群.
                          #   调小→更激进防同化, 高分长句不能永久主导. 个体每经历一个 epoch 年龄+1.
GA_EVICT_PROMOTED = True # gold 个体(已存档)从种群移除: 既存档就不该再占种群当进化锚点
                          #   否则一个 score-9 个体永久精英保护 → 子代淹没种群 (同化主因)
GA_EVICT_SCORE_THRESH = 6.0  # full 模型 max(scores)>=此值即视为"已挖到", 移出种群

# --- LLM 调用 ---
LLM_RETRIES = 2        # llm_chat 单次调用最大重试次数 (网络/限流/截断)
LLM_RETRY_BACKOFF = 1.5  # 重试退避基数 (第 N 次重试睡 BACKOFF*N 秒)
PROGRESS_HEARTBEAT_SEC = 5.0  # T2 多模型并行时的进度心跳间隔 (秒)


# ===========================================================================
# LLM 模型配置 (占位默认; 用户用 llm_config.toml 覆盖, 见 llm_config.toml.example)
#   T1 通顺预筛: 低成本模型, 关 thinking, 判两边通不通 (砍噪声, 省 T2 token).
#   T2 关系评分: k 模型 cross-check, 取均分, 任一 >=SCORE_THRESH 入 gold.
#
# 开源默认用 OpenAI 占位模型; 自托管/其他厂商用户请复制
# llm_config.toml.example → llm_config.toml 配置自己的模型.
# ===========================================================================
FLUENCY_MODEL = "DeepSeek-V4-Flash"
FLUENCY_THINKING = False  # T1 通顺判关思考 (省 token, 简单任务)

SCORE_MODELS = [
    {"model": "DeepSeek-V4-Flash", "thinking": True},
    {"model": "DeepSeek-V4-Pro", "thinking": False},
    {"model": "Qwen3.7-Max", "thinking": False},
    {"model": "GLM-5.1", "thinking": False},
]

# --- 用户配置覆盖 (llm_config.toml 优先于上面的占位默认) ---
API_BASE_URL = "https://llmapi.paratera.com/v1"   # 默认 endpoint


def _apply_user_config(g: dict) -> None:
    """启动时读 llm_config.toml 覆盖 LLM 相关默认 (模型/base_url/api_key 来源).

    在本模块尾部调用一次. 只覆盖用户在 toml 里声明了的字段, 没声明的保留默认.
    """
    from .llm_config import (load_user_config, resolve_base_url, models_section,
                             fluency_model)
    cfg = load_user_config()
    if not cfg:
        return
    # base_url
    bu = resolve_base_url(cfg)
    if bu:
        g["API_BASE_URL"] = bu
    # fluency 模型
    fm = fluency_model(cfg)
    if fm:
        g["FLUENCY_MODEL"] = fm["model"]
        g["FLUENCY_THINKING"] = fm["thinking"]
    # score 模型数组
    sm = models_section(cfg, "score")
    if sm:
        g["SCORE_MODELS"] = sm


_apply_user_config(globals())

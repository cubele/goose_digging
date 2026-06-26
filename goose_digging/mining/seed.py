# -*- coding: utf-8 -*-
"""基于已有金鹅语的 GA 进化积木系统 (mining/ga 的原料源).

长句 (3字+) 组合空间爆炸, 无法枚举. 解法: GA 进化 (mining/ga/evolve.py), 算子原料 =
从 scored.jsonl 里已评分的 pair 提取的两类积木:
  - 字映射: goose(a)=b 的单字对 (如 枯→赶), 频次高 = 验证过的有效字级变换
  - 词映射: 高分词级 pair (如 粪厂→实境, 奶头→奶片), 是长句神鹅语的词级积木

采样策略: 近均匀 + 加性小 boost. 高分/高频只提供很小的概率提升, 低分低频也尽量采,
保证完整覆盖. 动机: 低分词映射在成句以后可能反而出现高分长句 (低分 pair 单看无巧,
但塞进长句可能撞出神级反差), 故需尽量一视同仁, 不能只采高分积木.
权重 = 1.0 + signal * boost_alpha (boost_alpha 小, 如 0.05), 高分 9 → 1.45, 低分 1 → 1.05,
比值 ~1.4:1, 几乎均匀. GA 的 immigrant/mutation 算子每次用不同随机种子 (基于 epoch 号).

本模块提供积木源 (extract_*/sample_*), GA 主循环在 mining/ga/evolve.py:
  1. load_seed_pairs: 从 scored.jsonl 读 score>=min 的 pair
  2. extract_char_maps / extract_word_maps: 拆字映射 + 词映射, 带原始信号
  3. sample_char_maps / sample_word_maps: 近均匀 boost 采样 (GA immigrant/mutation 算子用)
  4. gen_ga_candidates: 编排 GA 进化 → 全模型打分 → gold/落盘 (见 mining/ga/evolve.py)
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable

import openai

from goose_digging.oracle import goose, goose_char
# (gen_sentences 旧 LLM 造句已删, 现走 mining/ga 进化; llm 只在 scoring/fluency 打分)


# ---- 素材提取 ----

def extract_char_maps(pairs: list[dict], top_k: int | None = None) -> list:
    """从评过分的 pair 提取 goose 字级映射 (a→b=goose(a)), 按出现频次降序.

    只取短词 (≤4字), 拆成单字对. 频次高 = 验证过的高价值字映射.
    去掉不动点 (a→a, 没有变换价值).
    top_k 默认不限 (全采, 交给近均匀采样决定取哪些, 靠多 iter 覆盖广).
    返回 Counter.most_common 格式 [((a,b), freq), ...].
    """
    cnt: Counter[tuple[str, str]] = Counter()
    for p in pairs:
        s = p.get("left", "")
        if len(s) > 4:
            continue
        for a in s:
            b = goose_char(a)
            if b != a:  # 跳过不动点字
                cnt[(a, b)] += 1
    return cnt.most_common(top_k) if top_k else cnt.most_common()


def extract_word_maps(pairs: list[dict], top_k: int | None = None) -> list[tuple[str, str, float]]:
    """从评过分的 pair 提取词级映射 (left→right), 带 score 权重, 按分降序.

    只取 WORD_MAP_LEN 字词级 pair (当造句积木). 不动点 (left==right) 也保留 —— 它是自指神鹅语.
    更长的 (射粪头/吗啡片 等) 不采 —— 它们是成品长句, 不该当积木喂回去.
    top_k 默认不限 (全采, 交给近均匀采样决定取哪些, 靠多 iter 覆盖广).
    """
    from .config import WORD_MAP_LEN
    out: list[tuple[str, str, float]] = []
    for p in pairs:
        s = p.get("left", "")
        r = p.get("right", "")
        sc = p.get("score", 0)
        if len(s) == WORD_MAP_LEN:
            out.append((s, r, float(sc)))
    out.sort(key=lambda x: -x[2])  # 按分降序
    return out[:top_k] if top_k else out


# ---- 近均匀 + 加性 boost 采样 ----

def _boost_sample(items: list, signals: list[float], k: int,
                  boost_alpha: float, rng: random.Random) -> list:
    """近均匀加权采样: 权重 = 1 + signal*boost_alpha, 不放回.

    高分/高频只微弱加成, 低分低频也尽量采, 保证完整覆盖.
    boost_alpha 小 (如 0.05) 时: signal=9→1.45, signal=1→1.05, 比值~1.4:1, 近均匀.
    boost_alpha=0 时退化为纯均匀随机.
    items/signals 等长, 返回 k 个 item.
    """
    if len(items) <= k:
        return list(items)
    # 加性 boost: 基础权重 1 + signal*alpha. signal 负或零时退回 1 (不惩罚).
    weights = [1.0 + max(0.0, s) * boost_alpha for s in signals]
    total = sum(weights)
    if total <= 0:
        return rng.sample(items, k)
    # 不放回加权抽样: 每轮按权重抽 1 个, 抽过就剔权重, 重复到 k 个.
    pool_idx = list(range(len(items)))
    pool_w = list(weights)
    result = []
    while len(result) < k and pool_idx:
        total_w = sum(pool_w)
        if total_w <= 0:
            rng.shuffle(pool_idx)
            for i in pool_idx[:k - len(result)]:
                result.append(items[i])
            break
        r = rng.random() * total_w
        acc = 0.0
        pick = pool_idx[-1]
        for pos, (i, w) in enumerate(zip(pool_idx, pool_w)):
            acc += w
            if r <= acc:
                pick = i
                pool_idx.pop(pos)
                pool_w.pop(pos)
                break
        result.append(items[pick])
    return result


def sample_char_maps(char_maps: list, k: int,
                     boost_alpha: float, rng: random.Random) -> list[tuple[str, str]]:
    """近均匀采样字映射. signal = 频次 (高频字映射微弱加成).

    char_maps 是 Counter.most_common 结果: [((a,b), freq), ...].
    返回采样后的纯 (a,b) tuple 列表.
    """
    items = [ab for ab, _ in char_maps]
    signals = [float(f) for _, f in char_maps]
    return _boost_sample(items, signals, k, boost_alpha, rng)


def sample_word_maps(word_maps: list[tuple[str, str, float]], k: int,
                     boost_alpha: float, rng: random.Random) -> list[tuple[str, str, float]]:
    """近均匀采样词映射. signal = score (高分词映射微弱加成)."""
    items = word_maps
    signals = [w for _, _, w in word_maps]
    return _boost_sample(items, signals, k, boost_alpha, rng)


# ---- 种子加载 ----

def load_seed_pairs(min_score: float | None = None,
                  out_dir=None) -> list[dict]:
    """从 scored.jsonl 读枚举路径评过分的 pair (fwd/rev/fixed), 当造句积木.

    排除 dir 以 "seed" 开头的 pair (造句产物, 可能含乱码, 喂回去会污染采样恶性循环).
    min_score 默认用 config.SEED_MIN_SCORE.
    """
    from .config import OUT_DIR, SEED_MIN_SCORE
    import json
    if min_score is None:
        min_score = SEED_MIN_SCORE
    sp = (out_dir or OUT_DIR) / "scored.jsonl"
    if not sp.exists():
        return []
    out = []
    lineno = 0
    with sp.open(encoding="utf-8") as f:
        for line in f:
            lineno += 1
            line = line.strip()
            if not line:
                continue   # 空行容忍 (末尾无换行/编辑器残留, 无害)
            # 坏 JSON 直接抛: scored.jsonl 每行都该是合法 JSON (_dump_scored 写的),
            # 出现坏行 = 写盘出错/磁盘满/并发写, 必须立刻炸出来查根因, 不能吞。
            d = json.loads(line)
            if str(d.get("dir", "")).startswith("seed"):  # seed / seed_<model> 都排除
                continue
            if d.get("score", 0) >= min_score:
                out.append(d)
    return out


# ---- 造句主流程 ----

def gen_ga_candidates(client: openai.OpenAI,
                      n: int,
                      logger: Callable[[str], None] | None = None,
                      debug_writer: Callable[[str], None] | None = None,
                      iter_seed: int = 0,
                      out_dir=None,
                      model_loggers=None,
                      existing_population=None,
                      existing_epoch: int = 0,
                      seen_s=None):
    """GA 神鹅语进化 (取代 LLM 造句). 返回 GAEpochResult.

    沿用 load_seed_pairs/extract_*/sample_* 的积木准备 (近均匀采样移民算子用),
    内部跑 mining/ga.evolve 一个 epoch:
      warm_start(续传种群) → aging淘汰 → 生成 offspring → 预筛 → 全模型 score_pairs
      打分(fitness=全模型均分) → crowding 入种群 → evict gold → 返回.

    返回的 GAEpochResult 已含全模型评过的候选 (findings/gold_hits/per_pair),
    调用方 (pipeline.iterate_seed) 直接落盘, 不再重复 score_pairs.

    iter_seed: epoch 随机种子 (一般传 iter 号), 跨 epoch 保证采到不同积木.
    existing_population/existing_epoch: 断点续传 (state.ga_population/ga_epoch).
    seen_s: 全局已评过的 S 集合 (防重复烧 LLM). 调用方传 state.ga_seen_s, 本函数加新的.
    out_dir: scored.jsonl 所在目录 (积木来源, 默认 config.OUT_DIR).
    """
    from .ga.evolve import evolve
    from .config import SEED_MIN_SCORE, OUT_DIR as _DEFAULT_OUT, LOG_DIR
    import random as _rng
    od = out_dir or _DEFAULT_OUT
    seed_pairs = load_seed_pairs(SEED_MIN_SCORE, out_dir=od)
    all_chars = extract_char_maps(seed_pairs)
    all_words = extract_word_maps(seed_pairs)
    if len(all_chars) < 3 or len(all_words) < 3:
        if logger:
            logger(f"\n  [GA] 种子太少 (字映射{len(all_chars)}/词映射{len(all_words)}, "
                   f"score>={SEED_MIN_SCORE}), 跳过\n")
        return None
    rng = _rng.Random(iter_seed)
    return evolve(client, seed_pairs, all_chars, all_words, n, iter_seed,
                  logger=logger or (lambda s: None), debug_writer=debug_writer,
                  model_loggers=model_loggers,
                  existing_population=existing_population,
                  existing_epoch=existing_epoch, seen_s=seen_s, rng=rng)

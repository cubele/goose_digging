# -*- coding: utf-8 -*-
"""神鹅语候选枚举: 一边词典锚定 + 另一边字级可读预筛.

设计依据 (经实测确立):
  1. 静态「判词」信号 (wordfreq 词级 zipf / jieba 整词) 对二字组合无判别力.
     zipf 度量的是字串作为子串的频率 (本质是单字频率乘积), 不是词性:
     办话=4.05 / 赶班=3.38 / 汤欧=3.22 —— 真词和噪声 zipf 完全交叉.
     jieba 对二字串几乎全认 (HMM), 亦无判别力.
  2. 唯一可靠信号: 字级 zipf (字认不认得) + 静态词典收录 (真词锚).
  3. "是不是通顺词" 这个语用判断只能 LLM 做 (已验证 LLM 精准区分
     赶班(通) vs 猖冉(不通)). 分两级: T1 (fluency.py) 判通顺砍噪声,
     T2 (scoring.py) 只对通顺的判关系.

所以本模块只做:
  - 字典正向: S 在词典 (真词锚), G=goose(S), G 做字级可读预筛 (砍生僻字).
  - 字典逆向: T 在词典 (真词锚), S=ungoose(T), S 做字级可读预筛.
  - 不做 G/S 的「是否通顺词」判断 (交给 T1/T2 LLM).
  覆盖 "一边在词典" 的目标; 两边都不在词典的盲区靠
  T1 LLM 通顺判定 (fluency.py) + 种子延展 (seed.py) 补.

所有产出保证 goose(left)==right (程序校验, 零误差).
"""
from __future__ import annotations

import random
from itertools import product
from typing import Iterator

from goose_digging.oracle import build_inverse, goose, is_traditional
from .readability import char_zipf
from .wordlist import load_dictionary

# 字长过滤 (枚举只产短句, 长句走种子延展, 见 seed.py)
MIN_LEN = 2
MAX_LEN = 4
# 字级可读阈值: left/right 每个字 zipf>=此值才保留 (砍含生僻字的组合).
# 字级 zipf 是真信号 (字认不认得), 与词级 zipf (假信号) 不同.
CHAR_ZIPF_THRESH = 2.0
# 随机但确定性的枚举顺序 (固定 seed, 跨 run cursor 断点续一致).
_SHUFFLE_RNG = random.Random(719260817)


def _valid_pair_word(s: str) -> bool:
    """s 长度在 [MIN_LEN, MAX_LEN] 且 全 CJK. 枚举函数共用."""
    return MIN_LEN <= len(s) <= MAX_LEN and all("\u4e00" <= c <= "\u9fff" for c in s)


def _chars_readable(s: str, thresh: float = CHAR_ZIPF_THRESH) -> bool:
    """s 的每个字都是常用字 (字级 zipf>=thresh). 砍含生僻字的组合."""
    return all(char_zipf(c) >= thresh for c in s)


def _ungoose_valid(target: str, inv: dict[str, list[str]]) -> str | None:
    """找 src 使 goose(src)==target (真正的反查).

    ungoose 不是 goose 的真逆 (goose 多对一, build_inverse 首选源未必满足).
    逐字遍历 target 每个字符的所有候选源做笛卡尔积, 返回第一个使
    goose(src)==target 的 src; 找不到返回 None. 限制组合爆炸: 每字最多试前2源.
    候选源过滤繁体 (只用简体/共享字造现代汉语).
    """
    char_options = []
    for c in target:
        srcs = inv.get(c, [c])
        cand = [s for s in srcs if goose(s) == c
                and not is_traditional(s)][:2]
        if not cand:
            return None
        char_options.append(cand)
    for combo in product(*char_options):
        src = "".join(combo)
        if goose(src) == target:
            return src
    return None


def enumerate_forward(words) -> Iterator[tuple[str, str, str]]:
    """正向: S 在词典 (真词锚), G=goose(S), G 字级可读 -> yield (S, G, 'fwd')."""
    for S in words:
        if not _valid_pair_word(S):
            continue
        G = goose(S)
        if G == S:
            continue  # 跳过全不动点 (fixed 类由 enumerate_fixed 单独处理)
        if _chars_readable(G):
            yield S, G, "fwd"


def enumerate_reverse(words, inv: dict[str, list[str]]) -> Iterator[tuple[str, str, str]]:
    """逆向: T 在词典 (真词锚), src=ungoose(T), src 字级可读 -> yield (src, T, 'rev')."""
    for T in words:
        if not _valid_pair_word(T):
            continue
        src = _ungoose_valid(T, inv)
        if src is None or src == T:
            continue
        if _chars_readable(src):
            yield src, T, "rev"


def enumerate_fixed(words) -> Iterator[tuple[str, str, str]]:
    """不动点: S 在词典 且 goose(S)==S -> yield (S, S, 'fixed').

    fixed 类神鹅语 (舔脚→舔脚 这种本不该是不动点却恰好是). 单独一路,
    因为正/逆向都跳过了 G==S.
    """
    for S in words:
        if not _valid_pair_word(S):
            continue
        if goose(S) == S:
            yield S, S, "fixed"


def enumerate_pairs() -> list[dict]:
    """合并三路枚举 (正向/逆向/不动点) + 去重, 返回 [{left, right, dir}].

    保证 goose(left)==right. 候选量以实际输出为准 (百万级), 由 pipeline 分批送 LLM 评分.
    """
    words = load_dictionary()
    inv = build_inverse()
    seen = set()
    out = []

    def _add(S, G, d):
        key = (S, G)
        if key in seen:
            return
        assert goose(S) == G, f"枚举违反 goose 不变量: {S} -> {G}"
        seen.add(key)
        out.append({"left": S, "right": G, "dir": d})

    for S, G, d in enumerate_forward(words):
        _add(S, G, d)
    for S, G, d in enumerate_reverse(words, inv):
        _add(S, G, d)
    for S, G, d in enumerate_fixed(words):
        _add(S, G, d)
    # 随机但确定性 (固定 seed): 每个 batch 均匀覆盖不同语义场, 跨 run cursor 一致.
    _SHUFFLE_RNG.shuffle(out)
    return out


def enumerate_both_in_dict() -> list[dict]:
    """双边词典 pair: S∈词典 且 goose(S)∈词典. 零噪声高精度子集.

    两边都在静态词典 = 通顺的强保证, 可跳过 T1 通顺筛直接进 T2 评分.
    规模以实际输出为准 (dictionary_stats() 可查). 返回 [{left, right, dir}].
    """
    words = load_dictionary()
    seen = set()
    out = []
    for S in words:
        if not _valid_pair_word(S):
            continue
        G = goose(S)
        if G == S or (S, G) in seen:
            continue
        if _chars_readable(G) and G in words:
            assert goose(S) == G
            seen.add((S, G))
            out.append({"left": S, "right": G, "dir": "fwd"})
    _SHUFFLE_RNG.shuffle(out)
    return out

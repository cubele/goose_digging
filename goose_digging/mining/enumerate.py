# -*- coding: utf-8 -*-
"""神鹅语候选枚举: 惰性现场算 pair (cursor 走预 shuffle 的词表).

设计:
  词表固定 seed shuffle 一次 (跨 run 顺序一致, 保证 cursor 续跑一致)。
  pair 不预生成 —— `goose(S)` 是确定性查表, cursor 走到哪个词就算哪个词的 pair。
  这样 `enumerate_pairs()` 瞬间返回 (只 shuffle 词表), 真 pair 计算摊到 cursor
  推进时 (每批候选 × us 级单词 = ms 级, 无感)。原版预生成 170万 list 要 18-24s。

  通过 `_LazyPairs` 类实现: 对外像 list (len() + [i] + 切片), pipeline/_next_batch
  零改动; 对内 `__getitem__(i)` 现场算第 i 个词的 pair。

判据 (经实测确立):
  1. 静态「判词」信号 (wordfreq 词级 zipf / jieba 整词) 对二字组合无判别力。
  2. 唯一可靠信号: 字级 zipf (字认不认得) + 静态词典收录 (真词锚)。
  3. "是不是通顺词" 这个语用判断只能 LLM 做 (预筛 fluency + 评分 scoring)。

所以本模块只做:
  - 字典正向: S 在词典, G=goose(S), G 字级可读预筛。
  - 字典逆向: T 在词典, src=ungoose(T), src 字级可读预筛。
  - 不动点: S 在词典 且 goose(S)==S。
  不做 G/S 的「是否通顺词」判断 (交给预筛/评分 LLM)。

所有产出保证 goose(left)==right (build_inverse 保证 inv 里的源 goose(src)==target)。
"""
from __future__ import annotations

import random
from functools import lru_cache
from itertools import product
from typing import Callable, Iterator

from goose_digging.oracle import build_inverse, goose, is_traditional
from .readability import char_zipf
from .wordlist import load_dictionary

# 字长过滤 (枚举只产短句, 长句走种子延展, 见 seed.py)
MIN_LEN = 2
MAX_LEN = 4
# 字级可读阈值: left/right 每个字 zipf>=此值才保留 (砍含生僻字的组合)。
CHAR_ZIPF_THRESH = 2.0
# 固定 seed shuffle 词表 (跨 run cursor 一致, 保证 cursor 处的 pair 跨 run 可重现)。
_SHUFFLE_SEED = 719260817


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


# ---------------------------------------------------------------------------
# 单词 → pair 现场计算 (三路: fwd/rev/fixed)
# ---------------------------------------------------------------------------

def _word_to_pairs_full(w: str, words: set, inv: dict) -> list[dict]:
    """一个词能形成的全部 pair (fwd + rev + fixed). 现场算, us 级.

    三路 (对单个词 w):
      fwd:   S=w, G=goose(w), 若 G!=w 且 G 字级可读 → (w, G, 'fwd')
      rev:   T=w, src=ungoose(w), 若 src 存在且 !=w 且 src 字级可读 → (src, w, 'rev')
      fixed: 若 goose(w)==w → (w, w, 'fixed')
    跨词去重由 pipeline 的 seen_pairs 负责 (cursor=词索引, 同词不会被重访)。
    """
    if not _valid_pair_word(w):
        return []
    out = []
    G = goose(w)
    if G != w and _chars_readable(G):
        out.append({"left": w, "right": G, "dir": "fwd"})
    # rev: w 当 target, 反查 src
    src = _ungoose_valid(w, inv)
    if src is not None and src != w and _chars_readable(src):
        out.append({"left": src, "right": w, "dir": "rev"})
    # fixed
    if G == w:
        out.append({"left": w, "right": w, "dir": "fixed"})
    return out


def _word_to_pairs_bidict(w: str, words: set, inv: dict) -> list[dict]:
    """双边词典 pair: S=w∈词典 且 goose(w)∈词典. 零噪声高精度子集.

    只算 fwd + 双边词典命中 (goose(w) 也在词典)。比 full 快 (无 rev 笛卡尔积)。
    """
    if not _valid_pair_word(w):
        return []
    G = goose(w)
    if G == w:
        return []
    if _chars_readable(G) and G in words:
        return [{"left": w, "right": G, "dir": "fwd"}]
    return []


# ---------------------------------------------------------------------------
# 词表加载 + 固定 seed shuffle (缓存, 跨 run 一致)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shuffled_words() -> list[str]:
    """加载词典 → list → 固定 seed shuffle. 跨 run 顺序一致 (cursor 续跑基础).

    set 迭代顺序跨进程不保证 (hash 随机化), 故必须固定 seed shuffle 一次固化顺序。
    """
    ws = list(load_dictionary())
    random.Random(_SHUFFLE_SEED).shuffle(ws)
    return ws


@lru_cache(maxsize=1)
def _words_set() -> frozenset[str]:
    """词典 frozenset (成员查询用, O(1))."""
    return frozenset(load_dictionary())


@lru_cache(maxsize=1)
def _inverse() -> dict[str, list[str]]:
    """goose 逆映射 (rev 路用). 缓存."""
    return build_inverse()


# ---------------------------------------------------------------------------
# 惰性 pair 序列: 对外像 list, 对内现场算
# ---------------------------------------------------------------------------

class _LazyPairs:
    """惰性 pair 序列: 持有 shuffled 词表 + word_to_pairs 函数。

    对外表现得像 list[dict]:
      len(seq)         → 词表长度 (瞬间, 不算 pair)
      seq[i]           → 第 i 个词的 pair (现场算, us 级)
      seq[i:j]         → 切片 (现场算一段)
    pipeline._next_batch 用 `all_pairs[cursor]` 单点取 + `len(all_pairs)` 判尽,
    都走 __getitem__ / __len__, 零接口改动。

    cursor 续跑语义: cursor = 词索引。词表固定 seed shuffle, 跨 run 一致;
    同词现场算出的 pair 确定 (goose 是查表), 故 cursor 处的 pair 跨 run 一致。
    """

    def __init__(self, words: list[str], word_to_pairs: Callable[[str, frozenset, dict], list[dict]]):
        self._words = words
        self._w2p = word_to_pairs
        # 共享缓存的单例 (避免每次 __getitem__ 重查)
        self._words_set = _words_set()
        self._inv = _inverse()

    def __len__(self) -> int:
        return len(self._words)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [p for w in self._words[idx]
                    for p in self._w2p(w, self._words_set, self._inv)]
        w = self._words[idx]
        return self._w2p(w, self._words_set, self._inv)

    def __iter__(self) -> Iterator[dict]:
        for w in self._words:
            for p in self._w2p(w, self._words_set, self._inv):
                yield p

# ---------------------------------------------------------------------------
# 对外入口 (返回 _LazyPairs, 瞬间; pair 现场算)
# ---------------------------------------------------------------------------

def enumerate_pairs() -> _LazyPairs:
    """全量枚举 (fwd + rev + fixed), 返回惰性序列 _LazyPairs。

    瞬间返回 (只 shuffle 词表, 不算 pair)。pair 在 pipeline cursor 推进时现场算。
    保证 goose(left)==right。跨 run cursor 一致 (词表固定 seed shuffle)。
    跨词去重由 pipeline seen_pairs 负责。
    """
    return _LazyPairs(_shuffled_words(), _word_to_pairs_full)


def enumerate_both_in_dict() -> _LazyPairs:
    """双边词典 pair (S∈词典 且 goose(S)∈词典), 返回惰性序列 _LazyPairs。

    零噪声高精度子集, 两边都在词典 = 通顺强保证, 跳过预筛直接评分。
    瞬间返回, pair 现场算。
    """
    return _LazyPairs(_shuffled_words(), _word_to_pairs_bidict)

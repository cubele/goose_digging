# -*- coding: utf-8 -*-
"""可信中文词表加载与合并 (七源, 全部来自 academic/open 数据).

数据来源 (随包分发, 见 goose_digging/data/):
  1. CustomPinyinDictionary — 自定义拼音输入法词库 (~150万词)
     https://github.com/wuhgit/CustomPinyinDictionary
  2. ci.txt — 汉语词典 (26.4万词, 从 ci.json 转换)
  3. 山间新月通用语料 — Jinghang 输入法词库 (7.3万词)
     https://github.com/kkhkl/Jinghang-Dictionary
  4. idiom.txt — 成语词典 (3万, 从 idiom.json 转换)
  5. THUOCL — 清华大学开放中文词库 (8.4万, 10个分类)
     https://github.com/thunlp/THUOCL
  6. jieba 词典 — wordfreq 包自带的 jieba_zh.txt (3.5万)
  7. wordfreq top_n_list('zh', N) — subtitles/wikipedia 频率统计

合并去重后规模以 dictionary_stats() 实际输出为准 (docstring 不写死数字防漂移).
盲区词 (赶班/撬松 这类真实通顺但词典没收的) 由 T1 (fluency.py) 的 LLM 通顺判定覆盖.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import wordfreq
from wordfreq import top_n_list

_PKG_DIR = Path(__file__).resolve().parents[1]   # goose_digging/ (包)
_DATA_DIR = _PKG_DIR / "data"
# wordfreq 自带的 jieba 词典 (词 + 频率), 与 wordfreq 同目录的 data/
_JIEBA_PATH = Path(wordfreq.__file__).parent / "data" / "jieba_zh.txt"

# wordfreq top N (主词典, 基于 subtitles/wikipedia 频率)
WORDFREQ_TOP_N = 200000


def _is_cjk_word(w: str, min_len: int = 2) -> bool:
    """≥min_len 字 且 全 CJK."""
    return len(w) >= min_len and all("\u4e00" <= c <= "\u9fff" for c in w)


def _load_txt_words(filename: str) -> set[str]:
    """加载 data/ 下的纯文本词表 (一行一词, 或 '词\\t频率' / '词 频率').
    只取≥2字纯CJK词. 文件不存在返回空集.
    """
    p = _DATA_DIR / filename
    if not p.exists():
        return set()
    out: set[str] = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            # 兼容多种分隔: 空格/tab/纯词
            w = line.strip().split("\t")[0].split()[0] if line.strip() else ""
            if _is_cjk_word(w):
                out.add(w)
    return out


def _load_thuocl() -> set[str]:
    """加载所有 THUOCL 词库 (data/THUOCL_*.txt, 格式 '词\\t词频')."""
    out: set[str] = set()
    if not _DATA_DIR.exists():
        return out
    for fn in sorted(_DATA_DIR.glob("THUOCL_*.txt")):
        out |= _load_txt_words(fn.name)
    return out


def _load_jieba() -> set[str]:
    """加载 jieba 词典 (wordfreq 自带, 格式 '词 频率')."""
    if not _JIEBA_PATH.exists():
        return set()
    out: set[str] = set()
    with _JIEBA_PATH.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts and _is_cjk_word(parts[0]):
                out.add(parts[0])
    return out


def _load_wordfreq(n: int = WORDFREQ_TOP_N) -> set[str]:
    """wordfreq top_n_list, 取≥2字纯CJK词."""
    return {w for w in top_n_list("zh", n) if _is_cjk_word(w)}


@lru_cache(maxsize=1)
def load_static_dictionary() -> frozenset[str]:
    """合并七源静态词典, 返回不可变集合 (模块级缓存, 只加载一次)."""
    return frozenset(
        _load_txt_words("CustomPinyinDictionary.txt")
        | _load_txt_words("ci.txt")
        | _load_txt_words("idiom.txt")
        | _load_txt_words("sxinyue_22.txt")
        | _load_thuocl()
        | _load_jieba()
        | _load_wordfreq()
    )


def load_dictionary() -> set[str]:
    """枚举锚集 = 静态词典. (盲区词由 T1 LLM 通顺判定覆盖, 不再扩充.)"""
    return set(load_static_dictionary())


def dictionary_stats() -> dict[str, int]:
    """各源词数 (诊断用, 实际合并规模以此为准)."""
    return {
        "CustomPinyin": len(_load_txt_words("CustomPinyinDictionary.txt")),
        "ci": len(_load_txt_words("ci.txt")),
        "idiom": len(_load_txt_words("idiom.txt")),
        "山间新月": len(_load_txt_words("sxinyue_22.txt")),
        "THUOCL": len(_load_thuocl()),
        "jieba": len(_load_jieba()),
        "wordfreq_topN": len(_load_wordfreq()),
        "merged": len(load_static_dictionary()),
    }

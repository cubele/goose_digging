# -*- coding: utf-8 -*-
"""
goose oracle —— 「鳄鱼」字符转换库 (打表版).

简体中文 -> 「鳄鱼」乱码: 模拟旧版 IE 在 EUC-JP 编码的日文网页里,
简体中文经 Windows 简繁转换 (LCMapString LCMAP_TRADITIONAL_CHINESE)
转繁体, 再以 EUC-JP 编码存盘, 之后页面被 GBK 误读产生的"实意汉字乱码".

本版基于全量 oracle 打表 (21391 字符, 100% 正确), 纯查询零算法.

oracle.json 与本文件同目录, 是全量打表的产物 (一次性离线生成).
"""
from __future__ import annotations

import json
import os
import unicodedata

try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")   # 繁→简, 用于 _is_traditional 判断
    _S2T = OpenCC("s2t")   # 简→繁, 用于 goose_block 产生繁体字节再错位
except ImportError as _e:  # pragma: no cover
    _T2S = None
    _S2T = None
    _IMPORT_ERROR = _e

__version__ = "2.0.0"
__all__ = ["goose", "goose_block", "goose_char", "ungoose", "build_inverse",
           "is_traditional", "iter_oracle_keys"]

_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_DIR, "oracle.json"), encoding="utf-8") as _f:
    _ORACLE: dict[str, str] = json.load(_f)


def _is_traditional(c: str) -> bool:
    """是否繁体字 (t2s 能转换为别的字). 简体/共享字返回 False."""
    if _T2S is None:
        return False
    return _T2S.convert(c) != c


def is_traditional(c: str) -> bool:
    """是否繁体字 (公开接口)."""
    return _is_traditional(c)


def iter_oracle_keys():
    """公开迭代器: 遍历 oracle 表的所有源字 (可被 goose 转换的简体字)."""
    return iter(_ORACLE.keys())


# ---------------------------------------------------------------------------
# 核心转换 (纯查询)
# ---------------------------------------------------------------------------
def goose_char(c: str) -> str:
    """转换单个字符. 查 oracle 表; 表外字原样返回.

    >>> goose_char('为')
    '百'
    """
    return _ORACLE.get(c, c)


def goose(text: str) -> str:
    """转换整段文本 (逐字符查表). 表外字原样.

    >>> goose('为以一星期为一期')
    '百贩一星期为贩期'
    """
    return "".join(_ORACLE.get(c, c) for c in text)


def goose_block(text: str) -> str:
    """整块字节流转换 (含三字节字错位, 见 README).

    用于产生逐字符 goose() 产不出的连锁错位神鹅语.
    """
    b = text.encode("gbk")
    # 简体 -> 繁体 (s2t, 不是 t2s)
    if _S2T is not None:
        trad = _S2T.convert(text)
    else:
        trad = text
    tb = trad.encode("gbk")
    # 繁体 GBK 字节 -> EUC-JP 解码 (部分字节非法, errors='replace' 兜底)
    try:
        jap = tb.decode("euc_jp", errors="replace")
    except Exception:  # pragma: no cover
        jap = tb.decode("euc_jp", errors="ignore")
    jb = jap.encode("euc_jp", errors="replace")
    # EUC-JP 字节 -> GBK 解码 (产生"鳄鱼乱码")
    try:
        return jb.decode("gbk", errors="replace")
    except Exception:  # pragma: no cover
        return jb.decode("gbk", errors="ignore")


# ---------------------------------------------------------------------------
# 逆向 (best-effort, 多对一不可完美逆)
# ---------------------------------------------------------------------------
def ungoose(text: str) -> str:
    """best-effort 逆向: 每个 goose 后的字找最常见的源字.

    goose 多对一不可完美逆 (如 '百'<-'为'|'為'). 这里取每个目标字
    在 oracle 中出现频率最高的源 (build_inverse 返回 list[0]).
    **不保证** goose(ungoose(x)) == x. 需要严格逆用 build_inverse.

    >>> ungoose('百贩')
    '为贩'
    """
    inv = build_inverse()
    return "".join(inv.get(c, [c])[0] for c in text)


def build_inverse() -> dict[str, list[str]]:
    """构建逆映射: {target: [sources...]}, sources 按出现频率降序.

    一个 target 可能有多个 source (goose 多对一). 如 '百' <- '为'|'為'.
    用于反向枚举/查询.

    >>> build_inverse()['百'][0]
    '为'
    """
    inv: dict[str, list[str]] = {}
    for src, tgt in _ORACLE.items():
        inv.setdefault(tgt, []).append(src)
    # 排序: 非不动点源优先 (goose(src)!=src), 简体优先于繁体, 最后 unicode 稳定化.
    # 这样 ungoose 取 list[0] 时优先返回"有变换价值的简体源".
    for tgt in inv:
        inv[tgt].sort(key=lambda s: (
            goose_char(s) == s,           # 非不动点(False=0) 排前
            _is_traditional(s),           # 简体(False=0) 排前
            s,                            # unicode 稳定化
        ))
    return inv


# 模块加载时校验 opencc 是否可用 (不可用时 goose_block 降级)
if _S2T is None:  # pragma: no cover
    import warnings
    warnings.warn(f"opencc 不可用, goose_block 将退化为逐字符: {_IMPORT_ERROR}",
                  stacklevel=2)

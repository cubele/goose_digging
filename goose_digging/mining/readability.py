# -*- coding: utf-8 -*-
"""可读性判据: 一个汉字正常人认不认得.

判据 = wordfreq 的 zipf_frequency (真实现代中文字频). 与 goose 内部机制无关,
纯粹是"这字正常人看懂的概率":
  - zipf >= ~3.0: 常用 (一/中/猫/硫/鹤/矢)
  - zipf 1.5-3.0: 生僻但能认 (嗵/酢/谘/儚), 这种字反而常引起"逆天"
  - zipf < 1.5:  没人认得 / 纯乱码字 (鿫/𠀀 / PUA 区), 滤掉

枚举的字级预筛阈值 (CHAR_ZIPF_THRESH) 在 enumerate.py 定义.
"""
from __future__ import annotations

from functools import lru_cache

try:
    from wordfreq import zipf_frequency
    _HAS_WF = True
except ImportError:  # pragma: no cover
    _HAS_WF = False


@lru_cache(maxsize=60000)
def char_zipf(c: str) -> float:
    """单字 zipf 频率. 无数据返回 0.0."""
    if not _HAS_WF or not c:
        return 0.0
    try:
        return float(zipf_frequency(c, "zh"))
    except Exception:  # noqa: BLE001
        return 0.0

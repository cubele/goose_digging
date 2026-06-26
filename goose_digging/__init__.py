# -*- coding: utf-8 -*-
"""goose_digging —— 神鹅语挖掘项目包.

顶层 re-export oracle 核心转换 (goose/ungoose 等), 供外部直接用:
  from goose_digging import goose, ungoose, goose_char, ...

oracle 实现在子包 goose_digging.oracle; 挖掘管线在 goose_digging.mining.
"""
from __future__ import annotations

from .oracle import (
    goose, goose_block, goose_char, ungoose, build_inverse,
    is_traditional, iter_oracle_keys,
    __version__,
)

__all__ = ["goose", "goose_block", "goose_char", "ungoose", "build_inverse",
           "is_traditional", "iter_oracle_keys"]

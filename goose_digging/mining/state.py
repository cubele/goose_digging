# -*- coding: utf-8 -*-
"""持续挖掘的持久状态: 跨轮/跨 run, 写盘可断点续.

state.json 结构:
  seen_pairs:    [[left,right],...]   已挖 pair (去重, 跨阶段)
  gold_pairs:    [{Finding asdict},...]  高分 pair 库 (score 降序)
  n_iter:        int                  已迭代次数
  phase:         bidict|full|seed|done  当前阶段
  cursor_bidict: 双边词典阶段游标
  cursor_full:   全量枚举阶段游标
  ga_population: [{s,scores,born},...]  GA 种群 (跨 run 续, 见 mining/ga)
  ga_epoch:      int                  GA 已跑 epoch 数

"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .finding import Finding
from .config import OUT_DIR

STATE_PATH = OUT_DIR / "state.json"


@dataclass
class MiningState:
    seen_pairs: set[tuple[str, str]] = field(default_factory=set)
    gold_pairs: list[Finding] = field(default_factory=list)
    n_iter: int = 0
    phase: str = "bidict"  # bidict(双边词典纯评分) | full(全量+预筛) | seed(种子发掘) | done
    cursor_bidict: int = 0
    cursor_full: int = 0
    # GA 神鹅语进化状态 (seed 阶段用, mining/ga)
    ga_population: list[dict] = field(default_factory=list)
    ga_epoch: int = 0
    ga_seen_s: list[str] = field(default_factory=list)  # GA 评过的所有 S, 跨 run 防重烧 LLM

    def add_gold(self, f: Finding):
        """pair 入 gold 库 (按 score 降序). 阈值过滤由调用方做 (pipeline 用 max(scores)>=SCORE_THRESH)."""
        self.gold_pairs.append(f)
        self.gold_pairs.sort(key=lambda z: -z.score)

    def save(self, path: Path | None = None):
        if path is None:
            path = STATE_PATH  # 读模块级, monkeypatch 能改 (默认参数会绑定死)
        data = {
            "seen_pairs": [[a, b] for a, b in self.seen_pairs],
            "gold_pairs": [asdict(f) for f in self.gold_pairs],
            "n_iter": self.n_iter,
            "phase": self.phase,
            "cursor_bidict": self.cursor_bidict,
            "cursor_full": self.cursor_full,
            "ga_population": self.ga_population,
            "ga_epoch": self.ga_epoch,
            "ga_seen_s": self.ga_seen_s,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # 原子写

    @classmethod
    def load(cls, path: Path | None = None) -> "MiningState":
        if path is None:
            path = STATE_PATH
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        st = cls()
        st.seen_pairs = {(a, b) for a, b in data["seen_pairs"]}
        st.gold_pairs = [Finding.from_dict(d) for d in data["gold_pairs"]]
        st.n_iter = data["n_iter"]
        st.cursor_bidict = data["cursor_bidict"]
        st.cursor_full = data["cursor_full"]
        st.phase = data["phase"]
        st.ga_population = data["ga_population"]
        st.ga_epoch = data["ga_epoch"]
        st.ga_seen_s = data["ga_seen_s"]
        return st

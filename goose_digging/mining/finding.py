# -*- coding: utf-8 -*-
"""Finding 数据结构 + 导出 (jsonl / markdown).

一条 Finding = 一个 pair (S -> real_goose_S, 已校验 goose(S)=real_goose_S).
字段: S/real_goose_S/why/round/score/direction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .config import OUT_DIR


@dataclass
class Finding:
    """一条挖掘结果. S -> real_goose_S 即 goose(S)=real_goose_S (已程序校验)."""
    S: str                       # pair 左边 (展示用)
    real_goose_S: str            # 程序用 goose() 算的真实值 (右边)
    why: str                     # 评分时的一句话关系说明
    round: int
    score: float = 0.0
    direction: str = ""          # fwd=(S,goose(S)) / rev=(ungoose(S),S) / fixed / seed

    # 容错: 只取当前定义的字段, 多余字段自动丢弃.
    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        keep = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keep})


def export_findings(findings: list[Finding], theme: str) -> Path:
    """导出 jsonl + 人类可读 markdown, 返回 jsonl 路径."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", theme) or "all"
    base = OUT_DIR / f"findings_{safe}_{ts}"
    path = base.with_suffix(".jsonl")
    with path.open("w", encoding="utf-8") as f:
        for x in findings:
            f.write(json.dumps(asdict(x), ensure_ascii=False) + "\n")
    md = base.with_suffix(".md")
    lines = [f"# 神鹅语 {ts}",
             f"共 {len(findings)} 条\n",
             "| dir | pair (左→右=goose) | score | why |",
             "|---|---|---|---|"]
    for x in sorted(findings, key=lambda z: -z.score):
        why = x.why.replace("|", "\\|") if x.why else ""
        lines.append(f"| {x.direction} | {x.S} → {x.real_goose_S} "
                     f"| {x.score:.0f} | {why} |")
    md.write_text("\n".join(lines), encoding="utf-8")
    return path

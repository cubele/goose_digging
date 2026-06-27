# -*- coding: utf-8 -*-
"""共享辅助: 打分 JSON 解析 (score_pairs 用)."""
from __future__ import annotations

from .jsonx import parse_json_list


def parse_scores(raw: str) -> dict[str, dict]:
    """细评解析: 返回 {left: {score, why}}."""
    out: dict[str, dict] = {}
    for item in parse_json_list(raw, label="评分"):
        if not isinstance(item, dict):
            continue
        left = str(item.get("left", "")).strip()
        if not left:
            continue
        try:
            sc = int(round(float(item.get("score", 0))))
        except (TypeError, ValueError):
            sc = 0
        out[left] = {
            "score": max(0, min(10, sc)),
            "why": str(item.get("why", "")).strip(),
        }
    return out



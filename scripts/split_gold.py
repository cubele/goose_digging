# -*- coding: utf-8 -*-
"""从 scored.jsonl (全量已打分) 筛选生成四个 gold 格式文件, 按均分降序.

输出到 mined/gold_split/ (单独文件夹, 不污染 mined/ 的源数据).
四个文件 (gold 格式 = {"pair":"左→右", "score":均分, "scores":[...], "whys":[...]}):
  gold_fixed.jsonl   不动点神鹅语: left==right 且 max(scores)>=6
                      (goose 变换前后同字, 如 脱内裤→脱内裤, 靠字本身谐音/语义反差)
  gold_short.jsonl   短神鹅语:    非不动点 且 len(left)<=2 且 max(scores)>=6
                      (2字及以下的双关, 如 粪厂→实境)
  gold_medium.jsonl  中神鹅语:    非不动点 且 3<=len(left)<=4 且 max(scores)>=5
                      (3-4字中长句, 如 短内裤→没内裤)
  gold_long.jsonl    长神鹅语:    非不动点 且 len(left)>=5 且 max(scores)>=5
                      (5字+长句, 阈值放宽到5因长句普遍分偏低, 不漏金子)

不动点判定: left==right (等价 goose(left)==left).
阈值用 max(scores) (任一模型>=thresh 即可), 不是均分 —— 跟主路径 gold 入选标准一致.

用法:
  python scripts/split_gold.py             # 从 mined/scored.jsonl 生成到 mined/gold_split/
  python scripts/split_gold.py [TOP_N]     # TOP_N: 额外预览每类 top N 到终端
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goose_digging.mining.config import OUT_DIR

OUT_SUBDIR = OUT_DIR / "gold_split"   # 筛选结果单独文件夹

FIXED_THRESH = 6    # 不动点: max(scores)>=6
SHORT_THRESH = 6    # 短(<=2字): max(scores)>=6
LONG_THRESH = 5     # 中(3-4字)/长(>=5字): max(scores)>=5 (放宽容差, 不漏金子)
SHORT_MAX_LEN = 2   # 短/中分界: <=2 字算短
MEDIUM_MAX_LEN = 4  # 中/长分界: 3-4 字算中, >=5 字算长


def _to_gold(d: dict) -> dict:
    """scored.jsonl 记录 → gold 格式 {pair, score, scores, whys}."""
    scores = d.get("scores") or []
    scs = scores if scores else [int(round(d.get("score", 0)))]
    return {
        "pair": f"{d['left']}→{d['right']}",
        "score": d.get("score", sum(scs) / len(scs) if scs else 0),
        "scores": scs,
        "whys": d.get("whys") or [],
    }


def split(out_dir: Path = OUT_DIR):
    """读 out_dir/scored.jsonl, 按四类筛分 + 去重 + 均分降序, 写到 out_dir/gold_split/."""
    src = out_dir / "scored.jsonl"
    out_sub = out_dir / "gold_split"
    out_sub.mkdir(exist_ok=True)
    if not src.exists():
        print(f"✗ 找不到 {src}")
        return {}, {}, {}, {}

    fixed, short, medium, long_ = [], [], [], []
    seen: dict[str, dict] = {}   # pair → 最高分记录 (跨 dir 去重, 同 pair 取最高均分)

    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            L, R = d.get("left", ""), d.get("right", "")
            if not L:
                continue
            scores = d.get("scores") or []
            mx = max(scores) if scores else d.get("score", 0)

            # 跨 dir 去重: 同 pair 保留均分最高那条
            key = f"{L}→{R}"
            g = _to_gold(d)
            if key in seen:
                if g["score"] <= seen[key]["score"]:
                    continue
            seen[key] = g

            is_fixed = (L == R)
            n = len(L)
            if is_fixed and mx >= FIXED_THRESH:
                fixed.append(g)
            elif not is_fixed:
                if n <= SHORT_MAX_LEN and mx >= SHORT_THRESH:
                    short.append(g)
                elif 3 <= n <= MEDIUM_MAX_LEN and mx >= LONG_THRESH:
                    medium.append(g)
                elif n >= 5 and mx >= LONG_THRESH:
                    long_.append(g)

    # 均分降序
    fixed.sort(key=lambda x: -x["score"])
    short.sort(key=lambda x: -x["score"])
    medium.sort(key=lambda x: -x["score"])
    long_.sort(key=lambda x: -x["score"])

    # 写盘 (gold 格式, ensure_ascii=False 保证可读) 到 gold_split/
    def _write(rows, name):
        p = out_sub / name
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  gold_split/{name}: {len(rows)} 条")

    print(f"从 {src} 筛分 → {out_sub}/:")
    _write(fixed, "gold_fixed.jsonl")
    _write(short, "gold_short.jsonl")
    _write(medium, "gold_medium.jsonl")
    _write(long_, "gold_long.jsonl")
    return fixed, short, medium, long_


def _preview(rows: list[dict], top_n: int, label: str):
    if not rows:
        return
    print(f"\n=== {label} top {min(top_n, len(rows))} ===")
    for r in rows[:top_n]:
        mx = max(r["scores"]) if r["scores"] else 0
        print(f"  [{r['score']:.1f} / max={mx}] {r['pair']}")


def main():
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fixed, short, medium, long_ = split(OUT_DIR)
    if top_n > 0:
        _preview(fixed, top_n, "不动点神鹅语")
        _preview(short, top_n, "短神鹅语")
        _preview(medium, top_n, "中神鹅语(3-4字)")
        _preview(long_, top_n, "长神鹅语(5字+)")


if __name__ == "__main__":
    main()

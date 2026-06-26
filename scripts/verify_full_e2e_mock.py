#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真·完整 e2e (mock LLM): 复刻真实跑. 同一 state 贯穿 bidict→full→seed, 1000 轮 GA.

取代旧的 verify_e2e_mock.py (冒烟测试, 已删) + 之前的 workaround 版 (造假候选).

真复刻:
  - 调真 enumerate_both_in_dict() / enumerate_pairs() (惰性序列, 秒级返回)
  - cursor 真走词表索引, pair 现场算 (测的就是这条链)
  - bidict 跑到候选耗尽; full 设个上限 (170万全评分要 34万批×4模型, 太久, 取前 N 个词)
  - seed 跑 1000 轮
  - mock LLM 分数有梯度 (按内容, 不是全 2 分), 让 GA 有真信号

全程 mock LLM (不调真实 API), 隔离 tmp, 绝不碰真实 mined/.

验证点:
  1. 三阶段同一 state 串接: cursor/gold/seen_pairs 跨阶段累积
  2. 1000 轮 GA: 多样性不塌缩, gold 持续累积, seen_s 增长
  3. 断电续传: load state 后 epoch 推进
  4. scored.jsonl 含 fwd + seed_ga
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goose_digging.mining import config, state, pipeline  # noqa: E402
from goose_digging.mining.llm import ModelLogger  # noqa: E402
from goose_digging.mining.log import Logger  # noqa: E402

N_GA_EPOCHS = 1000
BIDICT_WORD_LIMIT = 50000   # bidict 阶段跑前 5万词 (167万全扫描要几分钟, 验证够用)
FULL_WORD_LIMIT = 5000      # full 阶段跑前 5000 词


# ---------------------------------------------------------------------------
# mock LLM: 分数有梯度 (模拟真实神鹅语分布)
# ---------------------------------------------------------------------------

_calls = {"n": 0}


def _mock_score(left):
    """按内容打分, 有梯度 (大部分低分, 少数金子). 不是全 2 分."""
    if "义" in left:
        return 9
    if "粪" in left:
        return 8
    if "盗" in left:
        return 8
    if "挺" in left or "通" in left:
        return 7
    h = sum(ord(c) for c in left) % 10
    if h <= 5:
        return 2
    if h <= 8:
        return 4
    return 6


def install_mock(monkey):
    def fake_score(client, model, system, user, **kw):
        _calls["n"] += 1
        lefts = re.findall(r'\d+\. (.+?) .', user)
        parts = [{"left": L, "score": _mock_score(L), "why": "m"} for L in lefts]
        return json.dumps({"scores": parts}, ensure_ascii=False)

    def fake_fluency(client, model, system, user, **kw):
        _calls["n"] += 1
        texts = re.findall(r'\d+\. (.+)', user)
        items = []
        for t in texts:
            if not t:
                continue
            ok = not any(b in t for b in ("乱", "码", "硬", "拼"))
            items.append({"text": t, "ok": bool(ok)})
        return json.dumps({"items": items}, ensure_ascii=False)

    from goose_digging.mining import scoring, fluency
    monkey["_s"] = scoring.llm_chat
    monkey["_f"] = fluency.llm_chat
    scoring.llm_chat = fake_score
    fluency.llm_chat = fake_fluency


def restore(monkey):
    from goose_digging.mining import scoring, fluency
    scoring.llm_chat = monkey["_s"]
    fluency.llm_chat = monkey["_f"]


def make_loggers(tmp):
    return {m["model"]: ModelLogger(tmp / ("m_" + m["model"] + ".log"))
            for m in config.SCORE_MODELS}


def run():
    tmp = Path(tempfile.mkdtemp(prefix="goose_full_e2e_"))
    out_dir = tmp / "mined"
    out_dir.mkdir()
    (out_dir / "logs").mkdir()
    sp = out_dir / "state.json"
    log = Logger(out_dir / "logs" / "run.log")

    print("=" * 72)
    print("完整 e2e (mock): 真枚举 + bidict前" + str(BIDICT_WORD_LIMIT) + "词 + full前" + str(FULL_WORD_LIMIT) + "词 + " + str(N_GA_EPOCHS) + "轮GA")
    print("=" * 72)

    monkey = {}
    install_mock(monkey)
    mls = make_loggers(out_dir)
    t0 = time.time()
    failures = []
    div_curve = []

    try:
        st = state.MiningState()

        # ====== 阶段 1: bidict 真枚举 (前 N 词) ======
        print("\n[1/3] bidict 真枚举(惰性) + 评分 (前" + str(BIDICT_WORD_LIMIT) + "词)...")
        t_b = time.time()
        bidict_pairs = pipeline_enum_bidict()
        print("  词表: " + str(len(bidict_pairs)) + " (惰性, 跑前 " + str(BIDICT_WORD_LIMIT) + ")")
        n_iter = 0
        while st.cursor_bidict < BIDICT_WORD_LIMIT and st.cursor_bidict < len(bidict_pairs):
            pipeline.iterate(None, st, bidict_pairs, log, skip_T1=True,
                             phase="bidict", model_loggers=mls,
                             out_dir=out_dir, state_path=sp)
            n_iter += 1
        print("  bidict 完成: " + str(n_iter) + " 批, cursor=" + str(st.cursor_bidict) +
              ", gold=" + str(len(st.gold_pairs)) + " (" + str(round(time.time() - t_b, 1)) + "s)")

        # ====== 阶段 2: full 真枚举 (前 N 词) ======
        print("\n[2/3] full 真枚举(惰性) + T1 + 评分 (前" + str(FULL_WORD_LIMIT) + "词)...")
        t_f = time.time()
        full_pairs = pipeline_enum_full()
        print("  词表: " + str(len(full_pairs)) + " (惰性, 跑前 " + str(FULL_WORD_LIMIT) + ")")
        n_iter = 0
        while st.cursor_full < FULL_WORD_LIMIT and st.cursor_full < len(full_pairs):
            pipeline.iterate(None, st, full_pairs, log, skip_T1=False,
                             phase="full", model_loggers=mls,
                             out_dir=out_dir, state_path=sp)
            n_iter += 1
        print("  full 完成: " + str(n_iter) + " 批, cursor=" + str(st.cursor_full) +
              ", gold=" + str(len(st.gold_pairs)) + " (" + str(round(time.time() - t_f, 1)) + "s)")

        scored_lines = (out_dir / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        n_scored = len([ln for ln in scored_lines if ln])
        print("  scored.jsonl 累积: " + str(n_scored) + " 条 (bidict+full)")

        # ====== 阶段 3: GA 1000 轮 ======
        print("\n[3/3] GA 进化 " + str(N_GA_EPOCHS) + " 轮...")
        ga_t0 = time.time()
        for ep in range(N_GA_EPOCHS):
            pipeline.iterate_seed(None, st, log, config.GA_OFFSPRING_PER_EPOCH, None,
                                  model_loggers=mls, out_dir=out_dir, state_path=sp)
            if ep % 100 == 0 or ep == N_GA_EPOCHS - 1:
                from goose_digging.mining.ga.population import Population
                from goose_digging.mining.ga.genome import Individual
                pop = Population([Individual.from_dict(d) for d in st.ga_population],
                                 config.GA_POP_SIZE, config.GA_SHARING_SIGMA, config.GA_ELITE)
                div = pop.mean_diversity()
                div_curve.append((ep, round(div, 2), len(st.gold_pairs), len(st.ga_seen_s)))
                print("  epoch " + str(ep) + ": gold=" + str(len(st.gold_pairs)) +
                      " div=" + str(round(div, 2)) +
                      " seen_s=" + str(len(st.ga_seen_s)) +
                      " (" + str(round(time.time() - ga_t0, 1)) + "s)")

        print("\nGA " + str(N_GA_EPOCHS) + "轮 耗时 " + str(round(time.time() - ga_t0, 1)) + "s")

        # ====== 验证 ======
        print("\n" + "=" * 72)
        print("验证:")
        print("=" * 72)

        if len(div_curve) >= 2:
            first, last = div_curve[0][1], div_curve[-1][1]
            print("  [多样性] 首=" + str(first) + " 末=" + str(last))
            if last < 1.0:
                failures.append("早熟: 末轮 div=" + str(last))
            else:
                print("  OK 未早熟")

        g_after = div_curve[0][2] if div_curve else 0
        g_final = len(st.gold_pairs)
        print("  [gold] 枚举后=" + str(g_after) + " GA后=" + str(g_final))
        if g_final <= g_after:
            failures.append("GA 没新增 gold")

        print("  [seen_s] " + str(len(st.ga_seen_s)))
        if len(st.ga_seen_s) < N_GA_EPOCHS:
            failures.append("seen_s 太少: " + str(len(st.ga_seen_s)))

        print("  [cursor] bidict=" + str(st.cursor_bidict) + "/" + str(BIDICT_WORD_LIMIT) +
              " full=" + str(st.cursor_full) + "/" + str(FULL_WORD_LIMIT))
        if st.cursor_bidict < BIDICT_WORD_LIMIT:
            failures.append("bidict cursor 没跑到上限: " + str(st.cursor_bidict))

        # 重新读 scored.jsonl (含 GA 产物; 之前 GA 前读的快照不含 seed_ga)
        scored_lines = (out_dir / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        n_scored = len([ln for ln in scored_lines if ln])
        dirs = {}
        for ln in scored_lines:
            d = json.loads(ln).get("dir", "")
            dirs[d] = dirs.get(d, 0) + 1
        print("  [scored] " + str(sorted(dirs.items())) + " 总" + str(n_scored) + "条")
        if "fwd" not in dirs or "seed_ga" not in dirs:
            failures.append("缺阶段产物: " + str(sorted(dirs.keys())))

        st_l = state.MiningState.load(path=sp)
        ep_b = st_l.ga_epoch
        pipeline.iterate_seed(None, st_l, log, config.GA_OFFSPRING_PER_EPOCH, None,
                              model_loggers=mls, out_dir=out_dir, state_path=sp)
        st_l2 = state.MiningState.load(path=sp)
        print("  [续传] epoch " + str(ep_b) + " -> " + str(st_l2.ga_epoch))
        if st_l2.ga_epoch != ep_b + 1:
            failures.append("续传没推进")

        print("\n  LLM 调用: " + str(_calls["n"]) + "  总耗时: " + str(round(time.time() - t0, 1)) + "s")

    finally:
        restore(monkey)
        for ml in mls.values():
            ml.close()
        log.close()

    print("\n" + "=" * 72)
    if failures:
        print("FAIL (" + str(len(failures)) + "):")
        for f in failures:
            print("  - " + f)
        print("(tmp: " + str(tmp) + ")")
        return 1
    print("PASS 完整 e2e (真枚举 + 1000轮GA + 三阶段串接 + 断电续传)")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


def pipeline_enum_bidict():
    """真 enumerate_both_in_dict (惰性)."""
    from goose_digging.mining import enumerate as enum_mod
    return enum_mod.enumerate_both_in_dict()


def pipeline_enum_full():
    """真 enumerate_pairs (惰性)."""
    from goose_digging.mining import enumerate as enum_mod
    return enum_mod.enumerate_pairs()


if __name__ == "__main__":
    sys.exit(run())

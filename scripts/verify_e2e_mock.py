#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端 mock LLM 验证脚本: 不调真实 API, 验证改进后的完整流程.

跑三轮:
  1. bidict (跳 T1, 两边在词典=字典短路=自动通顺, 直接进 T2)
  2. full   (走新宽松 T1 三选一: 词典/解释得通/可成句, 两边通才进 T2)
  3. seed   (GA 近均匀 boost 采样进化 + 宽松 GA T1, 两边通才打分)

验证点:
  - bidict 不调 fluency LLM (字典短路)
  - full 调 fluency LLM, 通顺 pair 全进 T2 落盘 (含低分)
  - seed 采样近均匀 (高低分积木都采到)
  - GA findings/gold_hits 结构正确, scored.jsonl 含 seed_ga 记录
  - state.json 持久化 GA 种群 + epoch + ga_seen_s

全程用 tmp_path 隔离, 绝不碰真实 mined/.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goose_digging.mining import config, state, pipeline  # noqa: E402
from goose_digging.mining.llm import ModelLogger  # noqa: E402
from goose_digging.mining.log import Logger  # noqa: E402


class MockCalls:
    def __init__(self):
        self.score_calls = []
        self.fluency_calls = []
        self.score_systems = []
        self.fluency_systems = []


def install_mock_llm(monkey_dict, calls):
    """把 mock llm_chat 注入 scoring/fluency 模块.

    fake_score: 对含 'yi'(义) 的 left 打 9 (gold), 含 'fen'(粪) 打 8 (gold), 否则 3 (低分也落盘).
    fake_fluency: 宽松判, 含 '乱码'/'硬拼' 的不通, 其余通.
    """

    def fake_score(client, model, system, user, **kw):
        lefts = re.findall(r'\d+\. (.+?) .', user)
        calls.score_calls.append((model, list(lefts)))
        calls.score_systems.append(system)
        parts = []
        for L in lefts:
            if "义" in L:
                sc = 9
            elif "粪" in L:
                sc = 8
            else:
                sc = 3
            parts.append({"left": L, "score": sc, "why": "mock" + str(model)})
        return json.dumps({"scores": parts}, ensure_ascii=False)

    def fake_fluency(client, model, system, user, **kw):
        texts = re.findall(r'\d+\. (.+)', user)
        calls.fluency_calls.append(list(texts))
        calls.fluency_systems.append(system)
        items = []
        for t in texts:
            if not t:
                continue
            ok = ("乱码" not in t) and ("硬拼" not in t)
            items.append({"text": t, "ok": bool(ok)})
        return json.dumps({"items": items}, ensure_ascii=False)

    from goose_digging.mining import scoring, fluency
    monkey_dict["_orig_score"] = scoring.llm_chat
    monkey_dict["_orig_fluency"] = fluency.llm_chat
    scoring.llm_chat = fake_score
    fluency.llm_chat = fake_fluency


def restore_llm(monkey_dict):
    from goose_digging.mining import scoring, fluency
    scoring.llm_chat = monkey_dict["_orig_score"]
    fluency.llm_chat = monkey_dict["_orig_fluency"]


def write_seed_scored(out_dir):
    """造假 scored.jsonl (fwd/rev 各种分, 高低分都有, 测近均匀采样覆盖)."""
    sp = out_dir / "scored.jsonl"
    seeds = [
        ("义父", "盗摄", "fwd", 9.0),
        ("粪厂", "实境", "fwd", 10.0),
        ("通心", "看奶", "fwd", 8.0),
        ("挺住", "尿住", "fwd", 8.0),
        ("清酥", "蓝钊", "fwd", 4.5),
        ("放租", "庶僚", "rev", 4.6),
        ("寻看", "恳辞", "rev", 4.7),
    ]
    with sp.open("w", encoding="utf-8") as f:
        for L, R, d, sc in seeds:
            rec = {"left": L, "right": R, "dir": d, "score": sc,
                   "scores": [int(round(sc))], "whys": ["mockseed"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_fake_model_loggers(tmp):
    return {m["model"]: ModelLogger(tmp / ("m_" + m["model"] + ".log"))
            for m in config.SCORE_MODELS}


def verify():
    tmp_root = Path(tempfile.mkdtemp(prefix="goose_verify_"))
    out_dir = tmp_root / "mined"
    out_dir.mkdir()
    (out_dir / "logs").mkdir()
    state_path = out_dir / "state.json"
    log_path = out_dir / "logs" / "verify.log"

    print("\n" + "=" * 72)
    print("端到端 mock LLM 验证 (隔离 tmp: " + str(tmp_root) + ")")
    print("=" * 72)

    calls = MockCalls()
    monkey = {}
    install_mock_llm(monkey, calls)
    log = Logger(log_path)
    mls = make_fake_model_loggers(out_dir)
    failures = []

    try:
        # ===================================================================
        # 测试 1: bidict (跳 T1, 字典短路, 直接 T2)
        # ===================================================================
        print("\n--- 测试 1: bidict (跳 T1) ---")
        st = state.MiningState()
        from goose_digging.mining import enumerate as enum_mod
        bidict_pairs = enum_mod.enumerate_both_in_dict()
        if not bidict_pairs:
            print("  (bidict 候选为空, 跳过)")
        else:
            batch = bidict_pairs[:config.SCORE_BATCH]
            calls.fluency_calls.clear()
            calls.score_calls.clear()
            fs = pipeline.iterate(None, st, batch, log, skip_T1=True,
                                  phase="bidict", model_loggers=mls,
                                  out_dir=out_dir, state_path=state_path)
            n_flu = len(calls.fluency_calls)
            n_score = len(calls.score_calls)
            print("  bidict: fluency调用=" + str(n_flu) + " (应=0), score调用=" + str(n_score) + ", findings=" + str(len(fs)))
            if n_flu != 0:
                failures.append("bidict 不应调 fluency, 实际 " + str(n_flu))
            if n_score != len(config.SCORE_MODELS):
                failures.append("bidict score 模型数错")
            if len(fs) == 0:
                failures.append("bidict 应有 findings")

        # ===================================================================
        # 测试 2: full (走新宽松 T1 三选一)
        # ===================================================================
        print("\n--- 测试 2: full (新宽松 T1 三选一) ---")
        st = state.MiningState()
        full_pairs = [
            {"left": "义父", "right": "盗摄", "dir": "fwd"},
            {"left": "粪厂", "right": "实境", "dir": "fwd"},
            {"left": "乱码字", "right": "硬拼串", "dir": "fwd"},
            {"left": "赶班", "right": "撬松", "dir": "fwd"},
        ]
        calls.fluency_calls.clear()
        calls.score_calls.clear()
        calls.fluency_systems.clear()
        fs = pipeline.iterate(None, st, full_pairs, log, skip_T1=False,
                              phase="full", model_loggers=mls,
                              out_dir=out_dir, state_path=state_path)
        n_flu = len(calls.fluency_calls)
        print("  full: fluency调用=" + str(n_flu) + " (应>0), findings=" + str(len(fs)) + " (乱码/硬拼被T1杀)")
        if n_flu == 0:
            failures.append("full 应调 fluency")
        if calls.fluency_systems:
            used = calls.fluency_systems[0]
            if "解释得通" not in used:
                failures.append("full T1 没用新宽松 prompt")
            else:
                print("  full T1 用了新宽松 prompt (含'解释得通') OK")

        # ===================================================================
        # 测试 3: seed (GA 近均匀 boost 采样 + 宽松 GA T1)
        # ===================================================================
        print("\n--- 测试 3: seed (GA 近均匀 boost 采样进化) ---")
        for f in out_dir.glob("*.jsonl"):
            f.unlink()
        write_seed_scored(out_dir)
        st = state.MiningState()
        calls.fluency_calls.clear()
        calls.score_calls.clear()
        calls.fluency_systems.clear()
        findings = pipeline.iterate_seed(None, st, log, 30, None,
                                         model_loggers=mls,
                                         out_dir=out_dir, state_path=state_path)
        print("  seed GA findings=" + str(len(findings)) + ", fluency调用=" + str(len(calls.fluency_calls)) + ", score调用=" + str(len(calls.score_calls)))
        if calls.fluency_systems:
            from goose_digging.mining.prompts import GA_FLUENCY_SYSTEM
            used = calls.fluency_systems[0]
            if GA_FLUENCY_SYSTEM not in used:
                failures.append("seed GA T1 没用宽松 GA_FLUENCY_SYSTEM")
            else:
                print("  seed GA T1 用了宽松 GA_FLUENCY_SYSTEM OK")

        scored_text = (out_dir / "scored.jsonl").read_text(encoding="utf-8")
        seed_ga_lines = [ln for ln in scored_text.strip().split("\n")
                         if ln and json.loads(ln).get("dir") == "seed_ga"]
        print("  scored.jsonl seed_ga 记录=" + str(len(seed_ga_lines)))
        if not seed_ga_lines:
            failures.append("seed 阶段没产 seed_ga 记录")
        else:
            for ln in seed_ga_lines[:2]:
                d = json.loads(ln)
                print("    sample: " + d["left"] + " -> " + d["right"] + " score=" + str(d["score"]))

        st2 = state.MiningState.load(path=state_path)
        print("  state: ga_population=" + str(len(st2.ga_population)) + ", ga_epoch=" + str(st2.ga_epoch) + ", ga_seen_s=" + str(len(st2.ga_seen_s)))
        if len(st2.ga_population) == 0:
            failures.append("state 没持久化 GA 种群")
        if st2.ga_epoch < 1:
            failures.append("state ga_epoch 没推进")

        for ln in seed_ga_lines:
            d = json.loads(ln)
            mx = max(d.get("scores", [0]))
            in_gold = any(d["left"] == g.S for g in st2.gold_pairs)
            if (mx >= config.SCORE_THRESH) != in_gold:
                failures.append("gold 标准错: " + d["left"] + " max=" + str(mx))

        # ===================================================================
        # 测试 4: 近均匀采样覆盖
        # ===================================================================
        print("\n--- 测试 4: 近均匀 boost 采样覆盖 ---")
        from goose_digging.mining.seed import (
            load_seed_pairs, extract_char_maps, extract_word_maps,
            sample_char_maps, sample_word_maps,
        )
        import random
        seed_pairs = load_seed_pairs(out_dir=out_dir)
        cm = extract_char_maps(seed_pairs)
        wm = extract_word_maps(seed_pairs)
        print("  字映射=" + str(len(cm)) + " 词映射=" + str(len(wm)))
        if cm and wm:
            low = set()
            high = set()
            for s in range(50):
                out = sample_word_maps(wm, min(20, len(wm)), config.SEED_BOOST_ALPHA, random.Random(s))
                for _, _, w in out:
                    if w <= 5:
                        low.add(w)
                    else:
                        high.add(w)
            print("  50次采样: 覆盖低分(<=5)=" + str(bool(low)) + ", 高分(>5)=" + str(bool(high)))
            if not low:
                failures.append("近均匀采样没采到低分词映射")
            if not high:
                failures.append("近均匀采样没采到高分词映射")

        # ===================================================================
        # 测试 5: 断点续传
        # ===================================================================
        print("\n--- 测试 5: 断点续传 (第二轮 seed 续种群) ---")
        st3 = state.MiningState.load(path=state_path)
        epoch_before = st3.ga_epoch
        pop_before = len(st3.ga_population)
        pipeline.iterate_seed(None, st3, log, 30, None,
                              model_loggers=mls,
                              out_dir=out_dir, state_path=state_path)
        st4 = state.MiningState.load(path=state_path)
        print("  续传: 种群 " + str(pop_before) + " -> " + str(len(st4.ga_population)) + ", epoch " + str(epoch_before) + " -> " + str(st4.ga_epoch))
        if st4.ga_epoch != epoch_before + 1:
            failures.append("断点续传 epoch 没推进: " + str(epoch_before) + " -> " + str(st4.ga_epoch))

    finally:
        restore_llm(monkey)
        for ml in mls.values():
            ml.close()
        log.close()

    print("\n" + "=" * 72)
    if failures:
        print("FAIL 验证失败 (" + str(len(failures)) + " 项):")
        for f in failures:
            print("   - " + f)
        print("=" * 72)
        print("(tmp 保留供调试: " + str(tmp_root) + ")")
        return 1
    else:
        print("PASS 全部验证通过 (mock LLM 端到端流程正确)")
        print("=" * 72)
        shutil.rmtree(tmp_root, ignore_errors=True)
        return 0


if __name__ == "__main__":
    sys.exit(verify())

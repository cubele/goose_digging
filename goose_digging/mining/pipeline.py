# -*- coding: utf-8 -*-
"""挖掘管线主循环: T1 攒批 → T2 多模型评分 → 全量落盘.

两条路径:
  iterate      枚举驱动 (字典双向候选 → T1 → T2), 覆盖短词.
  iterate_seed 种子发掘 (scored.jsonl→字映射→LLM造长句→T2), 覆盖长句.

输出 (mined/):
  scored.jsonl   所有评过分的 pair (含低分) + 各模型分, 全量追加.
  gold.jsonl     任一模型打分>=SCORE_THRESH 的 pair (供人工看).
  state.json     gold 库 + seen/cursor (断点续传).
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import openai

from .log import Logger
from .finding import Finding
from .state import MiningState
from .config import SCORE_THRESH, SCORE_BATCH, T1_FETCH, T1_PARALLEL, STAR_TOP_OFFSET
from .prompts import SEED_SCORE_SYSTEM
from .fluency import fluency_filter
from .scoring import score_pairs
from .llm import ModelLogger




def _cursor_of(state: MiningState, phase: str) -> int:
    """按 phase 选对应 cursor (bidict/full 独立)."""
    return state.cursor_bidict if phase == "bidict" else state.cursor_full


def _set_cursor(state: MiningState, phase: str, val: int):
    """按 phase 写回 cursor."""
    if phase == "bidict":
        state.cursor_bidict = val
    else:
        state.cursor_full = val


def _next_batch(state: MiningState, all_pairs: list[dict],
                batch_size: int, phase: str = "bidict") -> list[dict]:
    """从 all_pairs 取下一批未见 pair. cursor 按 phase 选 (bidict/full 独立)."""
    cursor = _cursor_of(state, phase)
    batch = []
    while cursor < len(all_pairs) and len(batch) < batch_size:
        p = all_pairs[cursor]
        cursor += 1
        if (p["left"], p["right"]) not in state.seen_pairs:
            batch.append(p)
    _set_cursor(state, phase, cursor)
    return batch


def _dump_scored(findings: list[Finding], per_pair_scores: dict | None = None,
                  out_dir: Path = None):
    """全量评分落盘: 每个 pair 的 score/why/dir + 各模型分.

    追加写 scored.jsonl (跨 run 累积). 含所有评过分的 pair (含低分),
    供调试/审视/分布分析. per_pair_scores: {left: {scores:[...], whys:[...]}}
    (顺序同 SCORE_MODELS), 落盘含全部模型评论.
    """
    from .config import OUT_DIR as _DEFAULT_OUT
    sp = (out_dir or _DEFAULT_OUT) / "scored.jsonl"
    with sp.open("a", encoding="utf-8") as f:
        for fd in findings:
            rec = {
                "score": round(fd.score, 2), "left": fd.S, "right": fd.real_goose_S,
                "why": fd.why, "dir": fd.direction,
                "round": fd.round,
            }
            if per_pair_scores and fd.S in per_pair_scores:
                rec["scores"] = per_pair_scores[fd.S]["scores"]
                rec["whys"] = per_pair_scores[fd.S]["whys"]
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")


def _score_and_log(client, survivors, rnd, state, stream, debug_writer,
                   model_loggers: dict[str, ModelLogger] | None = None,
                   tag="", system: str | None = None,
                   out_dir: Path | None = None, state_path: Path | None = None):
    """共用: T2 评分 + 入 gold + 日志 + 全量落盘. 返回 findings.

    system: 覆默认 SCORE_SYSTEM (seed 用 SEED_SCORE_SYSTEM, 通顺门槛更严).
    评分后立即全量落盘 (scored.jsonl), 含所有 pair 的 score/why + 各模型分.
    model_loggers: {model名: ModelLogger} 一个 run 复用一组, 注入给 score_pairs.
    """
    findings, gold_hits, per_pair_scores = score_pairs(
        client, survivors, rnd, stream, debug_writer,
        model_loggers=model_loggers,
        **({"system": system} if system else {}))
    # gold 标准: 任一模型 max(scores)>=SCORE_THRESH 就入 gold (跟 gold.jsonl 一致, 不用 avg)
    for f in findings:
        pps = per_pair_scores.get(f.S, {})
        max_score = max(pps["scores"]) if pps.get("scores") else f.score
        if max_score >= SCORE_THRESH:
            state.add_gold(f)
        star = "✨" if f.score >= SCORE_THRESH + STAR_TOP_OFFSET else ("★" if f.score >= SCORE_THRESH else "·")
        log_tag = tag or f.direction
        stream(f"\n    {star} [{f.score:.1f}/10] [{log_tag}] "
               f"{f.S} → {f.real_goose_S}")
        if f.why:
            stream(f"        why: {f.why}")
    from .config import OUT_DIR as _DEFAULT_OUT
    _od = out_dir or _DEFAULT_OUT
    state.n_iter = rnd
    state.save(path=state_path)
    _dump_scored(findings, per_pair_scores, _od)
    return findings


def iterate(client: openai.OpenAI, state: MiningState,
            all_pairs: list[dict], log: Logger, skip_T1: bool,
            debug_writer: Callable[[str], None] | None = None,
            phase: str = "bidict",
            model_loggers: dict[str, ModelLogger] | None = None,
            out_dir: Path | None = None, state_path: Path | None = None) -> list[Finding]:
    """枚举驱动迭代 -> T2 -> gold.

    skip_T1=True (bidict): 直接取 SCORE_BATCH 候选进 T2 (双边词典已自带通顺保证).
    skip_T1=False (full):  T1 持续筛候选, 攒满 SCORE_BATCH 存活才进 T2.
    phase: "bidict" | "full", 决定读哪个 cursor.
    model_loggers: 一个 run 复用的一组 ModelLogger, 透传给 _score_and_log.
    """
    stream = log.stream
    rnd = state.n_iter + 1
    cur = _cursor_of(state, phase)
    goal = f"T1攒满{SCORE_BATCH}存活进T2" if not skip_T1 else f"取{SCORE_BATCH}候选直接进T2"
    log.log(f"\n{'='*72}\n[iter {rnd} {phase}]  {goal}"
            f"  cursor={cur}/{len(all_pairs)}  gold={len(state.gold_pairs)}\n{'='*72}")

    if not skip_T1:
        survivors: list[dict] = []
        t1_rounds = 0
        n_judged = 0  # 累计已判候选数 (进 fluency_filter 的)
        # 并行 T1: 每轮预取 T1_PARALLEL 个 FETCH 批, 并行判通顺, 合并存活.
        # 通顺判不依赖顺序 (短语独立), 可安全并行.
        while len(survivors) < SCORE_BATCH and cur < len(all_pairs):
            # 预取本轮的并行批 (每批 T1_FETCH, 共 T1_PARALLEL 批)
            batches = []
            for _ in range(T1_PARALLEL):
                if cur >= len(all_pairs):
                    break
                batch = _next_batch(state, all_pairs, T1_FETCH, phase)
                cur = _cursor_of(state, phase)
                if not batch:
                    break
                for q in batch:
                    state.seen_pairs.add((q["left"], q["right"]))
                batches.append(batch)
            if not batches:
                break
            t1_rounds += 1
            # 并行判通顺: 各批的详细日志吞掉 (不进主 stream), 只让主 stream 收轮汇总.
            # (否则 5 批同时往 stream 写, 词典命中/送LLM/存活行全交错乱套)
            _noop = lambda _s: None  # 静默 logger: 并行批的详细日志吞掉, 避免 5 路交错乱码
            with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                futs = [pool.submit(fluency_filter, client, b, _noop, debug_writer)
                        for b in batches]
                results = [f.result() for f in futs]
            for b, res in zip(batches, results):
                n_judged += len(b)
                survivors.extend(res)
            new_surv = sum(len(r) for r in results)
            log.log(f"  T1第{t1_rounds}轮: 并行判{len(batches)}批×{T1_FETCH} "
                    f"存活+{new_surv} 累计存活{len(survivors)}/{SCORE_BATCH} "
                    f"cursor={cur}/{len(all_pairs)}")
        total = len(all_pairs)
        remain = total - cur
        rate = len(survivors) / n_judged * 100 if n_judged else 0
        log.log(f"  T1完成({t1_rounds}轮): 已判{n_judged}/{total} 剩{remain} "
                f"存活{len(survivors)} (存活率{rate:.0f}%) -> T2评分")
    else:
        batch = _next_batch(state, all_pairs, SCORE_BATCH, phase)
        for p in batch:
            state.seen_pairs.add((p["left"], p["right"]))
        survivors = batch

    if not survivors:
        log.log(f"  (枚举候选耗尽, 无存活)")
        state.n_iter = rnd
        state.save(path=state_path)
        return []

    return _score_and_log(client, survivors, rnd, state, stream, debug_writer,
                          model_loggers=model_loggers, out_dir=out_dir, state_path=state_path)


def iterate_seed(client: openai.OpenAI, state: MiningState,
                 log: Logger, n_gen: int = 20,
                 debug_writer: Callable[[str], None] | None = None,
                 model_loggers: dict[str, ModelLogger] | None = None,
                 out_dir: Path | None = None, state_path: Path | None = None) -> list[Finding]:
    """种子发掘迭代: GA 神鹅语进化 (全模型直接打分, 废弃 surrogate).

    scored.jsonl→字/词映射积木(近均匀采样)→GA进化(全模型 score_pairs 打分)→goose校验→gold/落盘.
    GA 内部已跑完全模型 cross-check, 本函数直接落 GA 返回的 findings/gold_hits.

    n_gen: 每 epoch 产多少 offspring (= SEED_N_GEN).
    model_loggers: 一个 run 复用的一组 ModelLogger, 透传给 GA 打分.
    """
    from . import seed as seed_mod
    from .config import SEED_MIN_SCORE, OUT_DIR as _DEFAULT_OUT
    stream = log.stream
    rnd = state.n_iter + 1
    seed_pairs = seed_mod.load_seed_pairs(SEED_MIN_SCORE)
    # 全局已评 S (GA 专属): 防 GA 重新生成已评过的 S 烧 LLM.
    # 用 state.ga_seen_s 持久化 (跨 run 防重烧); evolve 原地往里加本 epoch 新评的 S.
    seen_s = set(state.ga_seen_s)
    log.log(f"\n{'='*72}\n[iter {rnd} 种子发掘(GA)]  种子{len(seed_pairs)}对"
            f"(score>={SEED_MIN_SCORE})  进化{n_gen} offspring  "
            f"种群{len(state.ga_population)}/epoch{state.ga_epoch}  已评S={len(seen_s)}\n{'='*72}")

    result = seed_mod.gen_ga_candidates(
        client, n=n_gen, logger=stream, debug_writer=debug_writer,
        iter_seed=rnd, out_dir=out_dir, model_loggers=model_loggers,
        existing_population=state.ga_population or None,
        existing_epoch=state.ga_epoch, seen_s=seen_s)
    if result is None:
        log.log(f"  (未进化, 种子可能太少或 scored.jsonl 空)")
        state.n_iter = rnd
        state.save(path=state_path)
        return []

    findings = result.findings
    _od = out_dir or _DEFAULT_OUT
    if not findings:
        log.log(f"  (无打分候选, 可能全模型失败或无新 S)")
        # 即使没候选也要持久化新种群 + 全局 seen_s (进化已推进)
        state.ga_population = result.pop
        state.ga_epoch += 1
        state.ga_seen_s = list(seen_s)
        state.n_iter = rnd
        state.save(path=state_path)
        return []

    # GA 已带全模型分: 标记 seen, 入 gold, 落盘, 不重复 score_pairs
    for f in findings:
        state.seen_pairs.add((f.S, f.real_goose_S))
    log.log(f"  GA 全模型打分 {len(findings)} 条")
    for f in findings:
        pps = result.per_pair.get(f.S, {})
        max_score = max(pps["scores"]) if pps.get("scores") else f.score
        if max_score >= SCORE_THRESH:
            state.add_gold(f)
        star = "✨" if f.score >= SCORE_THRESH + STAR_TOP_OFFSET else ("★" if f.score >= SCORE_THRESH else "·")
        stream(f"\n    {star} [{f.score:.1f}/10] [{f.direction}] "
               f"{f.S} → {f.real_goose_S}")
        if f.why:
            stream(f"        why: {f.why}")
    # scored.jsonl: 用 GA per_pair (含全模型 scores/whys) 落盘
    _dump_scored(findings, result.per_pair, _od)
    state.ga_population = result.pop
    state.ga_epoch += 1
    state.ga_seen_s = list(seen_s)
    state.n_iter = rnd
    state.save(path=state_path)
    if result.stats:
        log.log(f"  [GA] offspring={result.stats.get('offspring')} "
                f"t1_killed={result.stats.get('t1_killed')} "
                f"scored={result.stats.get('scored')} gold={result.stats.get('gold')} "
                f"diversity={result.stats.get('diversity')}")
    return findings

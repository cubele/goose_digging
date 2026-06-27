# -*- coding: utf-8 -*-
"""关系评分: k 模型并行 cross-check.

每个 SCORE_MODELS 里的模型并行打分 (ThreadPoolExecutor), 取均分为最终 score.
任一模型打分>=SCORE_THRESH 的 pair 另存 gold.jsonl 供人工看 (不只看均分, 任一模型见的 gold 都留).

日志策略:
  - 每个模型的思维链实时写到该模型独占的 ModelLogger 文件
    (mined/logs/model_<ts>_<model>.log), 不进主 term.
  - 主 term 不输出思维链, 只起监视: 每 PROGRESS_HEARTBEAT_SEC 秒汇报各模型进度
    (⏳在跑 字节数/已用秒 | ✅完成 | ❌失败), 直到全部结束.
  - 一个 run 复用同一组 ModelLogger (按 model 缓存, 跨 round 追加), tail -f 友好.

混合来源说明: Finding.score 取各模型均分; why 取最高分模型的字段
(高分模型更可能看到关系, 它的解读更接近神鹅语审美), 仅作展示代表.
各模型逐条明细 (scores/whys 平行列表, 顺序同 SCORE_MODELS) 经 per_pair_scores 原样落盘,
不丢任何一个模型的评论. 单模型失败/某 pair 缺值时不污染均值 (跳过该模型该 pair).
"""
from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

import openai

from .finding import Finding
from .shared import parse_scores
from .prompts import score_prompt, SCORE_SYSTEM
from .llm import ModelLogger, llm_chat
from .config import SCORE_MODELS, SCORE_THRESH, PROGRESS_HEARTBEAT_SEC


def _score_one_model(client: openai.OpenAI, model_cfg: dict, pairs: list[dict],
                     rnd: int, label: str,
                     model_logger: ModelLogger,
                     debug_writer: Callable[[str], None] | None,
                     system: str = SCORE_SYSTEM,
                     prompt_fn=score_prompt) -> dict[str, dict]:
    """单模型打分, 返回 {left: {score, why}}.

    system / prompt_fn 可覆盖 (不同阶段不同判据用). 默认走主路径 SCORE_SYSTEM + score_prompt.
    思维链/正文流式只喂给 model_logger (该模型独占文件), 不进主 term.
    """
    sp = [{"left": p["left"], "right": p["right"]} for p in pairs]
    model_logger.section(f"\n{'='*70}\n[r{rnd}] {label}  model={model_cfg['model']}\n{'='*70}")
    raw = llm_chat(client, model_cfg["model"], system, prompt_fn(sp),
                   label=label, thinking=model_cfg["thinking"],
                   logger=model_logger.write, debug_writer=debug_writer)
    return parse_scores(raw)


def score_pairs(client: openai.OpenAI, pairs: list[dict], rnd: int,
                logger: Callable[[str], None],
                debug_writer: Callable[[str], None] | None,
                models: list[dict] | None = None,
                model_loggers: dict[str, ModelLogger] | None = None,
                system: str = SCORE_SYSTEM,
                prompt_fn=score_prompt,
                thresh: float = SCORE_THRESH,
                ) -> tuple[list[Finding], list[dict], dict]:
    """k 模型 cross-check 并行. 返回 (findings, gold_hits, per_pair_scores).

    取均分为最终 score. 任一模型打分>=thresh 的 pair 进 gold_hits (落 gold.jsonl).
    per_pair_scores: {left: {scores:[...], whys:[...]}} 顺序同 models 的平行列表,
        落盘后含全部模型的评论, 不只最高分那一个.
    models: 默认 config.SCORE_MODELS; 可传自定义扩展到任意 k.
    model_loggers: {model名: ModelLogger} 调用方注入 (一个 run 复用一组).
        缺省 (None) 时本批临时建一组, 本批结束关闭 —— 不推荐, 主调用方应注入.
    system / prompt_fn / thresh: 覆盖主路径配置 (不同阶段用不同判据). 默认走主路径.

    主 term 行为: 每 5s 汇报所有模型进度 (在跑/已完成/失败), 全部完成后给汇总.
    """
    if models is None:
        models = SCORE_MODELS
    n_models = len(models)
    owned_loggers = False  # 标记本调用是否临时自建了 model_loggers (要负责关)

    if model_loggers is None:
        model_loggers = {}
        owned_loggers = True

    # 取/建每个模型的独占 logger (按 model 名缓存, 一个 run 复用)
    for mc in models:
        mname = mc["model"]
        if mname not in model_loggers:
            from .llm import make_model_logger
            model_loggers[mname] = make_model_logger(mname)

    all_scores: list[dict] = [None] * n_models  # type: ignore
    model_errors: list[str] = [None] * n_models  # type: ignore
    model_done = [False] * n_models
    model_t0 = [0.0] * n_models

    with ThreadPoolExecutor(max_workers=n_models) as pool:
        futures = {}
        for i, mc in enumerate(models):
            mname = mc["model"]
            model_t0[i] = time.time()
            fut = pool.submit(_score_one_model, client, mc, pairs, rnd,
                              f"评分-{mname} r{rnd}",
                              model_loggers[mname], debug_writer,
                              system, prompt_fn)
            futures[fut] = i

        pending = set(futures)
        # 主 term 心跳: 每 5s 汇报所有模型进度, 直到全部结束
        while pending:
            done, pending = wait(pending, timeout=PROGRESS_HEARTBEAT_SEC, return_when=FIRST_COMPLETED)
            for fut in done:
                i = futures[fut]
                model_done[i] = True
                try:
                    all_scores[i] = fut.result()
                except Exception as e:  # 单模型失败不炸全批, 降级用其余模型
                    model_errors[i] = str(e)[:80]
                    all_scores[i] = {}
            _report_progress(logger, models, model_done, model_errors,
                             all_scores, model_t0, model_loggers, final=False)

    # 最终汇总一行 (所有都完成)
    _report_progress(logger, models, model_done, model_errors,
                     all_scores, model_t0, model_loggers, final=True)

    # 仅在自建的情况下关闭 (主调用方注入的不关, 留给 __main__ 在 run 结束时关)
    if owned_loggers:
        for ml in model_loggers.values():
            ml.close()

    # 区分: 有异常的算失败 (进 err_info); 没异常但返回空的 (截断/格式错) 仍进 valid_models,
    # 由下面条数校验显式标出来 —— 否则空返回会被静默吞掉, 既不报错也不报漏.
    valid_models = [(i, all_scores[i]) for i in range(n_models) if not model_errors[i]]
    if not valid_models:
        logger("\n  [评分] ⚠️ 所有模型都失败, 跳过本批\n")
        return [], [], {}
    ok_names = ", ".join(models[i]["model"] for i, _ in valid_models)
    err_info = ""
    failed = [f"{models[i]['model']}:{model_errors[i]}" for i in range(n_models) if model_errors[i]]
    if failed:
        err_info = f" (失败: {'; '.join(failed)})"
    logger(f"\n  [评分] {len(valid_models)}/{n_models}模型完成: {ok_names}{err_info}\n")

    # 校验每个模型回的条数: 正常应等于输入 len(pairs). 少回 = 模型漏项 / 输出被截断.
    # 漏的 pair 会被均值跳过 (静默丢数据), 这里显式标出来让人知道.
    expected = len(pairs)
    expected_lefts = {p["left"] for p in pairs}
    for i, sc in valid_models:
        got_lefts = set(sc.keys())
        missing = expected_lefts - got_lefts
        n_got = expected - len(missing)
        if n_got < expected:
            mname = models[i]["model"]
            if n_got == 0:
                logger(f"  [评分] ⚠️ {mname} 一条都没回 ({len(sc)}项) —— 输出可能被截断/格式错, "
                       f"该模型对本批无贡献\n")
            else:
                sample = ",".join(list(missing)[:5])
                more = f" 等{len(missing)}条" if len(missing) > 5 else ""
                logger(f"  [评分] ⚠️ {mname} 只回了 {n_got}/{expected} 条 "
                       f"(漏: {sample}{more})\n")

    findings = []
    gold_hits = []
    per_pair_scores: dict[str, dict] = {}
    for p in pairs:
        left = p["left"]
        d = p.get("dir", "")
        # 收集每个模型对此 pair 的评分 (顺序同 SCORE_MODELS, 跳过此 pair 缺值的模型)
        per_model = []  # [score_dict]
        for _, sc in valid_models:
            s = sc.get(left)
            if s is None or "score" not in s:
                continue
            per_model.append(s)
        if not per_model:
            continue
        scores = [s["score"] for s in per_model]
        whys = [s.get("why", "") for s in per_model]
        avg = sum(scores) / len(scores)
        best = max(per_model, key=lambda s: s["score"])
        # 逐模型明细 (scores/whys 平行列表, 顺序同 SCORE_MODELS), 落盘不丢任何一个模型的评论
        per_pair_scores[left] = {"scores": scores, "whys": whys}
        f = Finding(
            S=left, real_goose_S=p["right"],
            why=best.get("why", ""),
            round=rnd, score=avg, direction=d)
        findings.append(f)
        if max(scores) >= thresh:
            # 任一模型见 gold 就单独存: 给人看的精简记录 (pair 展示 + 均分 + 各模型分/评论)
            gold_hits.append({
                "pair": f"{left}→{p['right']}",
                "score": round(avg, 1),
                "scores": scores, "whys": whys,
            })
    return findings, gold_hits, per_pair_scores


def _report_progress(logger: Callable[[str], None],
                     models: list[dict],
                     model_done: list[bool],
                     model_errors: list[str],
                     all_scores: list,
                     model_t0: list[float],
                     model_loggers: dict[str, ModelLogger],
                     final: bool) -> None:
    """一行进度心跳. final=False: 每 5s 的中间汇报; final=True: 全部完成时的汇总.

    每个模型一个状态:
      ⏳ 在跑 (已用秒, 已写 KB)   |   ✅ 完成 (用了秒, 评了几条)   |   ❌ 失败
    所有 ⏳ 用 [..] 包起来, 让主 term 一眼看到还有谁没完.
    """
    now = time.time()
    parts = []
    n_done = 0
    n_fail = 0
    for i, mc in enumerate(models):
        mname = mc["model"]
        if model_done[i]:
            n_done += 1
            if model_errors[i]:
                n_fail += 1
                parts.append(f"❌{mname}:{model_errors[i]}")
            else:
                n_ret = len(all_scores[i]) if all_scores[i] else 0
                parts.append(f"✅{mname}({n_ret}条,{now-model_t0[i]:.0f}s)")
        else:
            kb = model_loggers[mc["model"]].bytes_written
            parts.append(f"⏳{mname}({now-model_t0[i]:.0f}s,{kb:.0f}B)")
    in_flight = len(models) - n_done
    if final:
        logger(f"  [评分] 全部完成: {n_done}/{len(models)} "
               f"({'失败 '+str(n_fail) if n_fail else 'ok'})  {' | '.join(parts)}\n")
    else:
        # 用 [..] 把 ⏳ 的部分突出来, 让主 term 看到还在等的模型
        logger(f"  [评分] 进度 {n_done}/{len(models)} "
               f"(等{in_flight})  {' | '.join(parts)}\n")

# -*- coding: utf-8 -*-
"""稳态 GA 主循环 (单层全模型打分).

每个 epoch (= 一次 iterate_seed 调用产 n_gen 个候选) 的步骤:
  1. warm_start: 种群为空时, 从 scored.jsonl 高分对的 left 串 + 近均匀采样移民补满
  2. 生成 n_gen offspring:
       immigrant  GA_IMMIGRANT_RATE   近均匀采样全新拼 (探索源)
       crossover  GA_CROSSOVER_RATE   段拼接 (锦标赛选父, sharing 调整)
       mutation   GA_MUTATION_RATE    字映射替换/重采/删/重复 (余下比例)
  3. 廉价预筛 is_viable (零 LLM): 砍全不动点/已见/生僻字乱码
  4. 全模型 score_pairs 直接打分 (4 模型 cross-check) → fitness = 全模型均分
  5. crowding 替换入种群, 更新 fitness-sharing
  6. evict gold: 全模型 max>=SCORE_THRESH 的个体存档后移出种群
  7. 早熟监控: mean_diversity 过低则下 epoch 抬 immigrant_rate (震荡)
  返回 GAEpochResult(findings/gold_hits/per_pair + 新种群), 供 pipeline 落盘.

采样策略 (与 seed.py 一致): 近均匀 + 加性小 boost (权重=1+signal*alpha).
低分积木也能采到 (低分词映射成句后可能反而高分), 保证搜索空间完整覆盖.

设计要点:
  - 种群跨 epoch 持久 (state.ga_population), warm_start 只在首次/种群空时做.
  - fitness = 全模型均分 (findings 里直接有).
  - GA 参数调宽 (POP=60/IMMIGRANT=0.40/ELITE=1/MAX_AGE=4/SHARING=2.5): 广泛覆盖、持久探索.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import openai

from goose_digging.oracle import goose
from ..config import (
    GA_POP_SIZE, GA_OFFSPRING_PER_EPOCH, GA_IMMIGRANT_RATE, GA_CROSSOVER_RATE,
    GA_MUTATION_RATE, GA_TOURNAMENT_K, GA_ELITE, GA_SHARING_SIGMA,
    GA_IMMIGRANT_BOOST_ALPHA, SEED_MIN_LEN, SEED_MAX_LEN,
    SEED_N_CHARS, SEED_N_WORDS, SEED_BOOST_ALPHA,
    WORD_MAP_LEN, SCORE_THRESH, GA_MAX_AGE, GA_EVICT_PROMOTED,
    GA_EVICT_SCORE_THRESH,
)
from ..finding import Finding
from ..llm import ModelLogger
from .genome import Individual, crossover, mutate, immigrant, is_viable
from .population import Population
from .fitness import score_offspring, ScoreResult


@dataclass
class GAEpochResult:
    """一个 epoch 的产出 (全模型评过的候选)."""
    findings: list[Finding]              # 全模型评分结果 (含 left/right/score/why)
    gold_hits: list[dict]                # max(scores)>=SCORE_THRESH (落 gold.jsonl 用)
    per_pair: dict                       # {left: {scores, whys}}
    gen_model: str                       # "seed_ga" 标 dir
    pop: list[dict]                      # 新种群 (落 state.ga_population 用)
    stats: dict                          # 监控统计 (offspring/打分/gold/diversity)


def warm_start(seed_pairs: list[dict], all_chars, all_words,
               pop_size: int, epoch: int, rng: random.Random) -> Population:
    """构造初始种群: scored.jsonl 高分对的 left 串做骨干 + 近均匀采样移民补满.

    seed_pairs: load_seed_pairs 的结果 (全部枚举 pair, 不卡分数).
    all_chars/all_words: extract_char_maps/extract_word_maps 结果 (积木).
    骨干用 left 串 (已有验证过的鹅语碎片), 去掉全不动点 (is_viable 会过).
    骨干的 fitness 用其历史 score 存进 scores 字段 (作为进化起点的初始 fitness).
    """
    inds: list[Individual] = []
    # 骨干: 现有高分对的 left 串 (3-MAX_LEN 字), 当个体直接入种群.
    # 骨干是已验证高分, 其 fitness = 历史 score (存 scores 字段, 与全模型分结构一致).
    for p in seed_pairs:
        s = p.get("left", "")
        sc = p.get("score", 0)
        if SEED_MIN_LEN <= len(s) <= SEED_MAX_LEN and goose(s) != s:
            inds.append(Individual(s=s, scores=[int(round(sc))], born=epoch))
        if len(inds) >= pop_size:
            break
    # 移民补满 (带重试上限防死循环: 积木池小/生僻字多时 immigrant 可能持续不可行)
    tries = 0
    while len(inds) < pop_size and tries < pop_size * 20:
        tries += 1
        from ..seed import sample_char_maps, sample_word_maps
        cs = sample_char_maps(all_chars, SEED_N_CHARS, SEED_BOOST_ALPHA, rng)
        ws = sample_word_maps(all_words, SEED_N_WORDS, SEED_BOOST_ALPHA, rng)
        target_len = rng.randint(SEED_MIN_LEN, SEED_MAX_LEN)
        s = immigrant(cs, ws, target_len, rng)
        if is_viable(s, SEED_MIN_LEN, SEED_MAX_LEN):
            inds.append(Individual(s=s, born=epoch))
    return Population(inds, pop_size, GA_SHARING_SIGMA, GA_ELITE)


def _gen_offspring(pop: Population, n: int, all_chars, all_words,
                   epoch: int, rng: random.Random,
                   immigrant_rate: float) -> list[Individual]:
    """生成 n 个 offspring (immigrant/crossover/mutation 按比例)."""
    from ..seed import sample_char_maps, sample_word_maps
    n_imm = int(round(n * immigrant_rate))
    n_cross = int(round(n * GA_CROSSOVER_RATE))
    n_mut = n - n_imm - n_cross  # 余下给 mutation
    if n_mut < 0:  # rate 之和 > 1 时截断 immigrant
        n_imm = max(0, n_imm + n_mut)
        n_mut = 0
    out: list[Individual] = []

    def _fresh_maps():
        return (sample_char_maps(all_chars, SEED_N_CHARS, GA_IMMIGRANT_BOOST_ALPHA, rng),
                sample_word_maps(all_words, SEED_N_WORDS, GA_IMMIGRANT_BOOST_ALPHA, rng))

    for _ in range(n_imm):
        cs, ws = _fresh_maps()
        target_len = rng.randint(SEED_MIN_LEN, SEED_MAX_LEN)
        s = immigrant(cs, ws, target_len, rng)
        if s:
            out.append(Individual(s=s, born=epoch))

    for _ in range(n_cross):
        cs, ws = _fresh_maps()
        pa = pop.tournament(GA_TOURNAMENT_K, rng)
        pb = pop.tournament(GA_TOURNAMENT_K, rng)
        s = crossover(pa.s, pb.s, SEED_MIN_LEN, SEED_MAX_LEN, rng)
        # crossover 产物小概率再变异一次 (混合)
        if s and rng.random() < 0.3:
            s = mutate(s, cs, ws, SEED_MIN_LEN, SEED_MAX_LEN, rng)
        if s:
            out.append(Individual(s=s, born=epoch))

    for _ in range(n_mut):
        cs, ws = _fresh_maps()
        parent = pop.tournament(GA_TOURNAMENT_K, rng)
        s = mutate(parent.s, cs, ws, SEED_MIN_LEN, SEED_MAX_LEN, rng)
        if s:
            out.append(Individual(s=s, born=epoch))
    return out


def _immigrant_factory(all_chars, all_words, epoch):
    """产全新探索个体的工厂 (cull/evict 后补种群用). 必须过 is_viable."""
    def _factory(r):
        from ..seed import sample_char_maps, sample_word_maps
        for _ in range(20):
            cs = sample_char_maps(all_chars, SEED_N_CHARS, GA_IMMIGRANT_BOOST_ALPHA, r)
            ws = sample_word_maps(all_words, SEED_N_WORDS, GA_IMMIGRANT_BOOST_ALPHA, r)
            s = immigrant(cs, ws, r.randint(SEED_MIN_LEN, SEED_MAX_LEN), r)
            if s and is_viable(s, SEED_MIN_LEN, SEED_MAX_LEN):
                return Individual(s=s, born=epoch)
        return None
    return _factory


def evolve(client: openai.OpenAI,
           seed_pairs: list[dict],
           all_chars,
           all_words,
           n_gen: int,
           rnd: int,
           logger: Callable[[str], None],
           debug_writer: Callable[[str], None] | None = None,
           model_loggers: dict[str, ModelLogger] | None = None,
           existing_population: list[dict] | None = None,
           existing_epoch: int = 0,
           seen_s: set[str] | None = None,
           rng: random.Random | None = None) -> GAEpochResult:
    """跑一个 GA epoch. 返回 findings/gold_hits + 新种群.

    seed_pairs/all_chars/all_words: 调用方 (seed.gen_ga_candidates) 准备好的积木.
    n_gen: 本 epoch 产多少 offspring (= SEED_N_GEN).
    existing_population/existing_epoch: 断点续传的种群 (state.ga_population/ga_epoch).
    seen_s: 全局已评过的 S 集合 (防重复烧 LLM). 调用方传 state 的已评 S, 本函数加新的.
    model_loggers: 一个 run 复用的一组全模型 ModelLogger, 透传给 score_offspring.

    防同化三件套 (每 epoch):
      cull_aged     淘汰超龄个体 (GA_MAX_AGE), 高分个体不能永久存活主导种群
      evict gold    全模型 max>=EVICT_SCORE_THRESH 的个体存档后移出种群
      全局 seen_s   评过的 S 不再重烧 LLM
    """
    rng = rng or random.Random(rnd)
    n_gen = n_gen or GA_OFFSPRING_PER_EPOCH
    epoch = existing_epoch + 1
    seen_s = seen_s if seen_s is not None else set()
    factory = _immigrant_factory(all_chars, all_words, epoch)

    # 1. 种群: 续传 or warm_start
    if existing_population:
        pop = Population([Individual.from_dict(d) for d in existing_population],
                         GA_POP_SIZE, GA_SHARING_SIGMA, GA_ELITE)
        logger(f"\n  [GA] 续种群 {len(pop)}/{GA_POP_SIZE} (epoch {epoch})\n")
    else:
        pop = warm_start(seed_pairs, all_chars, all_words,
                         GA_POP_SIZE, epoch, rng)
        logger(f"\n  [GA] warm_start 种群 {len(pop)}/{GA_POP_SIZE} "
              f"(骨干{len(seed_pairs)}对+移民)\n")
    pop.set_epoch(epoch)

    # 1b. aging: 淘汰超龄个体 (防高分个体永久存活主导种群 → 同化)
    if GA_MAX_AGE:
        n_cull = pop.cull_aged(GA_MAX_AGE)
        if n_cull:
            pop.refill(factory, rng)
            logger(f"  [GA] aging: 淘汰 {n_cull} 超龄(>{GA_MAX_AGE}epoch) "
                  f"补移民 → 种群 {len(pop)}\n")

    if len(pop) < 2:
        logger(f"\n  [GA] 种群不足 ({len(pop)}), 跳过本 epoch\n")
        return GAEpochResult([], [], {}, "seed_ga",
                             [ind.to_dict() for ind in pop.to_list()],
                             {"offspring": 0, "scored": 0, "gold": 0})

    # 2. 生成 offspring (本 epoch 用抬高后的 immigrant_rate 抗早熟)
    base_div = pop.mean_diversity()
    immigrant_rate = GA_IMMIGRANT_RATE
    if base_div < 2.0:   # 早熟: 抬高移民率震荡
        immigrant_rate = min(0.6, GA_IMMIGRANT_RATE + 0.2)
        logger(f"  [GA] 早熟告警 (diversity={base_div:.1f}<2), 抬移民率→{immigrant_rate:.2f}\n")
    offspring = _gen_offspring(pop, n_gen, all_chars, all_words, epoch, rng,
                               immigrant_rate)

    # 3. 廉价预筛 (零 LLM) + 全局去重 (评过的 S 不再烧 LLM)
    viable = [o for o in offspring
              if o.s and is_viable(o.s, SEED_MIN_LEN, SEED_MAX_LEN, seen_pairs=None)]
    pop_s = {ind.s for ind in pop.to_list()}
    seen_batch: set[str] = set()
    dedup: list[Individual] = []
    n_seen_skip = 0
    for o in viable:
        if o.s in pop_s or o.s in seen_batch:
            continue
        if o.s in seen_s:
            n_seen_skip += 1
            continue
        seen_batch.add(o.s)
        dedup.append(o)
    logger(f"  [GA] offspring {len(offspring)} → 预筛存活 {len(viable)} "
          f"→ 去重 {len(dedup)} (全局已评跳过 {n_seen_skip})\n")
    if not dedup:
        logger(f"  [GA] 本 epoch 无新候选, 仅持久化种群\n")
        return GAEpochResult([], [], {}, "seed_ga",
                             [ind.to_dict() for ind in pop.to_list()],
                             {"offspring": len(offspring), "scored": 0, "gold": 0})

    # 3b. 通顺预筛 (便宜模型关思考, 成本低): 两边都通才进全模型, 省 4x token.
    #     GA 产物多为字映射随机拼, 大量"左边或右边读不通/堆砌"的串, 喂全模型=烧钱.
    #     复用 fluency.fluency_filter (两边都通才存活, 词典优先省 LLM).
    #     用 GA 专属宽松 prompt (GA_FLUENCY_SYSTEM): GA 候选本是谐音/联想/口语拼串,
    #     严格"通顺可懂"会误杀真·神鹅语 (吊挺吗=屌挺嘛). 只杀明显乱码/字硬拼的.
    from ..fluency import fluency_filter
    from ..prompts import GA_FLUENCY_SYSTEM
    fluency_pairs = [{"left": ind.s, "right": goose(ind.s)} for ind in dedup]
    n_before_fluency = len(fluency_pairs)
    survivors_pairs = fluency_filter(client, fluency_pairs, logger, debug_writer,
                                     system=GA_FLUENCY_SYSTEM)
    survivor_lefts = {p["left"] for p in survivors_pairs}
    n_fluency_killed = n_before_fluency - len(survivors_pairs)
    dedup = [ind for ind in dedup if ind.s in survivor_lefts]
    logger(f"  [GA] 通顺预筛: {n_before_fluency} → {len(dedup)} (两边都通才留, 杀 {n_fluency_killed})\n")
    if not dedup:
        logger(f"  [GA] 预筛后无候选, 仅持久化种群\n")
        return GAEpochResult([], [], {}, "seed_ga",
                             [ind.to_dict() for ind in pop.to_list()],
                             {"offspring": len(offspring), "scored": 0, "gold": 0,
                              "fluency_killed": n_fluency_killed})

    # 4. 全模型 score_pairs 直接打分 (fitness = 全模型均分)
    #    一把全评, 不子分批: 存活数受 fluency 自然约束 (~十几个, 实测峰值几十),
    #    远低于输出截断风险线, 同 full 模式 (full 攒满 SCORE_BATCH 后一把评几十条也安全).
    #    旧实现按 GA_SCORE_BATCH 切批会把零头 (如 13→12+1) 拆成两次 round-trip 白烧 API.
    result_all = ScoreResult([], [], {})
    res = score_offspring(client, dedup, rnd, logger, debug_writer,
                          model_loggers=model_loggers)
    result_all.findings.extend(res.findings)
    result_all.gold_hits.extend(res.gold_hits)
    result_all.per_pair.update(res.per_pair)
    # 把全模型分回填到 individual.scores (= 它的 fitness)
    by_left = {f.S: f for f in res.findings}
    for ind in dedup:
        f = by_left.get(ind.s)
        pps = res.per_pair.get(ind.s)
        if pps and pps.get("scores"):
            ind.scores = list(pps["scores"])
            seen_s.add(ind.s)   # 全局记录: 已评
        elif f:
            ind.scores = [int(round(f.score))]  # 兜底
            seen_s.add(ind.s)

    # 5. crowding 替换入种群 (max_age 让超龄精英可被替).
    #    跳过已达 gold 的个体: 它们马上要被 evict, 先占位再腾位是白塞.
    replaced = 0
    for ind in dedup:
        if GA_EVICT_PROMOTED and ind.scores and max(ind.scores) >= GA_EVICT_SCORE_THRESH:
            continue   # 即将 evict, 不浪费 crowding 替换
        if pop.crowd_replace(ind, rng, max_age=GA_MAX_AGE):
            replaced += 1
    pop.refresh_after_score()

    # 5b. evict gold: 全模型 max>=SCORE_THRESH 的个体存档后移出种群
    n_evict = 0
    if GA_EVICT_PROMOTED:
        n_evict = pop.evict(lambda ind: bool(ind.scores)
                            and max(ind.scores) >= GA_EVICT_SCORE_THRESH)
        if n_evict:
            pop.refill(factory, rng)
    n_gold = len(result_all.gold_hits)
    logger(f"  [GA] crowding 替换 {replaced}/{len(dedup)}; evict gold {n_evict}; "
          f"新种群 {len(pop)} (diversity={pop.mean_diversity():.1f})\n")

    return GAEpochResult(
        findings=result_all.findings,
        gold_hits=result_all.gold_hits,
        per_pair=result_all.per_pair,
        gen_model="seed_ga",
        pop=[ind.to_dict() for ind in pop.to_list()],
        stats={"offspring": len(offspring), "scored": len(result_all.findings),
               "gold": n_gold, "evicted": n_evict, "fluency_killed": n_fluency_killed,
               "diversity": round(pop.mean_diversity(), 2)},
    )

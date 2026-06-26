# -*- coding: utf-8 -*-
"""GA 适应度: 直接用全模型 score_pairs 打分 (单层, 废弃 surrogate).

设计变更 (基于实测诊断): 旧版用 Flash 单模型 surrogate 驱动进化, 导致 GA 朝 Flash
偏好收敛 (promote→gold 转化率仅 40%, 进化方向被带偏). 现改为全模型直接打分:
  fitness = score_pairs 4 模型 cross-check 的均分 —— 真正的神鹅语审美信号.
代价是 LLM 成本回到 ~4x (代数减少), 但每代质量真实, 进化方向不被单模型污染.

流程: 全模型 score_pairs 打分一批 offspring → 返回 ScoreResult:
  findings    全模型评分结果 (落 scored.jsonl 用), score=均分
  gold_hits   max(scores)>=SCORE_THRESH (落 gold.jsonl 用)
  per_pair    {left: {scores, whys}}
个体 fitness = 它在 findings 里的 score (全模型均分).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import openai

from goose_digging.oracle import goose
from ..config import SCORE_THRESH
from ..finding import Finding
from ..llm import ModelLogger
from ..prompts import SEED_SCORE_SYSTEM
from ..scoring import score_pairs
from .genome import Individual


@dataclass
class ScoreResult:
    """一批 offspring 的全模型评分产物."""
    findings: list[Finding]             # 全模型评分结果 (落 scored.jsonl 用), score=均分
    gold_hits: list[dict]               # max(scores)>=SCORE_THRESH (落 gold.jsonl 用)
    per_pair: dict                      # {left: {scores, whys}}


def score_offspring(client: openai.OpenAI,
                    individuals: list[Individual],
                    rnd: int,
                    logger: Callable[[str], None],
                    debug_writer: Callable[[str], None] | None = None,
                    model_loggers: dict[str, ModelLogger] | None = None) -> ScoreResult:
    """全模型 score_pairs 直接打分一批 offspring. 返回 ScoreResult.

    fitness = 各个体在 findings 里的 score (全模型均分).
    走主路径 SCORE_MODELS + SEED_SCORE_SYSTEM, 保证 gold 标准 max(scores)>=SCORE_THRESH
    与枚举/旧路径一致. 单模型失败由 score_pairs 内部降级处理 (跳过该模型).

    容错: 全模型都失败时 score_pairs 返回 ([],[],{}), 本函数返回空 ScoreResult,
    进化继续不崩 (这批 offspring 无 fitness, 由 crowding 自然淘汰).
    """
    if not individuals:
        return ScoreResult([], [], {})
    pairs = [{"left": ind.s, "right": goose(ind.s), "dir": "seed_ga"}
             for ind in individuals]
    findings, gold_hits, per_pair = score_pairs(
        client, pairs, rnd, logger, debug_writer,
        model_loggers=model_loggers, system=SEED_SCORE_SYSTEM)
    return ScoreResult(findings, gold_hits, per_pair)

# -*- coding: utf-8 -*-
"""种群 + 多样性机制: fitness sharing (小生境) + crowding 替换.

强多样性 (按设计选定的"小生境+移民"策略):
  fitness sharing  让聚集的相似个体互相稀释分数 → 惩罚套路重复 (如一堆"…逼客")
  crowding 替换     新生 offspring 找种群中最近(Levenshtein)且更差的替换 → 子代填
                    相似生态位, 保种群分散
  锦标赛选父        小 k 锦标赛 + sharing 调整后的 fitness, 平衡开发/探索

Levenshtein 距离用于多样性度量 (变长中文串的合理距离). 实现 O(mn) DP, 串短 (<=9) 可承受.
"""
from __future__ import annotations

import random

from .genome import Individual


# ---------------------------------------------------------------------------
# Levenshtein 距离 (变长中文串多样性度量)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """编辑距离 (插入/删除/替换各 1). 串短 (<=9), O(mn) DP 足够."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# fitness sharing (小生境)
# ---------------------------------------------------------------------------

def _sharing(d: float, sigma: float) -> float:
    """sharing 函数: 距离 d 在邻域 sigma 内贡献 (越近贡献越大)."""
    if d >= sigma:
        return 0.0
    return 1.0 - d / sigma


def shared_fitness(ind: Individual, pop: list[Individual], sigma: float) -> float:
    """fitness sharing: raw / (1 + Σ sh(d)).

    raw = 该个体全模型均分 (ind.raw_score).
    分母把附近相似个体的密度计入 → 聚集的相似个体分数被稀释.
    自己到自己是 d=0, sh=1, 故分母至少含 1 (1+1=2 自稀释一半, 避免除零且鼓励独特点).
    """
    raw_score = ind.raw_score()
    niche = 0.0
    for other in pop:
        d = levenshtein(ind.s, other.s)
        niche += _sharing(d, sigma)
    if niche <= 0:
        return raw_score
    return raw_score / (1.0 + niche)


# ---------------------------------------------------------------------------
# 种群容器
# ---------------------------------------------------------------------------

class Population:
    """稳态种群: 固定大小, crowding 替换, sharing 调整选择概率."""

    def __init__(self, individuals: list[Individual], pop_size: int,
                 sigma: float, elite: int):
        # 去重 (按 s), 保留同 s 里分更高的
        self.pop: list[Individual] = self._dedupe(individuals)[:pop_size]
        self.pop_size = pop_size
        self.sigma = sigma
        self.elite = elite
        self._cur_epoch = 0   # 当前 epoch (cull_aged/crowd_replace 判超龄用)

    @staticmethod
    def _dedupe(inds: list[Individual]) -> list[Individual]:
        """按 s 去重, 同 s 取最高分 (raw_score)."""
        best: dict[str, Individual] = {}
        for ind in inds:
            cur = best.get(ind.s)
            if cur is None or ind.raw_score() > cur.raw_score():
                best[ind.s] = ind
        return list(best.values())

    def __len__(self) -> int:
        return len(self.pop)

    def to_list(self) -> list[Individual]:
        return list(self.pop)

    def best(self) -> Individual | None:
        return max(self.pop, key=lambda z: z.raw_score()) if self.pop else None

    def mean_diversity(self) -> float:
        """种群平均两两 Levenshtein (早熟监控用). 过低时 evolve 里触发"震荡"."""
        if len(self.pop) < 2:
            return 0.0
        n = len(self.pop)
        total = 0.0
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += levenshtein(self.pop[i].s, self.pop[j].s)
                cnt += 1
        return total / cnt if cnt else 0.0

    # ---- 选择 ----

    def tournament(self, k: int, rng: random.Random) -> Individual:
        """锦标赛选父: 随机抽 k 个, 取 sharing 调整后 fitness 最高的."""
        if not self.pop:
            raise ValueError("空种群无法选父")
        contenders = rng.sample(self.pop, min(k, len(self.pop)))
        return max(contenders, key=lambda z: shared_fitness(z, self.pop, self.sigma))

    # ---- 替换 (crowding) ----

    def crowd_replace(self, offspring: Individual, rng: random.Random,
                      max_age: int | None = None) -> bool:
        """crowding 替换: 找种群中和 offspring 最近(Levenshtein)且更差的个体替换.

        子代填补相似生态位 (而非总替最差), 保种群分散. 精英(GA_ELITE 个最高分)受保护
        不被替换, 但超龄(max_age)精英不再受保护 —— 否则高分个体永久存活主导种群.
        返回是否替换成功 (offspring 比 crowding 里最差的还差则不替换).
        """
        if len(self.pop) < self.pop_size:
            # 种群未满, 直接加入 (warm-start 阶段)
            self.pop.append(offspring)
            return True

        # 精英保护: 最高分的 elite 个不参与被替换. 但超龄精英解除保护
        # (高分个体活太久会主导进化 → 子代同化, 见 GA_MAX_AGE).
        ranked = sorted(range(len(self.pop)),
                        key=lambda i: self.pop[i].raw_score())
        if self.elite and max_age is not None:
            protect = {i for i in ranked[-self.elite:]
                       if self.pop[i].born + max_age > self._cur_epoch}
        elif self.elite:
            protect = set(ranked[-self.elite:])
        else:
            protect = set()

        # 在 offspring 的最近邻 (crowding) 里找最差的候选替换位
        cand_idx = [i for i in range(len(self.pop)) if i not in protect]
        if not cand_idx:
            return False
        cand_idx.sort(key=lambda i: levenshtein(self.pop[i].s, offspring.s))
        # crowding: 取最近的若干 (min(5, len)) 里最差的
        crowd = cand_idx[:min(5, len(cand_idx))]
        worst_in_crowd = min(crowd, key=lambda i: self.pop[i].raw_score())
        if offspring.raw_score() > self.pop[worst_in_crowd].raw_score():
            self.pop[worst_in_crowd] = offspring
            return True
        return False

    def refresh_after_score(self):
        """打分变更后去重 (新生可能撞已有 s, 保留高分). 不截断."""
        self.pop = self._dedupe(self.pop)[:self.pop_size]

    # ---- 防同化: aging + eviction ----

    def set_epoch(self, epoch: int):
        """记录当前 epoch (供 crowd_replace 判超龄精英用)."""
        self._cur_epoch = epoch

    def cull_aged(self, max_age: int) -> int:
        """淘汰超龄个体 (born + max_age <= cur_epoch). 返回淘汰数.

        防高分个体永久存活主导种群 (同化主因): 一个 score-9 个体若永不被换, 它的
        crossover 后代会淹没整个种群. 年龄上限强制让出位给新探索.
        """
        if not max_age or not self.pop:
            return 0
        before = len(self.pop)
        self.pop = [ind for ind in self.pop
                    if ind.born + max_age > self._cur_epoch]
        return before - len(self.pop)

    def evict(self, predicate) -> int:
        """按谓词移除个体 (predicate(ind)->bool 的移除). 返回移除数.

        典型: evict 已 promoted(存 gold)的个体 —— 既存档就不该再占种群当进化锚点.
        """
        before = len(self.pop)
        self.pop = [ind for ind in self.pop if not predicate(ind)]
        return before - len(self.pop)

    def refill(self, factory, rng: random.Random):
        """用 factory(rng)->Individual 把种群补回 pop_size. 供 cull/evict 后补新探索血."""
        while len(self.pop) < self.pop_size:
            ind = factory(rng)
            if ind is None:
                break
            # 避免补进重复
            if ind.s in {x.s for x in self.pop}:
                continue
            self.pop.append(ind)

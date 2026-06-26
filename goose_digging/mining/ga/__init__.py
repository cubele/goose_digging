# -*- coding: utf-8 -*-
"""神鹅语遗传算法 (GA) 子包.

稳态 GA + 小生境多样性, 仅用 LLM 打分 (不造句).

单层全模型打分 (废弃旧 surrogate 两层架构: 早期用单模型 Flash surrogate 驱动进化,
导致 GA 朝该单模型偏好收敛, 现改为全模型 score_pairs 直接打分, fitness=均分).

模块:
  genome     基因型 (候选中文串 S) + 算子 (crossover/mutation/immigrant)
  population 种群 + fitness sharing (Levenshtein) + crowding 替换 + aging/evict
  fitness    全模型 score_pairs 打分 (fitness = k 模型 cross-check 均分)
  evolve     稳态主循环 (warm_start → 进化 epoch → 返回 findings/gold_hits + 新种群)

采样策略: 近均匀 + 加性 boost (字/词映射积木, 低分也采保证覆盖).
详见各模块 docstring.
"""

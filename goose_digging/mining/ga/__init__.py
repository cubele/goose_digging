# -*- coding: utf-8 -*-
"""神鹅语遗传算法 (GA) 子包.

稳态 GA + 小生境多样性, 直接用全模型 LLM 打分 (不造句).

适应度 = 全模型 score_pairs 的 k 模型 cross-check 均分.

模块:
  genome     基因型 (候选中文串 S) + 算子 (crossover/mutation/immigrant)
  population 种群 + fitness sharing (Levenshtein) + crowding 替换 + aging/evict
  fitness    全模型 score_pairs 打分 (fitness = k 模型 cross-check 均分)
  evolve     稳态主循环 (warm_start → 进化 epoch → 返回 findings/gold_hits + 新种群)

采样策略: 近均匀 + 加性 boost (字/词映射积木, 低分也采保证覆盖).
详见各模块 docstring.
"""

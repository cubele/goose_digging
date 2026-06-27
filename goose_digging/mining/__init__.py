# -*- coding: utf-8 -*-
"""神鹅语挖掘管线包.

入口: ``python -m goose_digging.mining [bidict|full|seed]`` (见 __main__.py).
按职责分模块:
  config      路径/模型/开关/风格 prompt/挖掘阈值
  log         实时双写 Logger (stdout + run_*.log, 跑预筛/评分正文+进度心跳)
  jsonx       容错 JSON 抽取与解析
  llm         客户端 + 流式对话 + ModelLogger (各模型思维链独占 model_*.log)
  finding     Finding 数据结构 + 导出
  readability wordfreq zipf 字频判据 (字正不正常人认得)
  prompts     各阶段 user prompt
  wordlist    七源可信词表加载 (CustomPinyin/ci/idiom/山间新月/THUOCL/jieba/wordfreq)
  enumerate   字典双向枚举 (短词候选) + 双边词典子集
  seed        句子发掘 (scored.jsonl→字映射→LLM造长句)
  fluency     通顺预筛 (词典优先, 只对词典外的调 LLM)
  scoring     关系评分 (k模型并行 cross-check, 取均分, 任一模型>=gold另存 gold.jsonl)
  pipeline    挖掘主循环 (iterate枚举 + iterate_seed种子)
  __main__    CLI 入口 (bidict/full/seed 模式)
"""

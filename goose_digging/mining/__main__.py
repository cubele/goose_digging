# -*- coding: utf-8 -*-
"""神鹅语挖掘 CLI 入口.

用法:
  python -m goose_digging.mining            # 默认: 按顺序跑 bidict→full→seed
  python -m goose_digging.mining bidict     # 只跑双边词典阶段 (跳过预筛, 纯评分)
  python -m goose_digging.mining full       # 只跑全量枚举阶段 (预筛+评分)
  python -m goose_digging.mining seed       # 只跑 GA 神鹅语进化 (需先有 scored.jsonl 种子)

阶段说明 (固定顺序, state.phase 记录当前进度, 跨 run 续):
  bidict  双边词典 (S∈词典 且 goose(S)∈词典), 零噪声, 跳过预筛直接评分. 优先挖, 产 scored 种子.
  full    全量枚举 (单边词典), 通顺预筛后评分. 覆盖赶班/撬松这类右边不在词典的, 继续产 scored 种子.
  seed    GA 神鹅语进化: scored.jsonl 高分 pair→字/词映射积木→稳态 GA 进化(全模型打分)→gold.
          从零开始: 先跑 bidict/full 产种子, 再 seed. scored.jsonl 空时 seed 会跳过.

LLM 配置: 复制 llm_config.toml.example → llm_config.toml 填自己的 base_url/api_key/模型.
          没配置则用 config.py 占位默认.

日志布局 (多模型并行打分):
  mined/logs/run_<ts>.log            主 term 全量日志 (启动信息 + 预筛/评分进度心跳 + 结果).
                                不再含任何模型思维链 —— 只起监视作用.
  mined/logs/model_<ts>_<model>.log  每个评分模型独占的思维链日志 (跨 round 追加).
                                tail -f 单个文件即可实时盯某模型的推理.

实时看进度: tail -f mined/logs/run_*.log
实时盯某模型思维链: tail -f mined/logs/model_*_<模型名>.log
"""
from __future__ import annotations

import sys
from datetime import datetime

from .config import OUT_DIR, LOG_DIR, SCORE_MODELS, GA_OFFSPRING_PER_EPOCH
from .log import Logger
from .llm import make_client, make_debug_writer, make_model_logger
from .llm_config import has_user_config
from .finding import export_findings
from .state import MiningState, STATE_PATH
from . import pipeline
from . import enumerate as enum_mod

# 阶段顺序 (固定)
PHASES = ["bidict", "full", "seed"]


def _run_bidict(client, state, log, dw, all_findings, model_loggers):
    """双边词典阶段: 跳过预筛直接评分. 跑到 cursor 耗尽."""
    log.log(f"# 双边词典候选中...")
    all_pairs = enum_mod.enumerate_both_in_dict()
    log.log(f"# {len(all_pairs)} 双边词典候选  cur={state.cursor_bidict}")
    while state.cursor_bidict < len(all_pairs):
        fs = pipeline.iterate(client, state, all_pairs, log, True, dw,
                              phase="bidict", model_loggers=model_loggers,
                              out_dir=OUT_DIR, state_path=STATE_PATH)
        all_findings.extend(fs)
    log.log(f"# bidict 完成 (cur={state.cursor_bidict})")


def _run_full(client, state, log, dw, all_findings, model_loggers):
    """全量枚举阶段: 预筛+评分. 跑到 cursor 耗尽."""
    log.log(f"# 全量枚举候选中 (需预筛)...")
    all_pairs = enum_mod.enumerate_pairs()
    log.log(f"# {len(all_pairs)} 全量候选  cur={state.cursor_full}")
    while state.cursor_full < len(all_pairs):
        fs = pipeline.iterate(client, state, all_pairs, log, False, dw,
                              phase="full", model_loggers=model_loggers,
                              out_dir=OUT_DIR, state_path=STATE_PATH)
        all_findings.extend(fs)
    log.log(f"# full 完成 (cur={state.cursor_full})")


def _run_seed(client, state, log, dw, all_findings, model_loggers):
    """GA 神鹅语进化阶段: 无尽循环, Ctrl-C 停. 需 scored.jsonl 有种子 (先跑 bidict/full)."""
    log.log(f"# GA 神鹅语进化 (Ctrl-C 停)")
    while True:
        fs = pipeline.iterate_seed(client, state, log, GA_OFFSPRING_PER_EPOCH, dw,
                                   model_loggers=model_loggers,
                                   out_dir=OUT_DIR, state_path=STATE_PATH)
        all_findings.extend(fs)



def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    client = make_client()
    state = MiningState.load()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{ts}.log"
    log = Logger(log_path)
    dw = make_debug_writer()

    # 提示 LLM 配置来源 (TOML 用户配置 vs config.py 默认占位)
    if has_user_config():
        log.log(f"# LLM 配置: llm_config.toml (用户配置)")
    else:
        log.log(f"# LLM 配置: config.py 默认占位 (复制 llm_config.toml.example → "
                f"llm_config.toml 配置自己的模型)")

    # 一个 run 一组 ModelLogger (按 model 名缓存, 一个文件跨 round 追加).
    # 思维链只进这些文件, 不进主 term; 主 term 只看多模型进度心跳.
    model_loggers: dict = {}
    for mc in SCORE_MODELS:
        model_loggers[mc["model"]] = make_model_logger(mc["model"], run_ts=ts)
    log.log(f"# 神鹅语挖掘  mode={mode or '(默认顺序)'}  ts={ts}  phase={state.phase}  "
            f"iter={state.n_iter}  gold={len(state.gold_pairs)}  "
            f"cur_bidict={state.cursor_bidict}  cur_full={state.cursor_full}")
    from .wordlist import dictionary_stats
    log.log(f"# 锚集: {dictionary_stats()}")
    log.log(f"# 模型思维链日志 (各模型独占, tail -f 盯单模型):")
    for mname, ml in model_loggers.items():
        log.log(f"#   {mname:<28} -> {ml.path}")

    all_findings: list = []
    try:
        if mode in PHASES:
            # 单阶段模式
            state.phase = mode
            state.save()
            if mode == "bidict":
                _run_bidict(client, state, log, dw, all_findings, model_loggers)
            elif mode == "full":
                _run_full(client, state, log, dw, all_findings, model_loggers)
            elif mode == "seed":
                _run_seed(client, state, log, dw, all_findings, model_loggers)
        else:
            # 默认: 按顺序 bidict→full→seed, 从 state.phase 续跑
            start_idx = PHASES.index(state.phase) if state.phase in PHASES else 0
            for phase in PHASES[start_idx:]:
                state.phase = phase
                state.save()
                log.log(f"\n{'#'*72}\n# 阶段: {phase}\n{'#'*72}")
                if phase == "bidict":
                    _run_bidict(client, state, log, dw, all_findings, model_loggers)
                elif phase == "full":
                    _run_full(client, state, log, dw, all_findings, model_loggers)
                elif phase == "seed":
                    _run_seed(client, state, log, dw, all_findings, model_loggers)
            state.phase = "done"
            state.save()
            log.log(f"# 所有阶段完成")
    finally:
        # 无论正常结束还是 Ctrl-C, 都把模型日志文件关掉 (flush + release fd)
        for ml in model_loggers.values():
            ml.close()

    log.log(f"\n本 run 挖掘 {len(all_findings)} 条, 累计 gold {len(state.gold_pairs)} 条")
    if state.gold_pairs:
        path = export_findings(state.gold_pairs, "GOLD")
        log.log(f"gold 库导出 -> {path}")
        for x in sorted(state.gold_pairs, key=lambda z: -z.score):
            log.log(f"  [{x.score:.1f}] {x.S} → {x.real_goose_S}  {x.why}")
    else:
        log.log("gold 库空, 无命中.")
    log.close()


if __name__ == "__main__":
    main()

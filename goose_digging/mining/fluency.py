# -*- coding: utf-8 -*-
"""T1 通顺预筛: 判 pair 的 left/right 讲得通不通, 砍明显乱码.

通顺判据宽松 (三选一即通顺):
  (a) 在词典里 / 真实词      —— 词典命中走字典短路 (零 LLM)
  (b) 是一个解释得通的词语    —— 谐音/口语/俚语/略不规范但讲得通 (赶班/撬松)
  (c) 可以作为句子的一部分    —— 能塞进一句话里说得通
只有两边都讲得通的 pair 才进 T2 打分 (存储所有打分 pair).
(b)/(c) 的判定只对词典外的串送 Flash (FLUENCY_SYSTEM 三选一判据).
"""
from __future__ import annotations

from typing import Callable

import openai

from .config import FLUENCY_MODEL, FLUENCY_THINKING, FLUENCY_RETRY_MAX
from .prompts import fluency_prompt, FLUENCY_SYSTEM
from .llm import llm_chat
from .jsonx import parse_json_list
from .wordlist import load_static_dictionary


def _judge_unknown(client: openai.OpenAI, strs: list[str],
                   logger: Callable[[str], None] | None,
                   debug_writer: Callable[[str], None] | None,
                   system: str = None) -> dict[str, bool]:
    """调 Flash 判一批短语讲得通不通, 返回 {text: ok}. 漏回的短语不进返回值.

    system: 可覆盖 (GA 用宽松判 GA_FLUENCY_SYSTEM). 默认 FLUENCY_SYSTEM.
    判据: 三选一即通顺 (词典/解释得通/可成句). 只杀明显乱码/字硬拼.
    """
    if system is None:
        from .prompts import FLUENCY_SYSTEM as _DEF
        system = _DEF
    raw = llm_chat(client, FLUENCY_MODEL, system,
                   fluency_prompt(strs),
                   label="T1通顺", thinking=FLUENCY_THINKING,
                   logger=logger, debug_writer=debug_writer)
    out: dict[str, bool] = {}
    for item in parse_json_list(raw, label="T1-fluency"):
        if isinstance(item, dict):
            t = str(item.get("text", "")).strip()
            ok = item.get("ok")
            if t:
                out[t] = bool(ok)
    return out


def fluency_filter(client: openai.OpenAI, pairs: list[dict],
                   logger: Callable[[str], None],
                   debug_writer: Callable[[str], None] | None,
                   system: str | None = None) -> list[dict]:
    """T1: 判每个 pair 的 left+right 讲得通不通, 只保留两边都讲得通的.

    通顺判据宽松 (三选一): 词典里/真实词 (字典短路零LLM), 或解释得通的词语 (谐音/口语),
    或可作为句子的一部分. 盲区词 (赶班/撬松 这类不在词典但讲得通的) 由 Flash LLM 判.
    只有两边都讲得通的 pair 才进 T2 打分 (存所有打分 pair).

    词典优先: 在词典的直接判通顺, 只把词典外的送 Flash.

    system: 可覆盖通顺判 prompt. 枚举路径默认 FLUENCY_SYSTEM (三选一, 放行谐音/口语/可成句);
      GA 用 GA_FLUENCY_SYSTEM (更宽松: 能解释得通产生联想即可) —— GA 长句候选本是字映射
      拼出的谐音/联想串, 判据最宽, 只杀纯乱码/字硬拼/叠字堆砌.

    Flash 漏回的短语 (模型少回/输出截断) 不判死刑 —— 立即小批量重判,
    重试到全部判过或达 FLUENCY_RETRY_MAX. 重试完仍漏的才按不通处理 (并显式报警),
    避免把可能的金子误杀.
    """
    # 收集去重短语
    all_strs: set[str] = set()
    for p in pairs:
        all_strs.add(p["left"])
        all_strs.add(p["right"])
    # 词典优先: 在词典的直接判 true
    static = load_static_dictionary()
    judged: dict[str, bool] = {}
    unknown: list[str] = []
    for s in all_strs:
        if s in static:
            judged[s] = True
        else:
            unknown.append(s)
    # 对词典外的调 Flash, 漏回的重试补判
    if unknown:
        pending = list(unknown)  # 还没判成功的短语
        for attempt in range(FLUENCY_RETRY_MAX + 1):  # 首判 + N 次重试
            if not pending:
                break
            got = _judge_unknown(client, pending, logger, debug_writer, system=system)
            judged.update(got)
            still_missing = [s for s in pending if s not in got]
            if not still_missing:
                break
            if attempt < FLUENCY_RETRY_MAX and logger:
                logger(f"\n  [T1] ⟳ 通顺判漏回 {len(still_missing)}/{len(pending)}, "
                       f"重试第{attempt+1}/{FLUENCY_RETRY_MAX} 次\n")
            pending = still_missing
        # 重试完仍漏的: 才按不通处理, 显式报警 (不再静默误杀)
        final_missing = [s for s in unknown if s not in judged]
        if final_missing and logger:
            if len(judged) - (len(all_strs) - len(unknown)) == 0:
                logger(f"\n  [T1] ⚠️ 通顺判重试 {FLUENCY_RETRY_MAX} 次仍一条都没回 "
                       f"({len(unknown)}个送判) —— 输出可能持续被截断/格式错, 本批 T1 不可信\n")
            else:
                sample = ",".join(final_missing[:5])
                more = f" 等{len(final_missing)}个" if len(final_missing) > 5 else ""
                logger(f"\n  [T1] ⚠️ 通顺判重试后仍漏 {len(final_missing)} 个 "
                       f"({sample}{more}), 这些按不通处理 (相关 pair 被砍)\n")
    # 过滤: 两边都通才存活
    survivors = [p for p in pairs
                 if judged.get(p["left"], False) and judged.get(p["right"], False)]
    if logger:
        hit = len(all_strs) - len(unknown)
        logger(f"\n  [T1] {len(all_strs)}短语: 词典命中{hit}, 送LLM{len(unknown)}, "
               f"两边通存活{len(survivors)}\n")
    return survivors

# -*- coding: utf-8 -*-
"""DeepSeek 客户端 + 流式对话调用 + 结构化 debug 日志.

关键约束: 部分 LLM (如 DeepSeek-R/V 系列, GLM, Qwen-Max) 是 reasoning 模型.
  - 不能设 max_tokens: 设了会把 reasoning 耗光导致 content 为空 (finish=length).
  - JSON 输出用 extra_body={"response_format":{"type":"json_object"}},
    但模型常返回数组, 解析端容错 (见 jsonx).
  - 流式事件 delta 里同时有 content (正文) 和 reasoning_content (思维链).
    思维链往往是 token 大头 (一次调用 reasoning 可达上万 token).

落盘策略:
  - 思维链只写到该模型独占的 ModelLogger 文件 (mined/logs/model_<ts>_<model>.log),
    不混进主 term. 主 term 只看多模型进度心跳, 起监视作用.
  - debug 单文件 (mined/logs/llm_debug_<ts>.log) 由调用方注入 writer, 兜底用,
    每次调用一段 (含 prompt/reasoning/content 预览), 供离线分析.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import openai

from .config import API_BASE_URL, LOG_DIR, LLM_RETRIES, LLM_RETRY_BACKOFF


def _load_key() -> str:
    """api_key 优先级: llm_config.toml [security] > 环境变量 GOOSE_API_KEY."""
    from .llm_config import load_user_config, resolve_api_key
    return resolve_api_key(load_user_config())


def make_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=_load_key(), base_url=API_BASE_URL)


def llm_chat(client: openai.OpenAI, model: str, system: str, user: str,
             retries: int = LLM_RETRIES, label: str = "",
             thinking: bool = False,
             temperature: Optional[float] = None,
             logger: Optional[Callable[[str], None]] = None,
             debug_writer: Optional[Callable[[str], None]] = None) -> str:
    """流式对话调用. 不设 max_tokens.

    label: 阶段名, 进度/debug 用. 返回完整 content (strip). 失败重试.
    thinking: 是否开 reasoning. 简单任务关掉省 80x 时间+token; 需要联想/评判的才开.
    temperature: LLM 采样温度. None=API 默认. 造句可升高提升创造力.
    logger: 流式 channel (一般传 Logger.stream). reasoning/content 实时喂给它,
            落盘 run_*.log. 若为 None, 静默 (并行预筛用).
    debug_writer: 兜底 debug 文件 writer (每段含 token/预览); None 则不写.
    """
    tag = f"[{label}] " if label else ""
    if logger is None:
        logger = lambda s: None  # 静默 (并行预筛传 None 吞日志; 勿退化 stdout, 否则多路交错乱码)

    extra = {"response_format": {"type": "json_object"}}
    if not thinking:
        extra["thinking"] = {"type": "disabled"}  # 关思考: 简单任务 80x 加速
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            create_kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                extra_body=extra,
                stream=True,
                stream_options={"include_usage": True},
            )
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            stream = client.chat.completions.create(**create_kwargs)
            content_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            usage = None
            finish_reason = None
            in_reasoning = False
            in_content = False
            for ev in stream:
                if ev.choices:
                    ch = ev.choices[0]
                    d = ch.delta
                    rc = getattr(d, "reasoning_content", None)
                    cc = getattr(d, "content", None)
                    if rc:
                        reasoning_chunks.append(rc)
                        if not in_reasoning:
                            logger(f"\n  {tag}🧠 思维链:\n")
                            in_reasoning = True; in_content = False
                        logger(rc)
                    if cc:
                        content_chunks.append(cc)
                        if in_reasoning or not in_content:
                            logger(f"\n  {tag}📝 正文:\n")
                            in_content = True; in_reasoning = False
                        logger(cc)
                    if getattr(ch, "finish_reason", None):
                        finish_reason = ch.finish_reason
                if getattr(ev, "usage", None):
                    usage = ev.usage
            content = "".join(content_chunks).strip()
            reasoning = "".join(reasoning_chunks)
            dt = time.time() - t0

            # ---- token 明细 ----
            pt = ct = rt = "?"
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", "?")
                ct = getattr(usage, "completion_tokens", "?")
                cd = getattr(usage, "completion_tokens_details", None)
                rt = getattr(cd, "reasoning_tokens", "?") if cd else 0
            # finish=length = 服务端 output token 上限截断了, 输出多半是半截 JSON
            if finish_reason == "length":
                logger(f"\n  {tag}⚠️ finish=length 截断! 输出被服务端 max_output_tokens 砍了, "
                       f"JSON 多半没闭合, 这批评分作废\n")
            logger(f"\n  {tag}✅ {dt:.1f}s  tok(prompt={pt} comp={ct})  "
                   f"正文{len(content)}字 思考{len(reasoning)}字  finish={finish_reason}\n")

            # ---- 兜底 debug (完整 prompt/reasoning/content 预览) ----
            if debug_writer is not None:
                debug_writer(
                    f"\n{'='*70}\n[{label}] attempt={attempt+1} {dt:.1f}s  "
                    f"thinking={thinking} finish={finish_reason}\n"
                    f"tokens: prompt={pt} completion={ct} reasoning={rt}\n"
                    f"--- prompt (user, 前 400 字) ---\n{user[:400]}\n"
                    f"--- reasoning (前 800 字) ---\n{reasoning[:800]}\n"
                    f"--- content (前 400 字) ---\n{content[:400]}"
                )
            return content
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            logger(f"\n  {tag}失败 ({dt:.1f}s): {e}  重试 {attempt+1}/{retries}\n")
            if debug_writer is not None:
                debug_writer(f"\n[{label}] 失败 attempt={attempt+1} {dt:.1f}s: {e}")
            last = e
            time.sleep(LLM_RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败: {last}")


def make_debug_writer(path: Optional[Path] = None) -> Callable[[str], None]:
    """构造一个兜底 debug writer: 追加写指定文件 (默认 mined/llm_debug_<ts>.log)."""
    if path is None:
        path = LOG_DIR / f"llm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    f = path.open("a", encoding="utf-8")

    def _w(record: str):
        f.write(record + "\n")
        f.flush()

    return _w


class ModelLogger:
    """单模型独占的思维链日志: 只落盘该模型的 run 文件, 不污染主 term.

    多模型并行打分时, 每个模型拿到自己的 ModelLogger, 思维链实时写到
    mined/model_<ts>_<model>.log; 主 term 只通过进度心跳汇报, 不再被
    流式输出淹没. 一个 run 一个文件 (跨 round 追加), 用 tail -f 实时盯某模型.

    self.write 是签名 (str)->None 的 callable, 可直接当作 llm_chat 的 logger 传.
    """

    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("a", encoding="utf-8")
        self.bytes_written = 0
        self._closed = False

    def write(self, chunk: str):
        """流式写一小段 (无换行追加). llm_chat 的 logger 走这个."""
        if not chunk or self._closed:
            return
        data = chunk.encode("utf-8", errors="replace")
        self._f.write(chunk)
        self._f.flush()
        self.bytes_written += len(data)

    def section(self, header: str):
        """写一段分隔的 header (思维链不混进同文件的别的段落)."""
        if self._closed:
            return
        self._f.write(header + "\n")
        self._f.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._f.close()
        except Exception:
            pass


def make_model_logger(model: str, tag: str = "",
                      run_ts: Optional[str] = None) -> ModelLogger:
    """构造一个模型独占日志: mined/logs/model_<run_ts>_<safe_model>.log.

    一个 run (同 run_ts) 一个文件, 跨 round 追加, tail -f 友好.
    safe_model: 把 model 名里的非字母数字/中文换成 _ 当文件名.
    """
    if run_ts is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    import re
    safe = re.sub(r"[^\w\u4e00-\u9fff.]+", "_", model).strip("_") or "model"
    suffix = f"_{tag}" if tag else ""
    path = LOG_DIR / f"model_{run_ts}_{safe}{suffix}.log"
    return ModelLogger(path)

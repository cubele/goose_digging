# -*- coding: utf-8 -*-
"""实时双写 Logger: 同时写 stdout 和 log file, 每行立即 flush, 供 tail -f.

T1/T2 的正文 + 进度心跳经这里落 run_*.log.
T2 多模型评分的思维链不走本 Logger, 而是各写各的 mined/model_<ts>_<model>.log
(见 llm.ModelLogger), 避免多模型并行时互相覆盖/淹没主日志.
"""
from __future__ import annotations

import sys
from pathlib import Path


class Logger:
    """双写日志: stdout + 文件. 每行立即 flush, 供 tail -f 实时查看进度."""

    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("w", encoding="utf-8")

    def log(self, msg: str = ""):
        """双写一行. 空 msg 也写空行."""
        line = str(msg)
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def stream(self, chunk: str):
        """流式写一小段 (无换行). LLM reasoning/content 用这个, 完整进 log."""
        if not chunk:
            return
        sys.stdout.write(chunk)
        sys.stdout.flush()
        self._f.write(chunk)
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

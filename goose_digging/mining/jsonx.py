# -*- coding: utf-8 -*-
"""容错 JSON 抽取与解析.

DeepSeek 常在 JSON 外加 markdown 代码块, 或返回数组而非对象 (尽管要求 json_object).
这里统一容错: 抠出最外层 {...}/[...], 处理字符串内的括号, 兼容两种契约.
"""
from __future__ import annotations

import json
import re
import sys

_JSON_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.IGNORECASE | re.MULTILINE)


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _JSON_FENCE_RE.sub("", text).strip()
    return text


def find_json(text: str) -> str | None:
    """容错: 从 LLM 输出里抠出最外层 JSON (对象或数组), 处理字符串内的括号."""
    text = _strip_fence(text)
    start = -1
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start < 0:
        return None
    close_ch = "]" if text[start] == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == text[start]:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_list(data, _depth: int = 0) -> list:
    """从解析结果里取 list: 直接是 list, 或 dict 里第一个 list 值 (如 results/items).

    容错: 模型偶尔把答案塞进字符串字段 (如 {"output":"<json>"} /
    {"reasoning":"...","content":"<json>"}). 此时顶层无 list, 递归一层: 对每个
    字符串型 value 尝试 json.loads, 解出来再取 list. 只递归一层 (depth) 防膨胀.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        # 顶层无 list: 尝试解字符串字段里嵌套的 JSON (只一层)
        if _depth == 0:
            for v in data.values():
                if isinstance(v, str):
                    blob = find_json(v)
                    if blob:
                        try:
                            inner = json.loads(blob)
                        except json.JSONDecodeError:
                            continue
                        got = extract_list(inner, _depth=1)
                        if got:
                            return got
    return []


def parse_json_list(raw: str, label: str = "") -> list:
    """解析 LLM 输出为 list. 兼容 [..] 和 {"results":[..]} 两种契约.

    label: 调用阶段名 (如 "评分"), 解析失败时用于 stderr 警告定位.
    失败仍返回 [] (不抛异常, 不炸调用方), 但会打可见警告, 避免 batch 全丢却无感知.
    """
    blob = find_json(raw)
    if not blob:
        tag = f"[{label}] " if label else ""
        print(f"⚠️ {tag}parse_json_list: 抽不出 JSON (可能正文空/token耗光), "
              f"raw前200字: {raw[:200]!r}", file=sys.stderr)
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        tag = f"[{label}] " if label else ""
        print(f"⚠️ {tag}parse_json_list: JSON 解析失败 {e}, "
              f"raw前200字: {raw[:200]!r}", file=sys.stderr)
        return []
    return extract_list(data)

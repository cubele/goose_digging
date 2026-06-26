# -*- coding: utf-8 -*-
"""用户 LLM 配置: 读 llm_config.toml, 覆盖 config.py 默认值.

开源友好: 用户复制 llm_config.toml.example → llm_config.toml, 填 base_url/api_key/
各阶段模型即可, 不动代码. 没有配置文件时走 config.py 占位默认 (用户需自行改对).

配置优先级 (api_key): TOML [security] api_key > 环境变量 GOOSE_API_KEY.
(明文 API_KEY 文件回退已移除: 易误提交泄露密钥, 不再支持.)

TOML 解析: 自写微型解析器 (纯标准库, 支持 3.8+), 不引入 tomli/toml 依赖.
只支持本项目用的结构: 顶层/表内的 key="str"/bool/int, 以及 [[array]] 重复表.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJ_DIR / "llm_config.toml"


# ---------------------------------------------------------------------------
# 微型 TOML 解析器 (本项目用到的子集: 字符串/布尔/整数, 单表 [x] 和数组表 [[x]])
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """去行尾注释 (不处理字符串内的 #)."""
    in_str = False
    quote = ""
    for i, c in enumerate(line):
        if in_str:
            if c == quote:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                quote = c
            elif c == "#":
                return line[:i]
    return line


def _parse_scalar(raw: str):
    """解析标量值: 'str' / \"str\" / true/false / int."""
    raw = raw.strip()
    if not raw:
        return None
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        return raw   # 兜底当字符串


def parse_toml(text: str) -> dict:
    """解析本项目用的 TOML 子集 → 嵌套 dict.

    支持:
      key = "value" / key = true / key = 123
      [section]         → 嵌套表
      [[section]]       → 数组表 (重复出现则 append)
    """
    result: dict = {}
    cur = result          # 当前写入的表
    cur_path: list[str] = []

    for line in text.splitlines():
        line = _strip_comment(line).strip()
        if not line:
            continue
        # 数组表 [[section]] (可能 [[a.b]])
        if line.startswith("[[") and line.endswith("]]"):
            parts = line[2:-2].strip().split(".")
            obj = result
            for p in parts[:-1]:
                obj = obj.setdefault(p.strip(), {})
            key = parts[-1].strip()
            arr = obj.setdefault(key, [])
            cur = {}
            arr.append(cur)
            cur_path = parts
            continue
        # 单表 [section]
        if line.startswith("[") and line.endswith("]"):
            parts = line[1:-1].strip().split(".")
            obj = result
            for p in parts:
                p = p.strip()
                obj = obj.setdefault(p, {})
            cur = obj
            cur_path = parts
            continue
        # key = value
        if "=" in line:
            k, v = line.split("=", 1)
            cur[k.strip()] = _parse_scalar(v)
    return result


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------

def load_user_config() -> dict:
    """读 llm_config.toml. 不存在返回 {} (用 config.py 占位默认)."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return parse_toml(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  配置解析失败不崩, 退回默认
        return {}


def resolve_api_key(cfg: dict) -> str:
    """api_key 优先级: TOML [security] api_key > GOOSE_API_KEY env. 无则返空串."""
    sec = cfg.get("security", {}) if isinstance(cfg, dict) else {}
    if sec.get("api_key") and sec["api_key"] != "sk-...":
        return str(sec["api_key"])
    env = os.environ.get("GOOSE_API_KEY", "").strip()
    if env:
        return env
    return ""


def resolve_base_url(cfg: dict) -> str:
    sec = cfg.get("security", {}) if isinstance(cfg, dict) else {}
    return str(sec.get("base_url", "")).rstrip("/")


def models_section(cfg: dict, name: str) -> list[dict]:
    """取 [[name]] 模型数组, 每项 {model, thinking}. 缺失返 []."""
    arr = cfg.get(name, [])
    if not isinstance(arr, list):
        return []
    out = []
    for item in arr:
        if isinstance(item, dict) and item.get("model"):
            out.append({"model": str(item["model"]),
                        "thinking": bool(item.get("thinking", False))})
    return out


def fluency_model(cfg: dict) -> dict | None:
    """取 [fluency] 单模型. 缺失返 None."""
    f = cfg.get("fluency")
    if isinstance(f, dict) and f.get("model"):
        return {"model": str(f["model"]),
                "thinking": bool(f.get("thinking", False))}
    return None


def has_user_config() -> bool:
    """是否有 llm_config.toml (用于 __main__ 启动提示)."""
    return CONFIG_PATH.exists()

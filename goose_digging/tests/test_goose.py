# -*- coding: utf-8 -*-
"""
goose 完整测试套件 (打表版).
覆盖: 全量自洽, 历史4组, 新oracle3组, 三字节字, ungoose逆向, goose_block错位.
"""
import pytest
from goose_digging import goose, goose_block, goose_char, ungoose, build_inverse
import json
import os


_ORACLE_PATH = os.path.join(os.path.dirname(__file__), "..", "oracle", "oracle.json")


# ---------------------------------------------------------------------------
# 1. 全量 oracle 自洽 (21391 字符)
# ---------------------------------------------------------------------------
def test_full_table_consistency():
    """oracle 表中每个映射, goose_char 必须复现."""
    oracle = json.load(open(_ORACLE_PATH, encoding="utf-8"))
    ok = sum(1 for s, t in oracle.items() if goose_char(s) == t)
    assert ok == len(oracle), f"{ok}/{len(oracle)}"


# ---------------------------------------------------------------------------
# 2. 历史4组 (题目最初给出)
# ---------------------------------------------------------------------------
HISTORY_4 = [
    ("以一星期为一期", "笆办辣袋百办袋"),
    ("每期选出一首当", "每袋联叫办俭崮"),
    ("课题曲最高分数", "草玛妒呵光尸谒"),
    ("满尚未达到正分", "捺景踏茫毗赖尸"),
]


@pytest.mark.parametrize("src,expected", HISTORY_4)
def test_history_4(src, expected):
    assert goose(src) == expected


# ---------------------------------------------------------------------------
# 3. 新 oracle 3组 (逐字符模型)
# ---------------------------------------------------------------------------
def test_new_oracle_3_chain():
    assert goose("锚锚锚猫锚锚锚锚锚锚锚锚锚锚锚锚锚锚锚锚锚锚锚") == \
           "膳膳膳猫膳膳膳膳膳膳膳膳膳膳膳膳膳膳膳膳膳膳膳"


def test_new_oracle_3_mixed():
    assert goose("中国：abc猫xyz，你好！这是什么？time。") == \
           "面寓：abc猫xyz，你攻！晴困胶么？time。"


# ---------------------------------------------------------------------------
# 4. 三字节字: goose vs goose_block 差异
# ---------------------------------------------------------------------------
def test_three_byte_passthrough():
    assert goose_char("猫") == "猫"
    assert goose_char("你") == "你"


def test_block_misalign():
    assert goose("锚锚锚猫锚锚锚")[3] == "猫"
    assert goose_block("锚锚锚猫锚锚锚")[3] != "猫"


# ---------------------------------------------------------------------------
# 5. ASCII / 标点穿过
# ---------------------------------------------------------------------------
def test_passthrough():
    assert goose("abc123") == "abc123"
    assert goose("Python3.10") == "Python3.10"
    assert goose("：，。（）") == "：，。（）"


# ---------------------------------------------------------------------------
# 6. ungoose 逆向 (反查表, 简体优先)
# ---------------------------------------------------------------------------
UNGOOSE_CASES = [
    ("笆办辣袋百办袋", "以一星期为一期"),
    ("草玛妒呵光尸谒", "课题曲最高分数"),
    ("捺景踏茫毗赖尸", "满尚未达到正分"),
]


@pytest.mark.parametrize("mojibake,original", UNGOOSE_CASES)
def test_ungoose(mojibake, original):
    assert ungoose(mojibake) == original


def test_roundtrip():
    for s in ["以一星期为一期", "课题曲最高分数", "中国abc", "为里个干么"]:
        assert ungoose(goose(s)) == s, f"roundtrip 失败: {s!r}"


def test_ungoose_multi_source():
    """百 <- 为(简, 非不动点), ungoose 取 为."""
    assert goose("为") == "百"
    assert ungoose("百") == "为"


# ---------------------------------------------------------------------------
# 7. 反查表性质
# ---------------------------------------------------------------------------
def test_inverse_nonfixed_priority():
    """多对一目标首个返回必须是非不动点(若有)."""
    inv = build_inverse()
    multi = {g: s for g, s in inv.items() if len(s) > 1}
    for g, srcs in multi.items():
        nonfixed = [s for s in srcs if goose_char(s) != s]
        if nonfixed:
            assert goose_char(srcs[0]) != srcs[0], (
                f"目标 {g!r}: 应优先非不动点, 但返回了不动点 {srcs[0]!r}"
            )

# -*- coding: utf-8 -*-
"""mining 纯单测: prompt 格式 / 采样数学 / 惰性枚举 / state 持久化.

不调 LLM, 全部确定性逻辑. 集成测试 (三阶段串接/GA 端到端/断点续传) 见 test_e2e.py.
文件 IO 通过传参 (out_dir) 隔离到 tmp_path, 绝不碰真实 mined/.
"""
import json

import pytest

from goose_digging.mining import config, state, finding, pipeline
from goose_digging.mining import prompts


# ---------------------------------------------------------------------------
# fixture: 造假 scored.jsonl + 临时 state_path, 全部传参 (不改全局)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_out(tmp_path):
    """临时输出目录, 造假 scored.jsonl 种子. 测试把 out_dir 传给被测函数."""
    out = tmp_path / "mined"
    out.mkdir()
    sp = out / "scored.jsonl"
    with sp.open("w", encoding="utf-8") as f:
        for L, R in [("义父", "盗摄"), ("挺住", "尿住"), ("通心", "看奶"),
                     ("鹿鹿", "集集"), ("粪厂", "实境")]:
            f.write(json.dumps({"left": L, "right": R, "dir": "fwd",
                                "score": 8.0, "scores": [8], "whys": ["x"]}) + "\n")
    return out


# ---------------------------------------------------------------------------
# 1. prompt 生成
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_fluency_prompt_format(self):
        p = prompts.fluency_prompt(["母鸡", "熟鸡"])
        assert "2 个" in p and "母鸡" in p and "熟鸡" in p
        assert '"items"' in p and '"ok"' in p

    def test_fluency_system_has_three_criteria(self):
        """新通顺判据 (三选一): 词典/解释得通/可成句 任一即通顺."""
        s = prompts.FLUENCY_SYSTEM
        # 三选一关键词出现
        assert "真实词" in s or "词典" in s
        assert "解释得通" in s
        assert "句子的一部分" in s or "成句" in s or "塞进一句" in s

    def test_ga_fluency_system_is_lenient(self):
        """GA 通顺判更宽松: 能解释得通/产生联想即通顺 (长句尤其宽松)."""
        s = prompts.GA_FLUENCY_SYSTEM
        assert "联想" in s
        assert "长句" in s
        # 比 FLUENCY_SYSTEM 更宽 (含长句措辞)
        assert s != prompts.FLUENCY_SYSTEM

    def test_score_prompt_format(self):
        p = prompts.score_prompt([{"left": "义父", "right": "盗摄"}])
        assert "1 条" in p and "义父 → 盗摄" in p
        assert '"scores"' in p and '"why"' in p

    def test_all_system_prompts_exist(self):
        for name in ["FLUENCY_SYSTEM", "GA_FLUENCY_SYSTEM", "SCORE_SYSTEM", "SEED_SCORE_SYSTEM"]:
            assert hasattr(prompts, name) and len(getattr(prompts, name)) > 10

# ---------------------------------------------------------------------------
# 2. 采样/温度/shuffle/去重
# ---------------------------------------------------------------------------

class TestSampling:
    def test_boost_sample_no_duplicates(self):
        """近均匀 boost 采样不放回: 返回的 k 个 item 互不相同."""
        import random
        from goose_digging.mining.seed import _boost_sample
        items = [(f"a{i}", f"b{i}") for i in range(100)]
        signals = [float(i + 1) for i in range(100)]
        for s in range(5):
            rng = random.Random(s)
            sampled = _boost_sample(items, signals, 50, 0.05, rng)
            assert len(sampled) == len(set(sampled))

    def test_boost_sample_returns_all_when_k_ge_len(self):
        """k >= len(items) 时返回全部 (顺序保留)."""
        import random
        from goose_digging.mining.seed import _boost_sample
        items = ["a", "b", "c"]
        out = _boost_sample(items, [1.0, 2.0, 3.0], 10, 0.05, random.Random(0))
        assert set(out) == {"a", "b", "c"}

    def test_boost_sample_near_uniform(self):
        """近均匀: 高分 item 被采概率只比低分略高 (boost_alpha 小).
        统计 1000 次采样, 最高分项与最低分项的被采次数比值应远小于纯按 signal 加权."""
        import random
        from goose_digging.mining.seed import _boost_sample
        # 10 个 item, signal 从 1 到 10 (高低分都有)
        items = list(range(10))
        signals = [float(i + 1) for i in range(10)]
        k = 5
        alpha = 0.05  # SEED_BOOST_ALPHA 默认
        counts = [0] * 10
        for s in range(2000):
            rng = random.Random(s)
            for it in _boost_sample(items, signals, k, alpha, rng):
                counts[it] += 1
        # 高分(9, signal=10) vs 低分(0, signal=1). 权重比 (1+10*0.05)/(1+1*0.05)=1.5/1.05≈1.43
        # 纯 signal 加权会是 10:1; 纯均匀是 1:1. 近均匀应落在 [1.0, 2.5] 区间.
        ratio = counts[9] / counts[0]
        assert 1.0 <= ratio <= 2.5, f"boost 偏置过大或过小: ratio={ratio:.2f}"

    def test_boost_sample_zero_alpha_is_uniform(self):
        """boost_alpha=0 时退化为纯均匀随机 (signal 无影响)."""
        import random
        from goose_digging.mining.seed import _boost_sample
        items = list(range(10))
        # 极端 signal: 一个 1000, 其余 1. alpha=0 应完全忽略.
        signals = [1.0] * 10
        signals[0] = 1000.0
        counts = [0] * 10
        for s in range(2000):
            rng = random.Random(s)
            for it in _boost_sample(items, signals, 5, 0.0, rng):
                counts[it] += 1
        # 均匀: 各项被采次数应接近 (max/min < 2)
        assert max(counts) / min(counts) < 2.0, f"alpha=0 不均匀: {counts}"

    def test_sample_same_seed_stable(self):
        import random
        from goose_digging.mining.seed import sample_word_maps
        wm = [(f"a{i}", f"b{i}", float(i)) for i in range(50)]
        assert sample_word_maps(wm, 20, 0.05, random.Random(1)) == \
               sample_word_maps(wm, 20, 0.05, random.Random(1))

    def test_sample_diff_seed_diff(self):
        import random
        from goose_digging.mining.seed import sample_word_maps
        wm = [(f"a{i}", f"b{i}", float(i)) for i in range(50)]
        assert sample_word_maps(wm, 20, 0.05, random.Random(1)) != \
               sample_word_maps(wm, 20, 0.05, random.Random(2))

    def test_load_seed_pairs_excludes_seed_dir(self, tmp_out):
        """load_seed_pairs 排除 dir 以 seed 开头的 pair (传参隔离, 不改全局)."""
        with (tmp_out / "scored.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"left": "x", "right": "y", "dir": "seed_Qwen",
                                "score": 9, "scores": [9], "whys": [""]}) + "\n")
        from goose_digging.mining.seed import load_seed_pairs
        pairs = load_seed_pairs(out_dir=tmp_out)  # 传参, 不改全局
        dirs = {p.get("dir") for p in pairs}
        assert all(not str(d).startswith("seed") for d in dirs)

    def test_load_seed_pairs_includes_low_score(self, tmp_out):
        """load_seed_pairs 不卡分数门槛: 低分 pair 也进池 (不丢任何积木材料).

        动机: 低分 pair 单看平淡, 但塞进长句后可能撞出高分反差. 高低分之间的权重区分
        交给下游 sample_* 的近均匀 boost 采样, 而不是在这里硬卡 score>=门槛 丢掉.
        旧版卡 SEED_MIN_SCORE=4.5 会丢掉 ~92% 的 pair, 违背"低分也采"的设计意图.
        """
        # tmp_out fixture 里全是 score=8 的 pair; 额外写两条低分的
        with (tmp_out / "scored.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"left": "低分甲", "right": "r1", "dir": "fwd",
                                "score": 1.0}) + "\n")
            f.write(json.dumps({"left": "低分乙", "right": "r2", "dir": "fwd",
                                "score": 0.0}) + "\n")
        from goose_digging.mining.seed import load_seed_pairs
        pairs = load_seed_pairs(out_dir=tmp_out)
        lefts = {p.get("left") for p in pairs}
        assert "低分甲" in lefts and "低分乙" in lefts, "低分 pair 被错误丢弃"
        # 既有高分 pair 仍在
        assert "义父" in lefts

    def test_extract_word_maps_only_2char(self):
        from goose_digging.mining.seed import extract_word_maps
        pairs = [{"left": "义父", "right": "盗摄", "score": 9},
                 {"left": "射粪头", "right": "纪实片", "score": 9},
                 {"left": "挺住", "right": "尿住", "score": 8}]
        wm = extract_word_maps(pairs)
        assert {len(s) for s, _, _ in wm} == {2}

    def test_load_seed_pairs_tolerates_blank_lines(self, tmp_path):
        """load_seed_pairs 容忍空行 (末尾无换行/编辑残留, 无害)."""
        sp = tmp_path / "scored.jsonl"
        sp.write_text(
            json.dumps({"left": "a", "right": "b", "dir": "fwd", "score": 8}) + "\n"
            "\n"   # 空行
            + json.dumps({"left": "c", "right": "d", "dir": "fwd", "score": 9}) + "\n",
            encoding="utf-8")
        from goose_digging.mining.seed import load_seed_pairs
        pairs = load_seed_pairs(out_dir=tmp_path)
        assert len(pairs) == 2   # 空行跳过, 2 条合法的都解析

    def test_load_seed_pairs_fails_fast_on_bad_json(self, tmp_path):
        """坏 JSON 行直接抛 (fail-fast): scored.jsonl 每行都该合法, 坏行=写盘 bug, 不能吞."""
        import pytest
        sp = tmp_path / "scored.jsonl"
        sp.write_text(
            json.dumps({"left": "a", "right": "b", "dir": "fwd", "score": 8}) + "\n"
            "{这不是合法json\n",   # 坏行
            encoding="utf-8")
        from goose_digging.mining.seed import load_seed_pairs
        with pytest.raises(json.JSONDecodeError):
            load_seed_pairs(out_dir=tmp_path)


# ---------------------------------------------------------------------------
# 2b. 惰性枚举 (enumerate.py: _LazyPairs, cursor 走词表, pair 现场算)
# ---------------------------------------------------------------------------

class TestLazyEnumeration:
    def test_lazy_pairs_len_is_word_count(self):
        """enumerate_pairs() 返回惰性序列, len() = 词表长度 (瞬间, 不算 pair)."""
        from goose_digging.mining import enumerate as enum_mod
        ep = enum_mod.enumerate_pairs()
        n = len(ep)
        assert n > 100000   # 词表百万级
        # len 应等于 shuffled_words 长度
        assert n == len(enum_mod._shuffled_words())

    def test_lazy_getitem_returns_list_of_pairs(self):
        """__getitem__(i) 返回第 i 个词的 pair 列表 (0-3 个, list[dict])."""
        from goose_digging.mining import enumerate as enum_mod
        ep = enum_mod.enumerate_pairs()
        # 找一个产 pair 的词 (前 1000 词里必有)
        found = False
        for i in range(1000):
            pairs = ep[i]
            assert isinstance(pairs, list)
            for p in pairs:
                assert "left" in p and "right" in p and "dir" in p
                assert p["dir"] in ("fwd", "rev", "fixed")
            if pairs:
                found = True
                break
        assert found, "前1000词没产出任何 pair"

    def test_lazy_deterministic_across_instances(self):
        """惰性序列跨实例确定: 两次 enumerate_pairs() 同索引产同样 pair (cursor 续跑基础)."""
        from goose_digging.mining import enumerate as enum_mod
        ep1 = enum_mod.enumerate_pairs()
        ep2 = enum_mod.enumerate_pairs()
        # 同索引 (词) 应产同样 pair (词表固定 seed shuffle, goose 是查表)
        for i in [0, 100, 1000]:
            assert ep1[i] == ep2[i], f"索引 {i} 跨实例不一致"

    def test_lazy_bidict_pairs_both_in_dict(self):
        """enumerate_both_in_dict() 的 pair: left 和 goose(left) 都在词典."""
        from goose_digging.mining import enumerate as enum_mod
        from goose_digging.oracle import goose
        from goose_digging.mining.wordlist import load_static_dictionary
        words = load_static_dictionary()
        eb = enum_mod.enumerate_both_in_dict()
        # 抽样前若干词, 验证产出的 pair 两边都在词典
        checked = 0
        for i in range(5000):
            for p in eb[i]:
                assert p["left"] in words
                assert goose(p["left"]) == p["right"]
                assert p["right"] in words
                checked += 1
                if checked >= 20:
                    return
        assert checked > 0, "前5000词没产 bidict pair"

    def test_next_batch_handles_lazy_list_items(self):
        """_next_batch 兼容惰性序列: __getitem__ 返回 list[dict] 时不崩, 正确取批."""
        from goose_digging.mining import pipeline, state as st_mod
        # 造假 lazy 序列: 每索引返回 list[dict] (模拟 _LazyPairs)
        class FakeLazy:
            def __init__(self, data):
                self._d = data
            def __len__(self):
                return len(self._d)
            def __getitem__(self, i):
                return self._d[i]
        # 每个索引 0-2 个 pair
        data = [[{"left": f"a{i}", "right": f"b{i}", "dir": "fwd"}] if i % 2 == 0 else []
                for i in range(10)]
        lazy = FakeLazy(data)
        st = st_mod.MiningState()
        batch = pipeline._next_batch(st, lazy, 3, phase="bidict")
        assert len(batch) == 3
        assert st.cursor_bidict > 0


# ---------------------------------------------------------------------------
# 3. state 持久化
# ---------------------------------------------------------------------------

class TestState:
    def test_save_load_roundtrip(self, tmp_path):
        sp = tmp_path / "state.json"
        st = state.MiningState()
        st.seen_pairs = {("a", "b"), ("c", "d")}
        st.gold_pairs = [finding.Finding(S="a", real_goose_S="b", why="x",
                                          round=1, score=9.0, direction="fwd")]
        st.n_iter = 42; st.cursor_full = 100
        st.ga_population = [{"s": "义父", "scores": [8, 7], "born": 3}]
        st.ga_epoch = 5
        st.save(path=sp)
        st2 = state.MiningState.load(path=sp)
        assert st2.seen_pairs == {("a", "b"), ("c", "d")}
        assert len(st2.gold_pairs) == 1 and st2.gold_pairs[0].S == "a"
        assert st2.n_iter == 42 and st2.cursor_full == 100
        assert st2.ga_epoch == 5
        assert len(st2.ga_population) == 1 and st2.ga_population[0]["s"] == "义父"

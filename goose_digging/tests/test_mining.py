# -*- coding: utf-8 -*-
"""mining 挖掘管线测试套件.

所有 LLM 调用 mock, 所有文件 IO 通过传参 (out_dir/state_path) 隔离到 tmp_path,
绝不碰真实 mined/ 数据 —— 不用 monkeypatch 改全局 (那会漏掉默认参数绑定等问题).

覆盖:
  - prompt 生成 (fluency/score/gen_sentence 格式)
  - 采样/温度/shuffle/去重 (seed.py)
  - iterate_seed 完整链路 (造句→T2→落盘→state.save, gold标准max)
  - iterate bidict (跳T1→T2) / full (T1并行→T2)
  - state 持久化 (save/load 往返)
  - fluency T1 (词典命中短路)
"""
import json
import re

import pytest

from goose_digging.mining import config, state, finding, pipeline
from goose_digging.mining import prompts
from goose_digging.mining.llm import ModelLogger


# ---------------------------------------------------------------------------
# fixture: 造假 scored.jsonl + 临时 state_path, 全部传参 (不改全局)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_out(tmp_path):
    """临时输出目录, 造假 scored.jsonl 种子. 测试把 out_dir/state_path 传给 pipeline."""
    out = tmp_path / "mined"
    out.mkdir()
    sp = out / "scored.jsonl"
    with sp.open("w", encoding="utf-8") as f:
        for L, R in [("义父", "盗摄"), ("挺住", "尿住"), ("通心", "看奶"),
                     ("鹿鹿", "集集"), ("粪厂", "实境")]:
            f.write(json.dumps({"left": L, "right": R, "dir": "fwd",
                                "score": 8.0, "scores": [8], "whys": ["x"]}) + "\n")
    return out


@pytest.fixture
def tmp_state_path(tmp_out):
    return tmp_out / "state.json"


@pytest.fixture
def mock_llm(monkeypatch):
    """mock 所有 llm_chat (T2评分/fluency通顺预筛), 返回 calls 供断言.
    GA 路径不再有造句 LLM (gen_ga_candidates 走进化). fake_score 对含 '义' 的 left
    打 9 (会被 promote, 入 gold), 否则 4."""
    calls = {"score": [], "fluency": []}

    def fake_score(client, model, system, user, **kw):
        lefts = re.findall(r'\d+\. (.+?) →', user)
        parts = []
        for L in lefts:
            sc = 9 if "义" in L else 4
            parts.append(f'{{"left":"{L}","score":{sc},"why":"测{model}"}}')
        calls["score"].append(model)
        return '{"scores":[' + ",".join(parts) + "]}"

    def fake_fluency(client, model, system, user, **kw):
        texts = re.findall(r'\d+\. (.+)', user)
        items = ",".join(f'{{"text":"{t}","ok":true}}' for t in texts if t)
        calls["fluency"].append(1)
        return '{"items":[' + items + "]}"

    monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)
    monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
    return calls


@pytest.fixture
def fake_model_loggers(tmp_path):
    return {m["model"]: ModelLogger(tmp_path / f"m_{m['model']}.log")
            for m in config.SCORE_MODELS}


# ---------------------------------------------------------------------------
# 1. prompt 生成
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_fluency_prompt_format(self):
        p = prompts.fluency_prompt(["母鸡", "熟鸡"])
        assert "2 个" in p and "母鸡" in p and "熟鸡" in p
        assert '"items"' in p and '"ok"' in p

    def test_fluency_system_has_three_criteria(self):
        """新 T1 通顺判据 (三选一): 词典/解释得通/可成句 任一即通顺."""
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

    def test_extract_word_maps_only_2char(self):
        from goose_digging.mining.seed import extract_word_maps
        pairs = [{"left": "义父", "right": "盗摄", "score": 9},
                 {"left": "射粪头", "right": "纪实片", "score": 9},
                 {"left": "挺住", "right": "尿住", "score": 8}]
        wm = extract_word_maps(pairs)
        assert {len(s) for s, _, _ in wm} == {2}


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
        st.ga_population = [{"s": "义父", "surrogate": 7.0, "scores": [8, 7], "born": 3}]
        st.ga_epoch = 5
        st.save(path=sp)
        st2 = state.MiningState.load(path=sp)
        assert st2.seen_pairs == {("a", "b"), ("c", "d")}
        assert len(st2.gold_pairs) == 1 and st2.gold_pairs[0].S == "a"
        assert st2.n_iter == 42 and st2.cursor_full == 100
        assert st2.ga_epoch == 5
        assert len(st2.ga_population) == 1 and st2.ga_population[0]["s"] == "义父"

    def test_load_tolerates_missing_ga_fields(self, tmp_path):
        """旧 state.json 无 GA 字段时 load 不崩 (容忍旧盘续跑)."""
        sp = tmp_path / "state.json"
        sp.write_text(json.dumps({
            "seen_pairs": [["a", "b"]], "gold_pairs": [], "n_iter": 1,
            "phase": "seed", "cursor_bidict": 0, "cursor_full": 0,
        }), encoding="utf-8")
        st = state.MiningState.load(path=sp)
        assert st.ga_population == []
        assert st.ga_epoch == 0


# ---------------------------------------------------------------------------
# 4. fluency T1
# ---------------------------------------------------------------------------

class TestFluency:
    def test_dict_hit_short_circuit(self, mock_llm):
        from goose_digging.mining.wordlist import load_static_dictionary
        from goose_digging.mining.fluency import fluency_filter
        static = load_static_dictionary()
        if not static:
            pytest.skip("词典空")
        dict_word = next(iter(static))
        pairs = [{"left": dict_word, "right": dict_word, "dir": "fwd"}]
        survivors = fluency_filter(None, pairs, lambda s: None, None)
        assert len(survivors) == 1
        assert len(mock_llm["fluency"]) == 0  # 词典命中, 没调 LLM

    def test_both_sides_must_pass(self, monkeypatch):
        """两边都讲得通才存活: 左通右不通的 pair 被杀."""
        import re as _re
        from goose_digging.mining.fluency import fluency_filter

        def fake_fluency(client, model, system, user, **kw):
            texts = _re.findall(r'\d+\. (.+)', user)
            # 含'乱码'的不通, 其余通
            items = ",".join(f'{{"text":"{t}","ok":{str("乱码" not in t).lower()}}}'
                             for t in texts if t)
            return '{"items":[' + items + "]}"

        monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
        # 左通右不通 (右含'乱码') + 两边都通
        pairs = [{"left": "赶班", "right": "乱码字", "dir": "fwd"},
                 {"left": "撬松", "right": "赶班", "dir": "fwd"}]
        survivors = fluency_filter(None, pairs, lambda s: None, None)
        assert len(survivors) == 1   # 只有两边都通的存活
        assert survivors[0]["left"] == "撬松"

    def test_lenient_criteria_passes_explainable_words(self, monkeypatch):
        """新宽松判据: 能解释得通的盲区词 (赶班/撬松) 应被 LLM 判通顺.
        验证 prompt 鼓励放行 'explainable' 词语."""
        import re as _re
        from goose_digging.mining.fluency import fluency_filter
        from goose_digging.mining.prompts import FLUENCY_SYSTEM

        passed_system = []

        def fake_fluency(client, model, system, user, **kw):
            passed_system.append(system)
            texts = _re.findall(r'\d+\. (.+)', user)
            # 模拟宽松判: 赶班/撬松 (解释得通) 判通, 乱码判不通
            def ok(t):
                return t in ("赶班", "撬松") or "乱" not in t
            items = ",".join(f'{{"text":"{t}","ok":{str(ok(t)).lower()}}}'
                             for t in texts if t)
            return '{"items":[' + items + "]}"

        monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
        pairs = [{"left": "赶班", "right": "撬松", "dir": "fwd"}]
        survivors = fluency_filter(None, pairs, lambda s: None, None)
        assert len(survivors) == 1   # 两边都通 → 存活
        assert FLUENCY_SYSTEM in passed_system   # 走的是新宽松 prompt


# ---------------------------------------------------------------------------
# 5. iterate_seed 完整链路 (传参隔离, 绝不碰真实数据)
# ---------------------------------------------------------------------------

class TestIterateSeed:
    def test_full_chain(self, tmp_out, tmp_state_path, mock_llm, fake_model_loggers):
        """iterate_seed 端到端 (GA 路径): 进化→全模型打分→落盘→state.save.
        全部传参隔离到 tmp_out, 不碰真实 mined/. GA 走全模型 score_pairs 打分."""
        from goose_digging.mining.log import Logger
        st = state.MiningState(); st.n_iter = 100
        log = Logger(tmp_out / "run.log")
        try:
            findings = pipeline.iterate_seed(
                None, st, log, 30, None, model_loggers=fake_model_loggers,
                out_dir=tmp_out, state_path=tmp_state_path)
        finally:
            log.close()
            for ml in fake_model_loggers.values(): ml.close()

        # GA 走全模型 score_pairs 打分 (不再有 surrogate/造句)
        assert len(mock_llm["score"]) >= 1   # 全模型打过
        # 落盘到 tmp_out (不是真实 mined/)
        scored = (tmp_out / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        seed_lines = [ln for ln in scored if "seed_ga" in ln]
        assert len(seed_lines) > 0
        for ln in seed_lines:
            d = json.loads(ln)
            assert d["dir"] == "seed_ga"
            assert "scores" in d
        # state 持久化了 GA 种群 + epoch
        st2 = state.MiningState.load(path=tmp_state_path)
        assert len(st2.ga_population) > 0
        assert st2.ga_epoch >= 1

    def test_gold_standard_max(self, tmp_out, tmp_state_path, mock_llm, fake_model_loggers):
        """gold 标准: max(scores)>=SCORE_THRESH 才入 gold (不是 avg). GA 路径同理."""
        from goose_digging.mining.log import Logger
        st = state.MiningState()
        log = Logger(tmp_out / "run.log")
        try:
            pipeline.iterate_seed(None, st, log, 30, None,
                                  model_loggers=fake_model_loggers,
                                  out_dir=tmp_out, state_path=tmp_state_path)
        finally:
            log.close()
            for ml in fake_model_loggers.values(): ml.close()

        scored = (tmp_out / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        st2 = state.MiningState.load(path=tmp_state_path)
        for ln in scored:
            d = json.loads(ln)
            if d.get("dir") != "seed_ga": continue
            mx = max(d.get("scores", [0]))
            in_gold = any(d["left"] == g.S for g in st2.gold_pairs)
            assert (mx >= config.SCORE_THRESH) == in_gold, \
                f"{d['left']} max={mx} in_gold={in_gold}"


# ---------------------------------------------------------------------------
# 6. iterate bidict / full (传参隔离)
# ---------------------------------------------------------------------------

class TestIterate:
    def _make_pairs(self, n):
        return [{"left": f"词{i}", "right": f"右{i}", "dir": "fwd"} for i in range(n)]

    def test_bidict_skip_t1(self, tmp_out, tmp_state_path, mock_llm, fake_model_loggers):
        from goose_digging.mining.log import Logger
        st = state.MiningState()
        pairs = self._make_pairs(config.SCORE_BATCH + 5)
        log = Logger(tmp_out / "run.log")
        try:
            findings = pipeline.iterate(None, st, pairs, log, skip_T1=True,
                                        phase="bidict", model_loggers=fake_model_loggers,
                                        out_dir=tmp_out, state_path=tmp_state_path)
        finally:
            log.close()
            for ml in fake_model_loggers.values(): ml.close()
        assert len(mock_llm["fluency"]) == 0  # bidict 不跑 T1
        assert len(mock_llm["score"]) == len(config.SCORE_MODELS)
        assert len(findings) > 0

    def test_full_runs_t1(self, tmp_out, tmp_state_path, mock_llm, fake_model_loggers):
        from goose_digging.mining.log import Logger
        st = state.MiningState()
        pairs = self._make_pairs(config.T1_FETCH + 5)
        log = Logger(tmp_out / "run.log")
        try:
            pipeline.iterate(None, st, pairs, log, skip_T1=False,
                                        phase="full", model_loggers=fake_model_loggers,
                                        out_dir=tmp_out, state_path=tmp_state_path)
        finally:
            log.close()
            for ml in fake_model_loggers.values(): ml.close()
        assert len(mock_llm["fluency"]) > 0  # full 跑了 T1
        assert len(mock_llm["score"]) == len(config.SCORE_MODELS)

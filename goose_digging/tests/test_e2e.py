# -*- coding: utf-8 -*-
"""秒级 e2e (mock LLM): 同一 state 贯穿 bidict → full → seed, 中途 assert 覆盖集成点.

替代旧的散落集成测试 (TestIterate/TestIterateSeed/TestEvolve/TestFluencyPrefilter):
一个测试跑完整三阶段链路, 中途断言保证关键行为不回归.

真复刻 (不造假候选):
  - 真 enumerate_both_in_dict() / enumerate_pairs() (惰性序列)
  - cursor 真走词表索引, pair 现场算
  - mock LLM 分数有梯度 (按内容, 不是全 2 分), 让 GA 有真信号

全程 mock LLM (不调真实 API), 隔离 tmp_path, 绝不碰真实 mined/.

验证点 (中途 assert):
  1. 三阶段同一 state 串接: cursor/gold/seen_pairs 跨阶段累积
  2. bidict 跳预筛; full 跑预筛 (词典命中短路 + 词典外送 LLM)
  3. GA 端到端: 进化 → 全模型打分 → 落盘 seed_ga → state 持久化种群/epoch
  4. gold 标准: max(scores) >= SCORE_THRESH 才入 gold
  5. 断点续传: load state 后 epoch 推进 1
  6. fluency_killed stats 字段存在
  7. GA 预筛用宽松 GA_FLUENCY_SYSTEM (不是枚举的严格 FLUENCY_SYSTEM)
"""
import json
import re

import pytest

from goose_digging.mining import config, state, pipeline
from goose_digging.mining.llm import ModelLogger
from goose_digging.mining.log import Logger


# ---------------------------------------------------------------------------
# 规模: 秒级. 词表百万级, 取前 N 个词; GA 轮数压低.
# ---------------------------------------------------------------------------
BIDICT_WORD_LIMIT = 3000    # bidict 跑前 3000 词 (够攒满 SCORE_BATCH 几批)
FULL_WORD_LIMIT = 2000      # full 跑前 2000 词
N_GA_EPOCHS = 5             # GA 轮数 (够触发 evict/aging/续传). offspring=220 后每轮重, 压低轮数保秒级


# ---------------------------------------------------------------------------
# mock LLM: 分数有梯度 (模拟真实神鹅语分布)
# ---------------------------------------------------------------------------

def _mock_score(left):
    """按内容打分, 有梯度. 含特定字给高分 (会进 gold), 其余按 hash 散布."""
    if "义" in left:
        return 9
    if "粪" in left:
        return 8
    if "盗" in left:
        return 8
    if "挺" in left or "通" in left:
        return 7
    h = sum(ord(c) for c in left) % 10
    if h <= 5:
        return 2
    if h <= 8:
        return 4
    return 6


@pytest.fixture
def mock_llm(monkeypatch):
    """mock 评分 + 预筛 LLM. 记录调用, 供中途 assert.

    预筛单边通顺率 = 28% (GA 路径 pair 存活率 8% 反推: 8% = 双边都通 = p², p≈28%).
    fluency 对每个 text 单边判 (去重 set 各判各), 代码回来配对: 两边都 ok 才存活.
    故单边判 28% → 双边联合 ~8% pair 存活 → offspring=220 存活 ~12 进打分, 与真实一致.
    若 mock 全判通, offspring=220 全存活全打分 (200+条), 慢且打分量与真实差 17×.
    用 hash(t)%100<28 确定性判通 (可复现). 词典短路由 fluency.py 在 LLM 调用前做.
    score 按内容打分有梯度让 GA 有真信号.
    """
    calls = {"score": [], "fluency": [], "fluency_systems": []}

    def fake_score(client, model, system, user, **kw):
        calls["score"].append(model)
        lefts = re.findall(r'\d+\. (.+?) .', user)
        parts = [{"left": L, "score": _mock_score(L), "why": "m"} for L in lefts]
        return json.dumps({"scores": parts}, ensure_ascii=False)

    def fake_fluency(client, model, system, user, **kw):
        calls["fluency"].append(1)
        calls["fluency_systems"].append(system)
        texts = re.findall(r'\d+\. (.+)', user)
        items = [{"text": t, "ok": (hash(t) % 100) < 28} for t in texts if t]
        return json.dumps({"items": items}, ensure_ascii=False)

    monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)
    monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
    return calls


@pytest.fixture
def model_loggers(tmp_path):
    return {m["model"]: ModelLogger(tmp_path / f"m_{m['model']}.log")
            for m in config.SCORE_MODELS}


# ---------------------------------------------------------------------------
# e2e
# ---------------------------------------------------------------------------

def test_full_pipeline_e2e(tmp_path, mock_llm, model_loggers):
    """三阶段串接 + 断点续传, 一个测试覆盖所有集成点."""
    from goose_digging.mining import enumerate as enum_mod
    from goose_digging.mining.ga.population import Population
    from goose_digging.mining.ga.genome import Individual

    out_dir = tmp_path / "mined"
    out_dir.mkdir()
    (out_dir / "logs").mkdir()
    sp = out_dir / "state.json"
    log = Logger(out_dir / "logs" / "run.log")
    try:
        st = state.MiningState()

        # ====== 阶段 1: bidict (跳预筛, 直接评分) ======
        bidict_pairs = enum_mod.enumerate_both_in_dict()
        n_bidict_iter = 0
        while st.cursor_bidict < BIDICT_WORD_LIMIT and st.cursor_bidict < len(bidict_pairs):
            pipeline.iterate(None, st, bidict_pairs, log, skip_fluency=True,
                             phase="bidict", model_loggers=model_loggers,
                             out_dir=out_dir, state_path=sp)
            n_bidict_iter += 1

        # [验证] bidict 跑了: cursor 推进, gold 累积, scored 落盘
        assert st.cursor_bidict > 0, "bidict cursor 没推进"
        assert n_bidict_iter >= 1
        assert len(mock_llm["score"]) >= 1   # 评分调过
        assert len(mock_llm["fluency"]) == 0  # bidict 跳预筛
        n_gold_after_bidict = len(st.gold_pairs)
        scored_lines = (out_dir / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert any(json.loads(ln).get("dir") in ("fwd", "rev", "fixed")
                   for ln in scored_lines if ln), "bidict 产物没落盘"
        n_scored_after_bidict = len([ln for ln in scored_lines if ln])

        # ====== 阶段 2: full (跑预筛 + 评分) ======
        n_fluency_before_full = len(mock_llm["fluency"])
        full_pairs = enum_mod.enumerate_pairs()
        n_full_iter = 0
        while st.cursor_full < FULL_WORD_LIMIT and st.cursor_full < len(full_pairs):
            pipeline.iterate(None, st, full_pairs, log, skip_fluency=False,
                             phase="full", model_loggers=model_loggers,
                             out_dir=out_dir, state_path=sp)
            n_full_iter += 1

        # [验证] full 跑了预筛 (bidict 不跑, full 跑 → fluency 调用数增加)
        assert len(mock_llm["fluency"]) > n_fluency_before_full, "full 没跑预筛"
        assert st.cursor_full > 0
        # seen_pairs 跨阶段累积 (不重判)
        assert len(st.seen_pairs) > 0

        # scored 跨阶段累积 (full 追加, 不覆盖)
        scored_lines = (out_dir / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        n_scored_after_full = len([ln for ln in scored_lines if ln])
        assert n_scored_after_full > n_scored_after_bidict, "full 没追加评分记录"

        # ====== 阶段 3: GA 多轮 ======
        n_gold_before_ga = len(st.gold_pairs)
        for ep in range(N_GA_EPOCHS):
            pipeline.iterate_seed(None, st, log, config.GA_OFFSPRING_PER_EPOCH, None,
                                  model_loggers=model_loggers, out_dir=out_dir, state_path=sp)

        # [验证] GA 端到端: 种群/epoch 持久化, seen_s 增长, gold 用 max 标准
        assert st.ga_epoch >= N_GA_EPOCHS, "GA epoch 没推进"
        assert len(st.ga_population) > 0, "GA 种群空"
        assert len(st.ga_seen_s) >= N_GA_EPOCHS, "ga_seen_s 太少 (应防重烧)"

        # scored.jsonl 含 seed_ga 产物
        scored_lines = (out_dir / "scored.jsonl").read_text(encoding="utf-8").strip().split("\n")
        dirs = {}
        for ln in scored_lines:
            d = json.loads(ln)
            dd = d.get("dir", "")
            dirs[dd] = dirs.get(dd, 0) + 1
        assert "seed_ga" in dirs, "GA 产物 (seed_ga) 没落盘"

        # gold 标准: max(scores) >= SCORE_THRESH 才入 gold (对 seed_ga 行校验)
        for ln in scored_lines:
            d = json.loads(ln)
            if d.get("dir") != "seed_ga":
                continue
            mx = max(d.get("scores", [0]))
            in_gold = any(d["left"] == g.S for g in st.gold_pairs)
            assert (mx >= config.SCORE_THRESH) == in_gold, \
                f"gold 标准不符: {d['left']} max={mx} in_gold={in_gold}"

        # gold 持续累积 (GA 可能新增, 至少不减少)
        assert len(st.gold_pairs) >= n_gold_before_ga

        # ====== [验证] 断点续传: load 后 epoch +1 ======
        st_loaded = state.MiningState.load(path=sp)
        ep_before = st_loaded.ga_epoch
        assert ep_before >= N_GA_EPOCHS, "load 出来的 epoch 不对"
        pipeline.iterate_seed(None, st_loaded, log, config.GA_OFFSPRING_PER_EPOCH, None,
                              model_loggers=model_loggers, out_dir=out_dir, state_path=sp)
        st_loaded2 = state.MiningState.load(path=sp)
        assert st_loaded2.ga_epoch == ep_before + 1, "续传没推进 epoch"

        # 种群多样性不塌缩 (防早熟回归)
        pop = Population([Individual.from_dict(d) for d in st_loaded2.ga_population],
                         config.GA_POP_SIZE, config.GA_SHARING_SIGMA, config.GA_ELITE)
        assert pop.mean_diversity() >= 1.0, f"种群早熟塌缩: div={pop.mean_diversity()}"

    finally:
        for ml in model_loggers.values():
            ml.close()
        log.close()


def test_ga_fluency_uses_lenient_prompt(tmp_path, monkeypatch):
    """GA 预筛用宽松 GA_FLUENCY_SYSTEM, 不是枚举路径的严格 FLUENCY_SYSTEM."""
    from goose_digging.mining import pipeline
    from goose_digging.mining.prompts import GA_FLUENCY_SYSTEM, FLUENCY_SYSTEM

    # 造假 scored.jsonl 种子 (GA 起点)
    out_dir = tmp_path / "mined"; out_dir.mkdir()
    (out_dir / "logs").mkdir()
    sp = out_dir / "state.json"
    from goose_digging.oracle import goose
    with (out_dir / "scored.jsonl").open("w", encoding="utf-8") as f:
        for L in ["义父", "粪厂", "通心", "挺住", "义父盗"]:
            f.write(json.dumps({"left": L, "right": goose(L), "dir": "fwd",
                                "score": 8.0, "scores": [8], "whys": ["x"]}) + "\n")

    seen_systems = []

    def fake_fluency(client, model, system, user, **kw):
        seen_systems.append(system)
        texts = re.findall(r'\d+\. (.+)', user)
        items = ",".join(f'{{"text":"{t}","ok":{str((hash(t) % 100) < 28).lower()}}}'
                         for t in texts if t)
        return '{"items":[' + items + "]}"

    def fake_score(client, model, system, user, **kw):
        lefts = re.findall(r'\d+\. (.+?) .', user)
        parts = [{"left": L, "score": 4, "why": "x"} for L in lefts]
        return json.dumps({"scores": parts}, ensure_ascii=False)

    monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
    monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)

    log = Logger(out_dir / "logs" / "run.log")
    mls = {m["model"]: ModelLogger(out_dir / f"m_{m['model']}.log")
           for m in config.SCORE_MODELS}
    try:
        pipeline.iterate_seed(None, state.MiningState(), log, 20, None,
                              model_loggers=mls, out_dir=out_dir, state_path=sp)
    finally:
        for ml in mls.values():
            ml.close()
        log.close()

    assert any(GA_FLUENCY_SYSTEM in s for s in seen_systems), "GA 预筛没用 GA_FLUENCY_SYSTEM"
    assert GA_FLUENCY_SYSTEM != FLUENCY_SYSTEM, "宽松/严格 prompt 不能一样"


def test_ga_stats_has_fluency_killed(tmp_path, monkeypatch):
    """evolve 返回的 stats 含 fluency_killed 字段 (预筛杀了多少候选)."""
    from goose_digging.mining.ga import evolve
    from goose_digging.mining.seed import extract_char_maps, extract_word_maps
    from goose_digging.oracle import goose

    def fake_fluency(client, model, system, user, **kw):
        texts = re.findall(r'\d+\. (.+)', user)
        items = ",".join(f'{{"text":"{t}","ok":{str((hash(t) % 100) < 28).lower()}}}'
                         for t in texts if t)
        return '{"items":[' + items + "]}"

    def fake_score(client, model, system, user, **kw):
        lefts = re.findall(r'\d+\. (.+?) .', user)
        parts = [{"left": L, "score": 4, "why": "x"} for L in lefts]
        return json.dumps({"scores": parts}, ensure_ascii=False)

    monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
    monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)

    ga_seed = [{"left": "义父盗", "right": goose("义父盗"), "dir": "fwd", "score": 9.0}]
    chars = extract_char_maps(ga_seed)
    words = extract_word_maps(ga_seed)
    r = evolve.evolve(None, ga_seed, chars, words, 20, 1, lambda s: None, None,
                      existing_population=None, existing_epoch=0, rng=__import__("random").Random(1))
    assert "fluency_killed" in r.stats

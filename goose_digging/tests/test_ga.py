# -*- coding: utf-8 -*-
"""GA 神鹅语进化单测: 算子 / 种群多样性 / 两层适应度 / 端到端 evolve.

全部 LLM 调用 mock, 全部纯函数/容器测试零外部依赖. evolve 端到端用 mock_llm 验证
surrogate→promote 两层流程产物结构正确.
"""
import random

import pytest

from goose_digging.mining.ga import genome, population, fitness, evolve
from goose_digging.mining.ga.genome import (
    Individual, crossover, mutate, immigrant, is_viable,
)
from goose_digging.mining.ga.population import (
    Population, levenshtein, shared_fitness,
)
from goose_digging.oracle import goose
from pathlib import Path


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

CHAR_SAMPLE = [("挺", "尿"), ("粪", "实"), ("义", "盗"),
               ("通", "看"), ("厂", "境"), ("心", "辞")]
WORD_SAMPLE = [("义父", "盗摄", 9.0), ("粪厂", "实境", 10.0),
               ("通心", "看奶", 8.0), ("挺住", "尿住", 8.0)]


# ---------------------------------------------------------------------------
# 1. 基因型
# ---------------------------------------------------------------------------

class TestIndividual:
    def test_raw_score_prefers_full_scores(self):
        ind = Individual(s="义父", surrogate=4.0, scores=[8, 9])
        assert ind.raw_score() == 8.5   # 全模型均分优先

    def test_raw_score_falls_back_surrogate(self):
        ind = Individual(s="义父", surrogate=7.0)
        assert ind.raw_score() == 7.0

    def test_raw_score_zero_when_unscored(self):
        assert Individual(s="x").raw_score() == 0.0

    def test_dict_roundtrip(self):
        """to_dict/from_dict 往返: scores/born 保留; surrogate 已废弃不再序列化."""
        ind = Individual(s="义父", surrogate=4.0, scores=[8, 9], born=3)
        ind2 = Individual.from_dict(ind.to_dict())
        assert ind2.s == "义父"
        assert ind2.scores == [8, 9] and ind2.born == 3
        # surrogate 已废弃: to_dict 不再写它, 往返后为 None
        assert ind2.surrogate is None

    def test_from_dict_reads_legacy_surrogate(self):
        """旧 state.json 含 surrogate 字段时 from_dict 不崩 (向后兼容读盘)."""
        ind = Individual.from_dict({"s": "老串", "surrogate": 7.0, "scores": None, "born": 1})
        assert ind.s == "老串"
        assert ind.surrogate == 7.0   # 旧盘的 surrogate 读进来 (raw_score 会回退用它)
        assert ind.raw_score() == 7.0  # scores 缺失时回退 surrogate


# ---------------------------------------------------------------------------
# 2. 算子: crossover / mutate / immigrant
# ---------------------------------------------------------------------------

class TestOperators:
    def test_crossover_length_within_bounds(self):
        rng = random.Random(7)
        for _ in range(50):
            child = crossover("义父盗摄", "通心看奶", 3, 9, rng)
            assert 3 <= len(child) <= 9
            assert all("\u4e00" <= c <= "\u9fff" for c in child)

    def test_crossover_substrings_from_parents(self):
        """段拼接: child 应是父A前缀 + 父B后缀的子串组合."""
        rng = random.Random(1)
        child = crossover("义父", "通心", 3, 4, rng)
        # 4 字以内, 由父片段拼成
        assert len(child) <= 4

    def test_crossover_empty_parent_degenerate(self):
        rng = random.Random(1)
        # 空父退化为另一父裁剪
        child = crossover("", "义父通心", 3, 4, rng)
        assert child  # 不空

    def test_mutate_changes_string(self):
        rng = random.Random(11)
        s = "义父看奶"
        changed = 0
        for _ in range(30):
            m = mutate(s, CHAR_SAMPLE, WORD_SAMPLE, 3, 9, rng)
            if m != s:
                changed += 1
            assert 3 <= len(m) <= 9
        # 绝大多数变异应改变串 (允许少数碰巧不变)
        assert changed >= 20

    def test_mutate_respects_max_len(self):
        rng = random.Random(3)
        for _ in range(50):
            m = mutate("义父看奶挺住", CHAR_SAMPLE, WORD_SAMPLE, 3, 5, rng)
            assert len(m) <= 5

    def test_immigrant_produces_cjk_of_target_len(self):
        rng = random.Random(5)
        im = immigrant(CHAR_SAMPLE, WORD_SAMPLE, 5, rng)
        assert len(im) == 5
        assert all("\u4e00" <= c <= "\u9fff" for c in im)

    def test_immigrant_diverse_across_seeds(self):
        # 不同种子拼出不同串 (温度采样移民是探索源)
        a = immigrant(CHAR_SAMPLE, WORD_SAMPLE, 5, random.Random(1))
        b = immigrant(CHAR_SAMPLE, WORD_SAMPLE, 5, random.Random(2))
        assert a != b


# ---------------------------------------------------------------------------
# 3. 预筛 is_viable
# ---------------------------------------------------------------------------

class TestViability:
    def test_rejects_too_short(self):
        assert not is_viable("义", 3, 9)

    def test_rejects_too_long(self):
        # 10 字超出 max_len=9
        assert not is_viable("义父通心挺住粪厂义通", 3, 9)

    def test_rejects_all_fixed_point(self):
        # 全不动点 (goose(S)==S): 词典里 goose 不变的串. 用表外字构造.
        s = "abc"  # 非 CJK, 也会被拒
        assert not is_viable(s, 3, 9)

    def test_rejects_non_cjk(self):
        assert not is_viable("abc", 3, 9)

    def test_accepts_valid_goose_transform(self):
        # 义父→盗摄 (有变换, 全常用字)
        assert is_viable("义父", 2, 9)

    def test_rejects_seen_pairs(self):
        seen = {("义父", goose("义父"))}
        assert not is_viable("义父", 2, 9, seen_pairs=seen)


# ---------------------------------------------------------------------------
# 4. 种群 + 多样性 (Levenshtein / fitness sharing / crowding)
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_levenshtein_basics(self):
        assert levenshtein("abc", "abc") == 0
        assert levenshtein("abc", "abd") == 1   # 替换
        assert levenshtein("abc", "ab") == 1    # 删除
        assert levenshtein("", "ab") == 2

    def test_levenshtein_cjk(self):
        assert levenshtein("脱内裤", "脱裤") == 1
        assert levenshtein("义父", "义父") == 0

    def test_shared_fitness_dilutes_clustered(self):
        """聚集的相似个体被稀释: 同分但独处的分数应高于被簇拥的."""
        lone = Individual(s="义父盗摄", surrogate=8.0)
        # 4 个和 lone 极相似的个体 (Levenshtein<=2) 围着它
        cluster = [Individual(s="义父盗摄", surrogate=8.0),
                   Individual(s="义父盗摄", surrogate=8.0)]
        sf_lone_in_cluster = shared_fitness(lone, [lone] + cluster, sigma=2.0)
        sf_lone_alone = shared_fitness(lone, [lone], sigma=2.0)
        assert sf_lone_in_cluster < sf_lone_alone   # 被簇稀释

    def test_dedupe_keeps_highest_score(self):
        inds = [Individual(s="义父", surrogate=4.0),
                Individual(s="义父", scores=[9, 9]),
                Individual(s="通心", surrogate=7.0)]
        pop = Population(inds, pop_size=10, sigma=2.0, elite=1)
        assert len(pop) == 2   # 义父 去重
        by_s = {i.s: i for i in pop.to_list()}
        assert by_s["义父"].raw_score() == 9.0   # 留高分那个

    def test_crowd_replace_preserves_elite(self):
        """精英 (最高分) 不被替换."""
        inds = [Individual(s=f"字{i}", scores=[float(i)]) for i in range(6)]
        pop = Population(inds, pop_size=6, sigma=2.0, elite=1)
        best_before = pop.best().s
        # 塞一个低分 offspring
        pop.crowd_replace(Individual(s="xyz低", scores=[0.0]), random.Random(1))
        assert pop.best().s == best_before   # 精英没被换掉

    def test_crowd_replace_replaces_worse_in_niche(self):
        """crowding: offspring 比 crowding 里最差的高才替换."""
        inds = [Individual(s="义父", scores=[3.0]),
                Individual(s="通心", scores=[5.0])]
        pop = Population(inds, pop_size=4, sigma=2.0, elite=0)
        replaced = pop.crowd_replace(
            Individual(s="义父盗摄", scores=[8.0]), random.Random(1))
        assert replaced   # 8 > 最近邻里最差(义父 3.0)

    def test_mean_diversity_increases_with_variety(self):
        similar = [Individual(s="义父"), Individual(s="义父盗")]
        diverse = [Individual(s="义父"), Individual(s="挺住通心粪厂")]
        p1 = Population(similar, 5, 2.0, 0)
        p2 = Population(diverse, 5, 2.0, 0)
        assert p2.mean_diversity() > p1.mean_diversity()

    def test_tournament_picks_higher_fitness(self):
        """锦标赛: 高分个体更可能被选 (统计上)."""
        inds = [Individual(s="低分", scores=[2.0]),
                Individual(s="高分", scores=[9.0])]
        pop = Population(inds, 2, 2.0, 0)
        rng = random.Random(0)
        picks = [pop.tournament(2, rng).s for _ in range(200)]
        assert picks.count("高分") > picks.count("低分")


# ---------------------------------------------------------------------------
# 4b. 防同化机制 (aging / evict gold / 全局 seen)
# ---------------------------------------------------------------------------

class TestAntiHomogenization:
    def test_cull_aged_removes_old_individuals(self):
        """超龄个体被淘汰: born + max_age <= cur_epoch 的移除."""
        pop = Population([
            Individual(s="老", scores=[9.0], born=0),   # 高分但老
            Individual(s="新", scores=[3.0], born=5),   # 低分但新
        ], pop_size=5, sigma=2.0, elite=0)
        pop.set_epoch(6)   # cur_epoch=6
        n = pop.cull_aged(max_age=3)   # born+3<=6 即 born<=3 被淘汰
        assert n == 1                     # "老"(born 0) 被淘汰
        strs = {i.s for i in pop.to_list()}
        assert "老" not in strs
        assert "新" in strs

    def test_cull_aged_high_score_still_culled(self):
        """关键: 即使是最高分, 超龄也要淘汰 (防高分个体永久主导种群)."""
        pop = Population([
            Individual(s="学霸", scores=[10.0], born=0),
            Individual(s="学渣", scores=[1.0], born=10),
        ], pop_size=5, sigma=2.0, elite=2)   # elite=2 但 aging 优先
        pop.set_epoch(10)
        n = pop.cull_aged(max_age=5)
        assert n == 1
        strs = {i.s for i in pop.to_list()}
        assert "学霸" not in strs   # 满分也被淘汰

    def test_evict_removes_promoted_individuals(self):
        """promoted (max>=thresh) 个体被 evict: 既存 gold 就不该再占种群."""
        pop = Population([
            Individual(s="已挖", scores=[7, 8]),    # max=8 >= 6, 已挖到
            Individual(s="未挖", scores=[2, 3]),    # max=3 < 6, 还在探索
            Individual(s="待挖", surrogate=5.0),    # 未升级
        ], pop_size=5, sigma=2.0, elite=0)
        from goose_digging.mining.config import GA_EVICT_SCORE_THRESH
        n = pop.evict(lambda ind: bool(ind.scores)
                      and max(ind.scores) >= GA_EVICT_SCORE_THRESH)
        assert n == 1
        strs = {i.s for i in pop.to_list()}
        assert "已挖" not in strs
        assert "未挖" in strs and "待挖" in strs

    def test_crowd_replace_aged_elite_can_be_replaced(self):
        """超龄精英可被替换 (max_age 让精英保护失效), 新生高分不挤不进."""
        # 一个满龄精英 + 一个空位让种群满
        inds = [Individual(s=f"字{i}", scores=[float(i)]) for i in range(6)]
        inds[5] = Individual(s="精英老", scores=[10.0], born=0)  # 最高分且最老
        pop = Population(inds, pop_size=6, sigma=2.0, elite=1)
        pop.set_epoch(10)
        # max_age=3: 精英 born=0, 0+3<=10 超龄 → 不再保护 → 可被替换
        replaced = pop.crowd_replace(
            Individual(s="新生高分", scores=[9.0]), random.Random(1), max_age=3)
        assert replaced

    def test_refill_restores_population_size(self):
        """cull/evict 后 refill 补回 pop_size."""
        pop = Population([
            Individual(s="老", scores=[9.0], born=0),
            Individual(s="新", scores=[3.0], born=5),
        ], pop_size=5, sigma=2.0, elite=0)
        pop.set_epoch(6)
        pop.cull_aged(max_age=3)   # 淘汰1个, 剩1个
        assert len(pop) == 1

        def factory(rng):
            cs = [("挺", "尿"), ("义", "盗")]
            return Individual(s=immigrant(cs, [], 4, rng), born=6)
        pop.refill(factory, random.Random(1))
        assert len(pop) == 5   # 补回


# ---------------------------------------------------------------------------
# 4c. Bug 修复回归测 (防止 5 个已修 bug 复现)
# ---------------------------------------------------------------------------

class TestBugFixes:
    def test_backbone_not_evicted(self):
        """Bug A/B: 骨干用 surrogate 存(非 scores), evict gold 不误杀.
        根因: 旧版骨干 scores=[int(score)], evict 看 max(scores)>=6 误杀高分骨干."""
        from goose_digging.mining.config import GA_EVICT_SCORE_THRESH
        # 骨干: surrogate=9 (高分, 但不是 scores)
        backbone = Individual(s="骨干高分", surrogate=9.0, born=1)
        promoted = Individual(s="成品", scores=[7, 8], born=1)  # 真 promoted
        pop = Population([backbone, promoted], pop_size=5, sigma=2.0, elite=0)
        n = pop.evict(lambda ind: bool(ind.scores)
                      and max(ind.scores) >= GA_EVICT_SCORE_THRESH)
        strs = {i.s for i in pop.to_list()}
        assert n == 1                       # 只 evict 真 promoted
        assert "骨干高分" in strs            # 骨干留存
        assert "成品" not in strs

    def test_refill_factory_produces_only_viable(self):
        """Bug E: refill 工厂补的个体必须 viable (拒全不动点/生僻字).
        根因: 旧版 _immigrant_factory 不查 is_viable, 36% 产物是垃圾."""
        # 工厂模拟: 只产 viable 个体
        def viable_factory(rng):
            for _ in range(20):
                cs = [("挺", "尿"), ("义", "盗"), ("粪", "实")]
                ws = [("义父", "盗摄", 9.0)]
                s = immigrant(cs, ws, rng.randint(3, 5), rng)
                if s and is_viable(s, 3, 9):
                    return Individual(s=s, born=1)
            return None
        pop = Population([Individual(s="种子甲")], pop_size=15, sigma=2.0, elite=0)
        pop.refill(viable_factory, random.Random(1))
        bad = [i for i in pop.to_list() if not is_viable(i.s, 3, 9) and i.s != "种子甲"]
        assert not bad, f"refill 补进垃圾: {[i.s for i in bad]}"

    def test_ga_seen_s_roundtrip(self, tmp_path):
        """Bug H: ga_seen_s 持久化往返 (跨 run 防重烧 LLM).
        根因: 旧版 surrogate-only S 不持久化, 跨 run 重新生成重烧 Flash."""
        from goose_digging.mining.state import MiningState
        sp = tmp_path / "state.json"
        st = MiningState()
        st.ga_seen_s = ["义父", "通心", "只过surrogate的串"]
        st.save(path=sp)
        st2 = MiningState.load(path=sp)
        assert st2.ga_seen_s == ["义父", "通心", "只过surrogate的串"]

    def test_ga_seen_s_old_state_compat(self, tmp_path):
        """Bug H: 旧 state.json 无 ga_seen_s 时 load 不崩."""
        import json
        from goose_digging.mining.state import MiningState
        sp = tmp_path / "old.json"
        sp.write_text(json.dumps({
            "seen_pairs": [], "gold_pairs": [], "n_iter": 0, "phase": "seed",
            "cursor_bidict": 0, "cursor_full": 0,
        }), encoding="utf-8")
        st = MiningState.load(path=sp)
        assert st.ga_seen_s == []

    def test_warm_start_no_infinite_loop(self):
        """Bug E 相关: warm_start 移民补满有重试上限, 不死循环.
        用极小积木池 (大部分 immigrant 不可行) 测试能正常返回."""
        from goose_digging.mining.ga.evolve import warm_start
        # 极小种子: 只有1对, immigrant 大概率失败 → 考验重试上限
        seed_pairs = [{"left": "义父盗", "right": "盗摄莨", "dir": "fwd", "score": 9.0}]
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        chars = extract_char_maps(seed_pairs)
        words = extract_word_maps(seed_pairs)
        import signal
        def alarm(s, f): raise TimeoutError("死循环")
        signal.signal(signal.SIGALRM, alarm); signal.alarm(10)
        try:
            pop = warm_start(seed_pairs, chars, words, 30, 1, random.Random(1))
            # 应在 10s 内返回 (即使补不满 pop_size 也不死循环)
            assert len(pop) >= 1
        finally:
            signal.alarm(0)

    def test_scoring_failure_does_not_crash_epoch(self, monkeypatch):
        """Bug F: 全模型打分失败时不应让整个 epoch 崩.
        根因: 旧版 score_offspring 直接调 llm_chat, 失败抛异常冒泡到 iterate_seed 崩.
        长时间挖掘时模型偶发故障很常见. T1 通顺预筛正常(默认全通), 全模型打分全失败."""
        from goose_digging.mining import pipeline, state as st_mod, config
        from goose_digging.mining.log import Logger
        from goose_digging.mining.llm import ModelLogger
        from goose_digging.oracle import goose
        import json as _json
        import re as _re

        tmp = _tmp_out_with_seeds()
        sp = tmp / "state.json"

        def fake_all_fail(client, model, system, user, **kw):
            raise RuntimeError(f"{model} 挂")
        # T1 通顺判: 正常全通 (让候选进到全模型打分环节, 才能测打分失败)
        def fake_fluency_ok(client, model, system, user, **kw):
            texts = _re.findall(r'\d+\. (.+)', user)
            items = ",".join(f'{{"text":"{t}","ok":true}}' for t in texts if t)
            return '{"items":[' + items + "]}"

        monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_all_fail)
        monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency_ok)

        st = st_mod.MiningState()
        log = Logger(tmp / "run.log")
        mls = {m["model"]: ModelLogger(tmp / f"m_{m['model']}.log")
               for m in config.SCORE_MODELS}
        try:
            findings = pipeline.iterate_seed(None, st, log, 30, None,
                                             model_loggers=mls, out_dir=tmp, state_path=sp)
        finally:
            for ml in mls.values(): ml.close()
            log.close()
        # 不崩 + 进化仍推进 (全模型失败 → score_pairs 内部降级返空, 无 findings)
        assert isinstance(findings, list)
        assert findings == []
        assert len(st.ga_population) > 0
        assert st.ga_epoch >= 1


def _tmp_out_with_seeds():
    """造假 scored.jsonl 种子目录 (供集成测试用)."""
    import tempfile, json as _json
    from goose_digging.oracle import goose
    tmp = Path(tempfile.mkdtemp()) / "mined"; tmp.mkdir()
    sp = tmp / "scored.jsonl"
    for L, R in [("义父", "盗摄"), ("挺住", "尿住"), ("通心", "看奶"),
                 ("粪厂", "实境"), ("义父盗", goose("义父盗"))]:
        sp.open("a", encoding="utf-8").write(
            _json.dumps({"left": L, "right": R, "dir": "fwd",
                         "score": 8.0, "scores": [8], "whys": ["x"]}) + "\n")
    return tmp


# ---------------------------------------------------------------------------
# 5. 端到端 evolve (mock LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def ga_mock_llm(monkeypatch):
    """mock scoring.llm_chat (全模型打分) + fluency.llm_chat (T1通顺预筛).
    对含 '义' 的 left 打 9 (promote+gold), 否则 4. T1 默认全通 (不杀)."""
    calls = {"score": [], "fluency": []}

    def fake_score(client, model, system, user, **kw):
        import re
        lefts = re.findall(r'\d+\. (.+?) →', user)
        parts = []
        for L in lefts:
            sc = 9 if "义" in L else 4
            parts.append(f'{{"left":"{L}","score":{sc},"why":"测{model}"}}')
        calls["score"].append(model)
        return '{"scores":[' + ",".join(parts) + "]}"

    def fake_fluency(client, model, system, user, **kw):
        """T1 通顺判: 默认全通 (返回所有 text ok), 不杀任何候选.
        个别测试需要杀可通过 calls['fluency_kill'] 控制."""
        import re
        texts = re.findall(r'\d+\. (.+)', user)
        items = ",".join(f'{{"text":"{t}","ok":true}}' for t in texts if t)
        calls["fluency"].append(1)
        return '{"items":[' + items + "]}"

    monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)
    monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
    return calls


@pytest.fixture
def ga_seed_pairs():
    """造假积木 (字映射 + 词映射 + 3字骨干)."""
    return [
        {"left": "义父", "right": "盗摄", "dir": "fwd", "score": 9.0},
        {"left": "粪厂", "right": "实境", "dir": "fwd", "score": 10.0},
        {"left": "通心", "right": "看奶", "dir": "fwd", "score": 8.0},
        {"left": "挺住", "right": "尿住", "dir": "fwd", "score": 8.0},
        {"left": "义父盗", "right": goose("义父盗"), "dir": "fwd", "score": 8.0},
    ]


class TestEvolve:
    def test_warm_start_populates_from_seeds(self, ga_seed_pairs):
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        rng = random.Random(1)
        chars = extract_char_maps(ga_seed_pairs)
        words = extract_word_maps(ga_seed_pairs)
        pop = evolve.warm_start(ga_seed_pairs, chars, words, 10, 1, rng)
        assert len(pop) >= 1
        # 骨干含 3 字+ 种子里的 left 串 (warm_start 只留 SEED_MIN_LEN=3 字以上骨干)
        pop_strs = {i.s for i in pop.to_list()}
        assert "义父盗" in pop_strs

    def test_evolve_returns_epoch_result_structure(self, ga_mock_llm, ga_seed_pairs):
        """evolve 一个 epoch: 返回 GAEpochResult, 结构完整."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        chars = extract_char_maps(ga_seed_pairs)
        words = extract_word_maps(ga_seed_pairs)
        rng = random.Random(42)
        result = evolve.evolve(
            client=None, seed_pairs=ga_seed_pairs, all_chars=chars, all_words=words,
            n_gen=20, rnd=1, logger=lambda s: None, debug_writer=None,
            model_loggers=None, existing_population=None, existing_epoch=0, rng=rng)
        # 结构 (单层全模型打分: 无 sentences/promoted, 有 findings/gold_hits/per_pair)
        assert result.gen_model == "seed_ga"
        assert isinstance(result.findings, list)
        assert isinstance(result.gold_hits, list)
        assert isinstance(result.per_pair, dict)
        assert isinstance(result.pop, list) and len(result.pop) > 0
        assert result.stats.get("offspring", 0) > 0
        assert result.stats.get("scored", 0) >= 0   # 打分条数
        # 种群个体有序列化字段
        for d in result.pop[:3]:
            assert "s" in d

    def test_evolve_scores_all_offspring(self, ga_mock_llm, ga_seed_pairs):
        """单层架构: 所有 offspring 都被全模型打分 (含'义'的得9, 进 gold)."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        chars = extract_char_maps(ga_seed_pairs)
        words = extract_word_maps(ga_seed_pairs)
        # 用固定种子 + 大 n_gen 增加含 '义' 候选出现概率
        rng = random.Random(7)
        result = evolve.evolve(
            client=None, seed_pairs=ga_seed_pairs, all_chars=chars, all_words=words,
            n_gen=60, rnd=1, logger=lambda s: None, debug_writer=None,
            model_loggers=None, existing_population=None, existing_epoch=0, rng=rng)
        # findings 非空 (全模型打了分), 含 '义' 的候选进了 gold (mock 给9分)
        assert len(result.findings) > 0
        if any("义" in f.S for f in result.findings):
            assert any("义" in gh["pair"].split("→")[0] for gh in result.gold_hits)
        # 全模型打分调过
        assert len(ga_mock_llm["score"]) >= 1

    def test_evolve_persists_population_for_resume(self, ga_mock_llm, ga_seed_pairs):
        """evolve 返回的 pop 可作为下轮 existing_population (断点续)."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        chars = extract_char_maps(ga_seed_pairs)
        words = extract_word_maps(ga_seed_pairs)
        rng1 = random.Random(1)
        r1 = evolve.evolve(None, ga_seed_pairs, chars, words, 20, 1,
                           lambda s: None, None, existing_population=None,
                           existing_epoch=0, rng=rng1)
        # 第二轮用第一轮的种群续传
        rng2 = random.Random(2)
        r2 = evolve.evolve(None, ga_seed_pairs, chars, words, 20, 2,
                           lambda s: None, None, existing_population=r1.pop,
                           existing_epoch=1, rng=rng2)
        assert len(r2.pop) > 0
        assert r2.stats.get("offspring", 0) > 0

    def test_seen_s_prevents_reevaluation(self, ga_mock_llm, ga_seed_pairs):
        """全局 seen_s: 已评过的 S 不再进全模型打分 (不重复烧 LLM)."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        chars = extract_char_maps(ga_seed_pairs)
        words = extract_word_maps(ga_seed_pairs)
        # 第一轮: 跑完, seen_s 累积了本 epoch 评过的所有 S
        seen_s = set()
        r1 = evolve.evolve(None, ga_seed_pairs, chars, words, 30, 1,
                           lambda s: None, None, existing_population=None,
                           existing_epoch=0, seen_s=seen_s, rng=random.Random(1))
        n_scored_epoch1 = len(seen_s)
        assert n_scored_epoch1 > 0   # 至少评过几个

        # 第二轮: 种群里所有 S 都已在 seen_s (它们上一轮被评过). 续传后这些骨干不再
        # 重评. 用同样的 seed 保证 offspring 部分撞上 seen_s.
        calls_before = len(ga_mock_llm["score"])
        r2 = evolve.evolve(None, ga_seed_pairs, chars, words, 30, 2,
                           lambda s: None, None, existing_population=r1.pop,
                           existing_epoch=1, seen_s=seen_s, rng=random.Random(1))
        # seen_s 持续增长 (第二轮新评的 S 也加进去)
        assert len(seen_s) >= n_scored_epoch1

    def test_evict_gold_removes_gold_individuals_from_population(self, ga_mock_llm, ga_seed_pairs):
        """全模型 max>=SCORE_THRESH(gold) 的个体被 evict 出种群: 既存档就不该再占.
        单层架构下所有 offspring 都被全模型打分, 但只有 gold(>=6)的才 evict."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        from goose_digging.mining.config import GA_EVICT_SCORE_THRESH
        chars = extract_char_maps(ga_seed_pairs)
        words = extract_word_maps(ga_seed_pairs)
        rng = random.Random(7)
        r = evolve.evolve(None, ga_seed_pairs, chars, words, 60, 1,
                          lambda s: None, None, existing_population=None,
                          existing_epoch=0, rng=rng)
        # gold 个体 (max(scores)>=GA_EVICT_SCORE_THRESH) 不应在最终种群里 (被 evict)
        gold_s = {gh["pair"].split("→")[0] for gh in r.gold_hits}
        if gold_s:
            pop_s = {d["s"] for d in r.pop}
            leftover = gold_s & pop_s
            assert not leftover, f"gold 个体未 evict: {leftover}"


# ---------------------------------------------------------------------------
# 5b. 近均匀 boost 采样覆盖 (低分积木也能采到, 保证完整覆盖)
# ---------------------------------------------------------------------------

class TestUniformBoostSampling:
    def test_sample_word_maps_covers_low_score(self):
        """近均匀采样: 低分词映射也能被采到 (不只采高分).
        动机: 低分词映射成句后可能反而高分, 需尽量一视同仁."""
        from goose_digging.mining.seed import sample_word_maps
        from goose_digging.mining.config import SEED_BOOST_ALPHA
        # 50 个词映射, 分数从 1 到 10 (高低分都有)
        wm = [(f"a{i}", f"b{i}", float((i % 10) + 1)) for i in range(50)]
        sampled_signals = set()
        for s in range(100):
            out = sample_word_maps(wm, 20, SEED_BOOST_ALPHA, random.Random(s))
            sampled_signals.update(w for _, _, w in out)
        # 100 次采样应覆盖各种分数段 (含低分 1-3)
        assert min(sampled_signals) <= 3, "低分词映射没被采到, 采样不够均匀"
        assert max(sampled_signals) >= 9, "高分词映射没被采到"

    def test_sample_char_maps_covers_low_freq(self):
        """近均匀采样: 低频字映射也能被采到."""
        from goose_digging.mining.seed import sample_char_maps
        from goose_digging.mining.config import SEED_BOOST_ALPHA
        # 字映射频次从 1 到 50
        cm = [((f"a{i}", f"b{i}"), i + 1) for i in range(50)]
        sampled_freqs = set()
        for s in range(100):
            out = sample_char_maps(cm, 20, SEED_BOOST_ALPHA, random.Random(s))
            # 反查频次
            fmap = {ab: f for ab, f in cm}
            sampled_freqs.update(fmap.get(ab, 0) for ab in out)
        assert min(sampled_freqs) <= 5, "低频字映射没被采到"
        assert max(sampled_freqs) >= 40, "高频字映射没被采到"

    def test_ga_config_params_widened(self):
        """GA 参数已调宽 (广泛覆盖、持久探索目标)."""
        from goose_digging.mining import config
        assert config.GA_POP_SIZE >= 60        # 种群更大
        assert config.GA_IMMIGRANT_RATE >= 0.40  # 更多移民
        assert config.GA_ELITE <= 1            # 精英更少 (防收敛)
        assert config.GA_MAX_AGE <= 5          # aging 更激进
        assert config.GA_SHARING_SIGMA >= 2.5  # sharing 邻域放宽
        # boost 采样参数存在且小 (近均匀)
        assert hasattr(config, "SEED_BOOST_ALPHA")
        assert 0 < config.SEED_BOOST_ALPHA <= 0.2
        assert hasattr(config, "GA_IMMIGRANT_BOOST_ALPHA")
        assert 0 < config.GA_IMMIGRANT_BOOST_ALPHA <= 0.2


# ---------------------------------------------------------------------------
# 6. T1 通顺预筛 (不通顺的杀掉再喂全模型, 省 token)
# ---------------------------------------------------------------------------

class TestT1FluencyPrefilter:
    def test_t1_kills_unfluent_both_sides_must_pass(self, monkeypatch):
        """T1 两边都通才进全模型: 左或右边不通的都杀.
        神鹅语必须两边都是人话形成反差 (吊挺吗→倪尿妈), 单边通的不算."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        from goose_digging.oracle import goose
        import re as _re
        calls = {"score_lefts": []}

        # fluency: 标含'垃圾'的不通, 其余通 (左右都判)
        def fake_fluency(client, model, system, user, **kw):
            texts = _re.findall(r'\d+\. (.+)', user)
            items = ",".join(f'{{"text":"{t}","ok":{str("垃圾" not in t).lower()}}}'
                             for t in texts if t)
            return '{"items":[' + items + "]}"

        def fake_score(client, model, system, user, **kw):
            lefts = _re.findall(r'\d+\. (.+?) →', user)
            calls["score_lefts"].extend(lefts)
            parts = [f'{{"left":"{L}","score":9,"why":"x"}}' for L in lefts]
            return '{"scores":[' + ",".join(parts) + "]}"

        monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
        monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)

        ga_seed = [{"left": "义父盗", "right": goose("义父盗"), "dir": "fwd", "score": 9.0}]
        chars = extract_char_maps(ga_seed)
        words = extract_word_maps(ga_seed)
        r = evolve.evolve(None, ga_seed, chars, words, 40, 1, lambda s: None, None,
                          existing_population=None, existing_epoch=0, rng=random.Random(1))
        # 被全模型打分的候选: 左和右都不含'垃圾'(两边都通才进来)
        scored = set(calls["score_lefts"])
        assert not any("垃圾" in s for s in scored)

    def test_t1_uses_lenient_ga_prompt_not_strict(self, monkeypatch):
        """GA 的 T1 用 GA_FLUENCY_SYSTEM (宽松), 不是枚举路径的严格 FLUENCY_SYSTEM.
        验证: evolve 调 fluency_filter 时传了宽松 prompt (放行谐音/联想/口语)."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        from goose_digging.oracle import goose
        from goose_digging.mining.prompts import GA_FLUENCY_SYSTEM, FLUENCY_SYSTEM
        import re as _re
        seen_systems = []

        def fake_fluency(client, model, system, user, **kw):
            seen_systems.append(system)
            texts = _re.findall(r'\d+\. (.+)', user)
            items = ",".join(f'{{"text":"{t}","ok":true}}' for t in texts if t)
            return '{"items":[' + items + "]}"

        def fake_score(client, model, system, user, **kw):
            lefts = _re.findall(r'\d+\. (.+?) →', user)
            parts = [f'{{"left":"{L}","score":4,"why":"x"}}' for L in lefts]
            return '{"scores":[' + ",".join(parts) + "]}"

        monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
        monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)

        ga_seed = [{"left": "义父盗", "right": goose("义父盗"), "dir": "fwd", "score": 9.0}]
        chars = extract_char_maps(ga_seed)
        words = extract_word_maps(ga_seed)
        evolve.evolve(None, ga_seed, chars, words, 20, 1, lambda s: None, None,
                      existing_population=None, existing_epoch=0, rng=random.Random(1))
        # T1 用的是宽松 GA prompt, 不是严格的枚举 prompt
        assert any(GA_FLUENCY_SYSTEM in s for s in seen_systems), \
            "GA 应传 GA_FLUENCY_SYSTEM"
        assert GA_FLUENCY_SYSTEM != FLUENCY_SYSTEM, "两个 prompt 不能一样(否则宽松失效)"

    def test_t1_stats_reports_killed(self, monkeypatch):
        """stats 里有 t1_killed 字段."""
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        from goose_digging.oracle import goose
        import re as _re

        def fake_fluency(client, model, system, user, **kw):
            texts = _re.findall(r'\d+\. (.+)', user)
            items = ",".join(f'{{"text":"{t}","ok":true}}' for t in texts if t)
            return '{"items":[' + items + "]}"

        def fake_score(client, model, system, user, **kw):
            lefts = _re.findall(r'\d+\. (.+?) →', user)
            parts = [f'{{"left":"{L}","score":4,"why":"x"}}' for L in lefts]
            return '{"scores":[' + ",".join(parts) + "]}"

        monkeypatch.setattr("goose_digging.mining.fluency.llm_chat", fake_fluency)
        monkeypatch.setattr("goose_digging.mining.scoring.llm_chat", fake_score)

        ga_seed = [{"left": "义父盗", "right": goose("义父盗"), "dir": "fwd", "score": 9.0}]
        chars = extract_char_maps(ga_seed)
        words = extract_word_maps(ga_seed)
        r = evolve.evolve(None, ga_seed, chars, words, 20, 1, lambda s: None, None,
                          existing_population=None, existing_epoch=0, rng=random.Random(1))
        assert "t1_killed" in r.stats

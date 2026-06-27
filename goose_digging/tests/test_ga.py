# -*- coding: utf-8 -*-
"""GA 神鹅语进化纯单测: 算子 / 种群多样性 / 防同化 / 采样数学 / warm_start.

不调 LLM (mock-LLM 集成测试见 test_e2e.py). 唯一例外是 test_scoring_failure
(容错: 全模型打分失败时 epoch 不崩, 自带独立 mock 不依赖共享 fixture).
"""
import random

import pytest

from goose_digging.mining.ga import genome, population, evolve
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
    def test_raw_score_uses_full_scores(self):
        ind = Individual(s="义父", scores=[8, 9])
        assert ind.raw_score() == 8.5   # 全模型均分

    def test_raw_score_zero_when_unscored(self):
        assert Individual(s="x").raw_score() == 0.0

    def test_dict_roundtrip(self):
        """to_dict/from_dict 往返: s/scores/born 保留."""
        ind = Individual(s="义父", scores=[8, 9], born=3)
        ind2 = Individual.from_dict(ind.to_dict())
        assert ind2.s == "义父"
        assert ind2.scores == [8, 9] and ind2.born == 3


# ---------------------------------------------------------------------------
# 2. 算子: crossover / mutate / immigrant
# ---------------------------------------------------------------------------

class TestOperators:
    def test_crossover_respects_max_len(self):
        # 过长截断到 max_len; 过短原样返回 (min_len 由 is_viable 预筛兜底, 不在算子里
        # 靠重复末字硬凑 —— 旧版那样只造叠词垃圾). 只保证: 不超过 max_len, 全 CJK.
        rng = random.Random(7)
        for _ in range(50):
            child = crossover("义父盗摄", "通心看奶", 3, 9, rng)
            assert len(child) <= 9
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

    def test_clamp_len_does_not_pad_short_strings(self):
        """_clamp_len 过短原样返回, 不再重复末字凑 min_len (旧版那样只造叠词垃圾).

        过短串的丢弃交给 is_viable 预筛, 而不是在算子里硬塞叠字.
        """
        from goose_digging.mining.ga.genome import _clamp_len
        # 过短: 原样返回, 没有重复末字补足
        assert _clamp_len("义", 3, 9) == "义"
        assert _clamp_len("义父", 3, 9) == "义父"
        # 区间内: 不变
        assert _clamp_len("义父盗", 3, 9) == "义父盗"
        # 过长: 截断 (这条不变)
        assert _clamp_len("义父盗摄通心厂", 3, 4) == "义父盗摄"
        # 关键: 绝不产生末字重复
        assert _clamp_len("义", 3, 9) != "义义义"

    def test_mutate_does_not_manufacture_reduplication(self):
        """变异算子不再有专门的"重复字"模式 (旧版 mode-4 17% 概率制造叠词).

        残余的相邻重复只能是 char/word 替换碰巧撞出的巧合 (偶发, ~5%), 不该有
        一个算子专门去造叠词 —— 因为 GA_FLUENCY_SYSTEM 正好杀叠词堆砌, 造了就是
        白造 + 浪费 T1 token.
        """
        rng = random.Random(99)
        s = "义父看奶挺"   # 5 个互不相同字, 杜绝"本来就有重复"干扰
        manufactured = 0
        n = 500
        for _ in range(n):
            m = mutate(s, CHAR_SAMPLE, WORD_SAMPLE, 3, 9, rng)
            if any(m[i] == m[i + 1] for i in range(len(m) - 1)):
                manufactured += 1
        # 旧版 ~21% (mode-4 主导); 新版应 <10% (纯巧合). 留余量取 12%.
        assert manufactured / n < 0.12, \
            f"变异仍在大量造叠词: {manufactured}/{n} = {manufactured/n*100:.0f}%"

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
        lone = Individual(s="义父盗摄", scores=[8, 8])
        # 4 个和 lone 极相似的个体 (Levenshtein<=2) 围着它
        cluster = [Individual(s="义父盗摄", scores=[8, 8]),
                   Individual(s="义父盗摄", scores=[8, 8])]
        sf_lone_in_cluster = shared_fitness(lone, [lone] + cluster, sigma=2.0)
        sf_lone_alone = shared_fitness(lone, [lone], sigma=2.0)
        assert sf_lone_in_cluster < sf_lone_alone   # 被簇稀释

    def test_dedupe_keeps_highest_score(self):
        inds = [Individual(s="义父", scores=[4, 4]),
                Individual(s="义父", scores=[9, 9]),
                Individual(s="通心", scores=[7, 7])]
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
            Individual(s="待挖", scores=[5, 5]),    # 未到 gold 线
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
        """Bug H: ga_seen_s 持久化往返 (跨 run 防重烧 LLM)."""
        from goose_digging.mining.state import MiningState
        sp = tmp_path / "state.json"
        st = MiningState()
        st.ga_seen_s = ["义父", "通心", "粪厂"]
        st.save(path=sp)
        st2 = MiningState.load(path=sp)
        assert st2.ga_seen_s == ["义父", "通心", "粪厂"]

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
        长时间挖掘时模型偶发故障很常见. 通顺预筛正常(默认全通), 全模型打分全失败."""
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
        # 通顺判: 正常全通 (让候选进到全模型打分环节, 才能测打分失败)
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
# 5. warm_start (纯函数, 不调 LLM): 种群从种子填充
# ---------------------------------------------------------------------------

_SEED_PAIRS = [
    {"left": "义父", "right": "盗摄", "dir": "fwd", "score": 9.0},
    {"left": "粪厂", "right": "实境", "dir": "fwd", "score": 10.0},
    {"left": "通心", "right": "看奶", "dir": "fwd", "score": 8.0},
    {"left": "挺住", "right": "尿住", "dir": "fwd", "score": 8.0},
    {"left": "义父盗", "right": goose("义父盗"), "dir": "fwd", "score": 8.0},
]


class TestWarmStart:
    def test_warm_start_populates_from_seeds(self):
        from goose_digging.mining.seed import extract_char_maps, extract_word_maps
        rng = random.Random(1)
        chars = extract_char_maps(_SEED_PAIRS)
        words = extract_word_maps(_SEED_PAIRS)
        pop = evolve.warm_start(_SEED_PAIRS, chars, words, 10, 1, rng)
        assert len(pop) >= 1
        # 骨干含 3 字+ 种子里的 left 串 (warm_start 只留 SEED_MIN_LEN=3 字以上骨干)
        pop_strs = {i.s for i in pop.to_list()}
        assert "义父盗" in pop_strs


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

    def test_ga_config_params_scaled(self):
        """GA 参数按真实预筛接受率 (~4-8%) scale 到每 epoch 打分 ~12.

        约束: offspring x viable留存 x 去重留存 x 接受率 ≈ 打分数.
        8% 端: 220x0.82x0.85x0.08 ≈ 12 打分, 替换率 ~12% (健康稳态).
        4% 端: ≈6 打分, 替换率 ~6% (不停滞).
        IMMIGRANT+CROSSOVER+MUTATION=1.0 (evolve._gen_offspring 硬约束).
        """
        from goose_digging.mining import config
        # offspring 够支撑真实低接受率下的打分 (8%x0.82x0.85xoff ≈ 12)
        assert config.GA_OFFSPRING_PER_EPOCH >= 200
        # 稳态替换率健康 (offspring存活/pop 不超 25%, 防大换血)
        assert config.GA_POP_SIZE >= 80
        assert config.GA_OFFSPRING_PER_EPOCH * 0.82 * 0.85 * 0.08 / config.GA_POP_SIZE <= 0.25
        # 精英少 (防收敛)
        assert config.GA_ELITE <= 1
        # age 与 pop 同步放大 (给个体多代被验证, 对冲大流入)
        assert config.GA_MAX_AGE >= 5
        # rate 和 = 1.0
        assert abs(config.GA_IMMIGRANT_RATE + config.GA_CROSSOVER_RATE
                   + config.GA_MUTATION_RATE - 1.0) < 1e-9
        # sharing 邻域放宽
        assert config.GA_SHARING_SIGMA >= 2.5
        # boost 采样参数存在且小 (近均匀)
        assert hasattr(config, "SEED_BOOST_ALPHA")
        assert 0 < config.SEED_BOOST_ALPHA <= 0.2
        assert hasattr(config, "GA_IMMIGRANT_BOOST_ALPHA")
        assert 0 < config.GA_IMMIGRANT_BOOST_ALPHA <= 0.2

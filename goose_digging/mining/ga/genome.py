# -*- coding: utf-8 -*-
"""基因型 + 遗传算子.

基因 = 一条候选中文串 S (3-9 字). 直接以 S 为基因, 不引入额外 genotype/phenotype
映射, 保证 goose(S) 可由 oracle 确定性算出 (mining/ga 落盘前仍由 pairs_from_sentences
校验 goose(S)!=S, 这里只造不校验).

每个个体 (Individual) 缓存:
  s              基因串
  length         字数
  scores         全模型分列表 (fitness = 均分; 未评时 None)
  born           出生 epoch 号
  surrogate      DEPRECATED. 旧版单模型适应度 (已废弃改全模型直接打分).
                 仅保留字段以兼容旧 state.json 续跑 (from_dict 读旧盘不崩);
                 生产代码不再写入. raw_score() 仅在 scores 缺失时回退用它.

算子全部从现有"积木"构造 (不引入新数据源):
  crossover      段拼接: 父A前缀 + 父B后缀, 裁剪到 [MIN_LEN, MAX_LEN]
  mutation       按概率选一: 字映射替换 / 词积木插值 / 删字 / 重复字
  immigrant      近均匀采样从零拼一条全新 S (探索源, 对应用户说的"随机字映射采样"基因)

所有算子接受外部传入的 rng (random.Random), 保证跨 epoch 可复现/断点续一致.
长度约束 GA_MIN_LEN/GA_MAX_LEN 来自 config (复用 SEED_MIN/MAX_LEN).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from goose_digging.oracle import goose, goose_char


@dataclass
class Individual:
    """一条候选神鹅语基因串."""
    s: str
    scores: list[int] | None = None   # 全模型分 (fitness=均分; 未评时 None)
    born: int = 0                     # 出生 epoch
    surrogate: float | None = None    # DEPRECATED 旧单模型分 (仅兼容旧 state.json 读盘)

    @property
    def length(self) -> int:
        return len(self.s)

    def to_dict(self) -> dict:
        """序列化 (落 state.json 断点续). surrogate 已废弃, 不再写入."""
        return {"s": self.s, "scores": self.scores, "born": self.born}

    @classmethod
    def from_dict(cls, d: dict) -> "Individual":
        # 兼容旧 state.json: 旧盘可能含 surrogate 字段, 读进来不崩 (字段保留但不再使用).
        return cls(s=str(d.get("s", "")),
                   scores=d.get("scores"),
                   born=int(d.get("born", 0)),
                   surrogate=d.get("surrogate"))

    def raw_score(self) -> float:
        """当前最佳已知分: 优先全模型均分; scores 缺失时回退 surrogate (兼容旧盘); 都没有返 0.

        供排序/锦标赛/替换比较用 (不掺多样性调整, 那是 shared_fitness 的事).
        """
        if self.scores:
            return sum(self.scores) / len(self.scores)
        if self.surrogate is not None:
            return self.surrogate   # 仅旧 state.json 续跑时可能命中
        return 0.0


# ---------------------------------------------------------------------------
# 长度裁剪 (所有算子共用)
# ---------------------------------------------------------------------------

def _clamp_len(s: str, min_len: int, max_len: int) -> str:
    """裁剪到 [min_len, max_len]: 过长截断, 过短重复末字补足 (叠词也是合法神鹅语)."""
    if len(s) > max_len:
        return s[:max_len]
    while len(s) < min_len and s:
        s += s[-1]   # 末字重复补足 (产生叠词, 呼应 gold_long 里高频叠词不动点)
    return s


def _all_cjk(s: str) -> bool:
    return all("\u4e00" <= c <= "\u9fff" for c in s)


# ---------------------------------------------------------------------------
# crossover: 段拼接 (父A前缀 + 父B后缀)
# ---------------------------------------------------------------------------

def crossover(parent_a: str, parent_b: str, min_len: int, max_len: int,
              rng: random.Random) -> str:
    """段拼接: 取 A 的前缀 + B 的后缀, 切点随机, 裁剪到合法长度.

    这是"已有鹅语碎片"的拼接 —— 把两个验证过的短句搭成长句骨干.
    退化情形 (A/B 太短) 直接随机选一个父裁剪.
    """
    a = parent_a
    b = parent_b
    if not a or not b:
        base = a or b
        return _clamp_len(base, min_len, max_len) if base else ""
    cut = rng.randint(1, max(1, len(a)))   # A 前缀长度 [1, len(a)]
    tail = rng.randint(0, max(0, len(b) - 1))  # B 后缀起点 [0, len(b)-1] (留至少1字)
    child = a[:cut] + b[tail:]
    if not _all_cjk(child):
        # 含非 CJK (理论不会, 兜底): 退回纯父裁剪
        child = a
    return _clamp_len(child, min_len, max_len)


# ---------------------------------------------------------------------------
# mutation: 字映射替换 / 词积木插值 / 删字 / 重复字
# ---------------------------------------------------------------------------

def mutate(s: str, char_sample: list[tuple[str, str]],
           word_sample: list[tuple], min_len: int, max_len: int,
           rng: random.Random) -> str:
    """变异: 随机选一种, 直接驱动 goose(S) 变化.

    char_sample: 近均匀采样后的字映射 [(a,b=goose(a)), ...] (extract_char_maps 产物).
    word_sample: 近均匀采样后的词映射 [(s,r,score), ...] (extract_word_maps 产物).
    """
    if not s:
        return s
    mode = rng.random()
    if mode < 0.40 and char_sample:
        # 1) 字映射替换: 把 S 里某字 a 换成"能映射到目标 b 的另一个源字",
        #    直接改变 goose(S) 该位输出. 用采样的 (a,b): 若 S 含 a, 替换成 b
        #    (b 是 goose(a), 但 b 本身也可能是某字的源, goose(b) 会变, 制造新变换).
        a, b = rng.choice(char_sample)
        if a in s:
            s = s.replace(a, b, 1)
        else:
            # S 不含 a: 把随机一位换成 b (注入新字)
            i = rng.randrange(len(s))
            s = s[:i] + b + s[i + 1:]
    elif mode < 0.65 and word_sample:
        # 2) 词积木插值: 从词映射取一块, 替换 S 中等长的一段
        w = rng.choice(word_sample)
        chunk = w[0]  # 词映射的 left (2字)
        if len(s) >= len(chunk) and len(chunk) >= 1:
            i = rng.randint(0, len(s) - len(chunk))
            s = s[:i] + chunk + s[i + len(chunk):]
        else:
            # S 比块短: 插入到随机位
            i = rng.randint(0, len(s))
            s = s[:i] + chunk + s[i:]
    elif mode < 0.83 and len(s) > min_len:
        # 3) 删字 (产生更紧凑的短句)
        i = rng.randrange(len(s))
        s = s[:i] + s[i + 1:]
    else:
        # 4) 重复字 (产生叠词/不动点, 呼应 gold_long 高频的不动点神鹅语如 脱内裤→脱内裤)
        i = rng.randrange(len(s))
        s = s[:i] + s[i] + s[i:]
    return _clamp_len(s, min_len, max_len)


# ---------------------------------------------------------------------------
# immigrant: 近均匀采样从零拼一条全新 S (探索源)
# ---------------------------------------------------------------------------

def immigrant(char_sample: list[tuple[str, str]],
              word_sample: list[tuple], target_len: int,
              rng: random.Random) -> str:
    """近均匀采样积木从零拼一条全新 S.

    对应用户定义的基因: "随机字映射采样". 混搭字映射源字 + 词映射积木,
    拼到约 target_len 字. 采样本身由调用方用 sample_char_maps/sample_word_maps
    (近均匀 boost) 完成, 这里只做拼接.
    """
    pieces: list[str] = []
    # 先铺一些词映射积木 (每个 2 字), 再用字映射源字补到 target_len
    while sum(len(p) for p in pieces) < target_len and word_sample:
        pieces.append(rng.choice(word_sample)[0])  # 词积木 left
        if rng.random() < 0.5:
            break  # 一半概率只用 1 个词积木 + 字填充, 增加多样性
    chars = [a for a, _ in char_sample] or list("你我他大小中上")
    while sum(len(p) for p in pieces) < target_len:
        pieces.append(rng.choice(chars))
    rng.shuffle(pieces)
    s = "".join(pieces)
    return _clamp_len(s, target_len, target_len) if len(s) >= target_len else s


# ---------------------------------------------------------------------------
# 廉价预筛 (零 LLM): 砍明显垃圾, 省全模型打分 token
# ---------------------------------------------------------------------------

def is_viable(s: str, min_len: int, max_len: int,
              seen_pairs: set | None = None,
              zipf_thresh: float = 2.0) -> bool:
    """零成本预筛: 砍明显的垃圾 offspring, 不浪费全模型 score_pairs 调用.

    判据 (全部纯计算):
      - 长度合法 [min_len, max_len]
      - 全 CJK
      - goose(S) != S (有变换价值, 非全不动点) —— 但允许部分不动点 (脱内裤式)
      - 不在 seen_pairs (已评过)
      - 每字 zipf >= zipf_thresh (复用 readability 思路, 砍生僻字乱码)

    注: 全不动点 (goose(S)==S) 在 gold_long 里虽高频, 但那类是词典命中枚举路径
    自然覆盖的; GA 聚焦"有变换的长句", 故预筛砍掉全不动点. 单字不动点无妨.
    """
    if not (min_len <= len(s) <= max_len):
        return False
    if not _all_cjk(s):
        return False
    if goose(s) == s:
        return False
    if seen_pairs is not None and (s, goose(s)) in seen_pairs:
        return False
    # 字级 zipf 预筛 (砍含生僻字的乱码拼凑, 复用 enumerate 思路)
    try:
        from ..readability import char_zipf
        if not all(char_zipf(c) >= zipf_thresh for c in s):
            return False
        if not all(char_zipf(c) >= zipf_thresh for c in goose(s)):
            return False
    except Exception:  # noqa: BLE001  wordfreq 缺失时不卡 (兜底, 不阻断)
        pass
    return True

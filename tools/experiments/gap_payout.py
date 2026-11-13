"""割安ワイド: ワイドの払戻の見込みを「下限〜上限の真ん中」から「下限 + α×幅」に直すと選び方が良くなるか(中央 2022〜2026年・確定オッズ)
  OUTDIR=/path/to/dir python tools/experiments/gap_payout.py     (gap_model.py が作った pairs.pkl を使う)
  ワイドは「下限〜上限」で出て、当たったときの払戻は3頭目が人気馬なら下限寄りになる。今の割安度は真ん中の値で計算している。
  1) 当たった組の払戻が、下限〜上限のどこに来るか(α。年ごと・オッズの帯ごと)
  2) 期待値 = 3連複の見込み(0.75×組の強さ)×(下限 + α×幅)。α と買う下限を変えた表(1レースで期待値がいちばん高い組だけ)
  3) 今の選び方と新しい選び方の差が、たまたまでどこまで動くか(レースを選び直す)"""
import os, pickle, sys
from collections import defaultdict
import numpy as np
OUT = os.environ.get("OUTDIR", "/tmp/gap_model")
D = pickle.load(open(os.path.join(OUT, "pairs.pkl"), "rb"))
YS = ("2022", "2023", "2024", "2025", "2026")
lo, hi, pt, pay, y, Y, race = D["lo"], D["hi"], D["pt"], D["pay"].astype(float), D["hit"], D["y"], D["race"]
mid = (lo + hi) / 2
N = len(y)

print("■ 1) 当たった組の払戻は、下限〜上限のどこに来るか(0 = 下限、1 = 上限)")
h = (y > 0) & (hi > lo)
pos = (pay / 100 - lo) / np.where(hi > lo, hi - lo, 1)
print("  全部 {:.3f}  年ごと [{}]".format(pos[h].mean(), " ".join("{:.2f}".format(pos[h & (Y == t)].mean()) for t in YS)))
for a, b in ((1.3, 2), (2, 3), (3, 5), (5, 10), (10, 30)):
    m = h & (lo >= a) & (lo < b)
    print("  ワイド下限 {:>4}〜{:<3}倍 {:.3f}({}点)".format(a, b, pos[m].mean(), m.sum()))

band = (mid >= 1.5) & (mid < 15)
rid, inv = np.unique(race, return_inverse=True)


def best_of(ev):
    """1レースで ev がいちばん高い組(ワイドの真ん中 1.5〜15倍の中で)の位置"""
    e = np.where(band, ev, -1.0)
    order = np.lexsort((e, inv))
    last = np.r_[np.where(np.diff(inv[order]) != 0)[0], len(order) - 1]
    ix = order[last]
    return ix[e[ix] > 0]


def stat(ix):
    v = pay[ix]
    return len(v), (v.mean() if len(v) else 0), [pay[ix[Y[ix] == t]].mean() if (Y[ix] == t).any() else 0 for t in YS]


print("\n■ 2) 期待値 = 0.75×組の強さ ×(下限 + α×幅)。1レースで期待値がいちばん高い組が基準以上なら買う  回収率(レース数) [年ごと]")
for alpha in (0.0, 0.2, 0.35, 0.5, 0.65):
    ev = pt * (lo + alpha * (hi - lo))
    bi = best_of(ev)
    print("  α {:.2f}{}".format(alpha, "(今の計算と同じ。真ん中)" if alpha == 0.5 else "(確かめた値)" if alpha == 0.35 else ""))
    for th in (0.95, 1.0, 1.05, 1.1):
        n, r, ys = stat(bi[ev[bi] >= th])
        print("    期待値 {:.2f} 以上  {:5.1f}%({:>4}R) [{}]".format(th, r, n, " ".join("{:4.0f}".format(v) for v in ys)))

print("\n■ 3) 今の選び方(α 0.5・期待値 1.0075 以上 = 割安度 1.30 以上)と、新しい選び方(α 0.35・期待値 1.00 以上)")
ev_old, ev_new = pt * mid, pt * (lo + 0.35 * (hi - lo))
bo = best_of(ev_old); bo = bo[ev_old[bo] >= 0.775 * 1.30]
bn = best_of(ev_new); bn = bn[ev_new[bn] >= 1.0]
so, sn = set(bo.tolist()), set(bn.tolist())
ro, rn = set(race[bo].tolist()), set(race[bn].tolist())
for nm, ix in (("今", bo), ("新しい", bn), ("両方で同じ組", np.array(sorted(so & sn))), ("今だけ買う組", np.array(sorted(so - sn))), ("新しい方だけ買う組", np.array(sorted(sn - so)))):
    n, r, ys = stat(ix)
    print("  {:12s} {:5.1f}%({:>4}R) [{}] 的中 {:.1%} 平均オッズ(真ん中) {:.1f}".format(nm, r, n, " ".join("{:4.0f}".format(v) for v in ys), (pay[ix] > 0).mean(), mid[ix].mean()))
rng = np.random.default_rng(0)
allr = np.array(sorted(ro | rn)); po = {race[i]: pay[i] for i in bo}; pn = {race[i]: pay[i] for i in bn}
do, dn = [], []
for _ in range(4000):
    s = rng.choice(allr, len(allr))
    a = [po[r] for r in s if r in po]; b = [pn[r] for r in s if r in pn]
    do.append(np.mean(a)); dn.append(np.mean(b))
do, dn = np.array(do), np.array(dn)
print("  今 95%の幅 {:.0f}〜{:.0f}%(100%超の見込み {:.0%}) / 新しい {:.0f}〜{:.0f}%({:.0%}) / 差(新しい−今) {:+.1f}〜{:+.1f} ポイント・新しい方が上の見込み {:.0%}".format(
    np.percentile(do, 2.5), np.percentile(do, 97.5), (do > 100).mean(), np.percentile(dn, 2.5), np.percentile(dn, 97.5), (dn > 100).mean(),
    np.percentile(dn - do, 2.5), np.percentile(dn - do, 97.5), (dn > do).mean()))
hy = defaultdict(list)
for i in bn:
    hy[Y[i]].append(pay[i])
print("  新しい選び方の年ごとのレース数: " + " ".join("{} {}R".format(t, len(hy[t])) for t in YS))

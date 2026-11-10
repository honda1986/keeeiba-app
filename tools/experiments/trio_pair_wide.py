"""3連複の市場が見ている「2頭とも3着内」の強さと、ワイドのオッズのずれで、ワイドを買う(中央。確定オッズ)
  python tools/experiments/trio_pair_wide.py [pre]
  trio_pair_final(その2頭を含む3連複の組の 1/オッズ の合計)× 0.75 = 3連複の市場の見込み、0.775 ÷ ワイドのオッズ = ワイドの市場の見込み。
  trio_vs_wide.py では ◎-○-x の組だけで、3連複の方が正確で、ずれた組のワイドが良かった。ここでは全部の組で確かめる。
  1) 見込みの比(3連複 ÷ ワイド)の帯 × ワイドのオッズの帯ごとの、的中と回収率(年ごと)
  2) 当たる確率を2つの見込みのロジスティック回帰で出し(ほかの年で係数を決める)、期待値(確率 × ワイドのオッズ)の帯ごとの回収率
  3) 1レースで期待値がいちばん高い組だけを買う
  引数に pre を付けると、発走前のオッズ(trio_pair と wide_odds。当日に取った分だけ)で同じことをする(払戻は確定)"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import keiba_final as F
R = os.path.join(HERE, "..", "..")
PRE = len(sys.argv) > 1 and sys.argv[1] == "pre"
KT, KW = ("trio_pair", "wide_odds") if PRE else ("trio_pair_final", "wide_final")
mid = lambda v: (v[0] + v[1]) / 2 if isinstance(v, (list, tuple)) else v
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
rows = []        # (年, レース, 組, 3連複の見込み, ワイドの見込み, ワイドのオッズ, 当たり, 払戻)
for f in sorted(glob.glob(R + "/keiba/days/20*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        tp, wf, res = r.get(KT), r.get(KW), r.get("result") or {}
        if r.get("center") != "jra" or not tp or not wf or len(res.get("order") or []) < 3:
            continue
        top3 = set(res["order"][:3]); pay = (res.get("payout") or {}).get("ワイド") or {}
        for k, v in wf.items():
            w = mid(v)
            if not w or w <= 1.0 or k not in tp:
                continue
            a, b = (int(x) for x in k.split("-"))
            rows.append((d[:4], rid, k, min(.75 * tp[k], .98), min(.775 / w, .98), w, int(a in top3 and b in top3), pay.get(k, 0)))
YS = sorted({r[0] for r in rows})
Y = np.array([r[0] for r in rows]); rid = np.array([r[1] for r in rows]); pt = np.array([r[3] for r in rows]); pw = np.array([r[4] for r in rows])
W = np.array([r[5] for r in rows]); hit = np.array([r[6] for r in rows], dtype=float); pay = np.array([r[7] for r in rows], dtype=float)
print("{}のオッズ: {}レース・ワイドの組 {}点・当たり {}  年 {}".format("発走前" if PRE else "確定", len(set(rid)), len(rows), int(hit.sum()), YS))


def line(name, m):
    print("  {:36s} {:7d}点 {:4.0f}% [{}] 的中 {:5.1%}".format(name, int(m.sum()), pay[m].mean() if m.any() else 0, " ".join(
        "{:4.0f}".format(pay[m & (Y == y)].mean()) if (m & (Y == y)).any() else "   -" for y in YS), hit[m].mean() if m.any() else 0))


ratio = np.log(pt / pw)
print("\n■ 見込みの比(3連複 ÷ ワイド)の帯 × ワイドのオッズの帯  回収率 [{}]".format(" ".join(YS)))
for lo, hi in ((1, 2), (2, 3), (3, 6), (6, 12), (12, 30), (30, 100)):
    m0 = (W >= lo) & (W < hi)
    if m0.sum() < 200:
        continue
    qs = np.quantile(ratio[m0], [0, .05, .2, .5, .8, .95, 1])
    print("  ワイド {}〜{}倍".format(lo, hi))
    for a_, b_ in zip(qs[:-1], qs[1:]):
        line("    比 {:.2f}〜{:.2f}".format(np.exp(a_), np.exp(b_)), m0 & (ratio >= a_) & (ratio <= b_))

print("\n■ 当たる確率 = 3連複の見込みとワイドの見込みのロジスティック回帰(確かめる年を除いて係数を決める)")
p = np.zeros(len(rows))
X = np.column_stack([lg(pt), lg(pw)])
for t in YS:
    tr, te = (Y != t) if len(YS) > 1 else (Y == t), Y == t
    mu, sd = X[tr].mean(0), X[tr].std(0)
    w = np.array(F.fit_logit(((X[tr] - mu) / sd).tolist(), hit[tr].tolist(), 0.01))
    p[te] = 1 / (1 + np.exp(-(np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ w)))
    ll = lambda q: float(np.mean(hit[te] * np.log(np.clip(q, 1e-6, 1)) + (1 - hit[te]) * np.log(np.clip(1 - q, 1e-6, 1))))
    print("  {}年 係数 3連複 {:+.2f}・ワイド {:+.2f}  対数尤度 回帰 {:.5f} / 3連複だけ {:.5f} / ワイドだけ {:.5f}".format(
        t, w[1] / sd[0], w[2] / sd[1], ll(p[te]), ll(pt[te]), ll(pw[te])))
ev = p * W
print("\n■ 期待値(確率 × ワイドのオッズ)の帯ごと")
for a_, b_ in ((0, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 1.4), (1.4, 99)):
    line("期待値 {:.1f}〜{:.1f}".format(a_, b_), (ev >= a_) & (ev < b_))
print("\n■ 期待値 1.0 以上を、ワイドのオッズの帯で")
for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 30), (30, 9999)):
    line("ワイド {}〜{}倍".format(lo, hi), (ev >= 1.0) & (W >= lo) & (W < hi))
print("\n■ 1レースで期待値がいちばん高い組だけ(ワイド30倍未満の中で)")
best = np.zeros(len(rows), dtype=bool)
by = defaultdict(list)
for i in range(len(rows)):
    if W[i] < 30:
        by[rid[i]].append(i)
for ix in by.values():
    best[ix[int(np.argmax(ev[ix]))]] = True
for a_ in (0, .9, 1.0, 1.1, 1.2):
    line("いちばん高い組の期待値が {:.1f} 以上".format(a_), best & (ev >= a_))

"""3連複のオッズは、ワイドのオッズと食い違っていないか(中央 2024〜2026年。◎-○-x の3連複の確定オッズ trio_final があるレース)
  python tools/experiments/trio_vs_wide.py
  beyond_market.py で、3着内の当たり方はワイドのオッズがいちばん正確だった。3連複の値段がワイドの見立てとずれているなら、
  ずれた組を買えば得になるはず。
  1) 「◎と○が両方3着内」を、ワイド◎-○ で買うのと、3連複◎-○-全部を払戻が同じになるように配って買うのと、どちらが高いか
  2) 3連複◎-○-x が当たる確率を、ワイドの3つの組(◎○・◎x・○x)のオッズから出し(ほかの年で係数を決める)、
     期待値(確率×3連複のオッズ)の帯ごとに、確かめる年の回収率を見る"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import keiba_final as F
R = os.path.join(HERE, "..", "..")
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))
mid = lambda v: (v[0] + v[1]) / 2 if isinstance(v, (list, tuple)) else v
races = []
RACE = {}
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        tf, wf, res = r.get("trio_final"), r.get("wide_final") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not tf or len(res.get("order") or []) < 3:
            continue
        ks = [set(int(x) for x in k.split("-")) for k in tf]
        pair = set.intersection(*ks)
        if len(pair) != 2:
            continue
        a, b = sorted(pair)
        wab = mid(wf.get(srt((a, b)), 0))
        if not wab or wab <= 1.0:
            continue
        top3 = set(res["order"][:3]); pay = res.get("payout") or {}
        xs = []
        for k, o in tf.items():
            x = next(int(v) for v in k.split("-") if int(v) not in pair)
            wax, wbx = mid(wf.get(srt((a, x)), 0)), mid(wf.get(srt((b, x)), 0))
            if o and o > 1.0:
                xs.append(dict(x=x, o=o, wax=wax, wbx=wbx, hit=int({a, b, x} <= top3), pay=(pay.get("3連複") or {}).get(k, 0)))
        if len(xs) < 3:
            continue
        RACE[rid] = dict(order=res["order"], pay=pay)
        races.append(dict(y=d[:4], rid=rid, a=a, b=b, wab=wab, xs=xs, both=int({a, b} <= top3),
                          wpay=(pay.get("ワイド") or {}).get(srt((a, b)), 0), n=len(xs) + 2))
print("3連複とワイドの確定オッズがそろうレース {}R".format(len(races)), {y: sum(1 for r in races if r["y"] == y) for y in YS})


def line(name, rows):
    by = defaultdict(list)
    for y, p in rows:
        by[y].append(p)
    v = [p for _, p in rows]
    print("  {:40s} {:6d}点 {:4.0f}% [{}] 的中 {:5.1%}".format(name, len(v), np.mean(v) if v else 0, " ".join(
        "{:4.0f}".format(np.mean(by[y])) if by[y] else "   -" for y in YS), np.mean([p > 0 for p in v]) if v else 0))


print("\n■ 「◎と○が両方3着内」をどちらの馬券で買うか(100円あたりの払戻)")
D = [1 / sum(1 / x["o"] for x in r["xs"]) for r in races]      # 3連複を配って買ったときの倍率
ratio = np.array([d / r["wab"] for d, r in zip(D, races)])
print("  3連複を配った倍率 ÷ ワイドの倍率: 中央値 {:.2f}・下から10% {:.2f}・上から10% {:.2f}(控除の差だけなら 0.97)".format(
    np.median(ratio), np.quantile(ratio, .1), np.quantile(ratio, .9)))
line("ワイド◎-○(確定の払戻)", [(r["y"], r["wpay"]) for r in races])
line("3連複◎-○-全部を配って買う", [(r["y"], r["both"] * d * 100) for d, r in zip(D, races)])
line("高い方で買う", [(r["y"], max(r["wpay"], r["both"] * d * 100)) for d, r in zip(D, races)])
for a_, b_ in ((0, .9), (.9, 1.0), (1.0, 1.1), (1.1, 9)):
    sel = [(d, r) for d, r, q in zip(D, races, ratio) if a_ <= q < b_]
    line("  比 {:.1f}〜{:.1f} のレース ワイド".format(a_, b_), [(r["y"], r["wpay"]) for d, r in sel])
    line("  比 {:.1f}〜{:.1f} のレース 3連複を配る".format(a_, b_), [(r["y"], r["both"] * d * 100) for d, r in sel])

print("\n■ 3連複◎-○-x の期待値(ワイドの3つの組のオッズから当たる確率を出す。係数はほかの2年で決める)")
rows = [(r["y"], np.log(1 / r["wab"]), np.log(1 / x["wax"]), np.log(1 / x["wbx"]), np.log(r["n"]), x["hit"], x["o"], x["pay"])
        for r in races for x in r["xs"] if x["wax"] and x["wbx"] and x["wax"] > 1 and x["wbx"] > 1]
Y = np.array([r[0] for r in rows]); X = np.array([r[1:5] for r in rows]); hit = np.array([r[5] for r in rows], dtype=float)
odds = np.array([r[6] for r in rows]); pay = np.array([r[7] for r in rows], dtype=float)
ev = np.zeros(len(rows)); pp = np.zeros(len(rows))
for t in YS:
    tr, te = Y != t, Y == t
    mu, sd = X[tr].mean(0), X[tr].std(0)
    w = np.array(F.fit_logit(((X[tr] - mu) / sd).tolist(), hit[tr].tolist(), 0.01))
    p = 1 / (1 + np.exp(-(np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ w)))
    pp[te] = p; ev[te] = p * odds[te]
print("  組 {}・当たり {}".format(len(rows), int(hit.sum())))
for a_, b_ in ((0, .5), (.5, .65), (.65, .8), (.8, .95), (.95, 1.1), (1.1, 1.3), (1.3, 99)):
    m = (ev >= a_) & (ev < b_)
    line("期待値 {:.2f}〜{:.2f}(見込みの的中 {:.1%})".format(a_, b_, pp[m].mean() if m.any() else 0), [(y, p) for y, p in zip(Y[m], pay[m])])

print("\n■ 細かく: 3連複を配った倍率 ÷ ワイドの倍率 の帯ごと(3連複の見込み = 0.75÷配った倍率、ワイドの見込み = 0.775÷ワイドの倍率)")
for a_, b_ in ((0, .75), (.75, .8), (.8, .85), (.85, .9), (.9, .95), (.95, 1.0), (1.0, 1.05), (1.05, 1.1), (1.1, 1.2), (1.2, 9)):
    sel = [(d, r) for d, r, q in zip(D, races, ratio) if a_ <= q < b_]
    if not sel:
        continue
    print("  比 {:.2f}〜{:.2f} {:4d}R  実際 {:5.1%}  3連複の見込み {:5.1%}  ワイドの見込み {:5.1%}  ワイド◎-○ {:4.0f}% [{}]  ワイドの倍率(中央値) {:.1f}".format(
        a_, b_, len(sel), np.mean([r["both"] for d, r in sel]), np.mean([.75 / d for d, r in sel]), np.mean([.775 / r["wab"] for d, r in sel]),
        np.mean([r["wpay"] for d, r in sel]), " ".join("{:4.0f}".format(np.mean([r["wpay"] for d, r in sel if r["y"] == y]) if any(r["y"] == y for d, r in sel) else 0) for y in YS),
        np.median([r["wab"] for d, r in sel])))
# ワイドの倍率の帯ごとに(倍率が高いレースほど比がぶれやすいので、同じ倍率の中でくらべる)
print("\n■ ワイド◎-○の倍率の帯ごとに、比が 0.9 未満 / 0.9〜1.0 / 1.0 以上")
for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 10), (10, 999)):
    out = []
    for a_, b_ in ((0, .9), (.9, 1.0), (1.0, 9)):
        sel = [r for d, r, q in zip(D, races, ratio) if a_ <= q < b_ and lo <= r["wab"] < hi]
        out.append("{:4.0f}%({:3d}R・的中 {:3.0%})".format(np.mean([r["wpay"] for r in sel]) if sel else 0, len(sel), np.mean([r["both"] for r in sel]) if sel else 0))
    print("  ワイド {:>3}〜{:<3}倍  {}".format(lo, hi, "  ".join(out)))

# 3) 相手の馬 x ごと: 3連複◎-○-x の売れ方(レースの中での割合)と、ワイド◎-x・○-x の売れ方(同じく割合)のずれ
print("\n■ ワイド◎-x・○-x を買う: 3連複の方が x をワイドより高く見ている(ずれ +)ほど、ワイドの回収率は上がるか")
bets = []
for r in races:
    xs = [x for x in r["xs"] if x["wax"] and x["wbx"] and x["wax"] > 1 and x["wbx"] > 1]
    if len(xs) < 3:
        continue
    st = sum(1 / x["o"] for x in xs); sa = sum(1 / x["wax"] for x in xs); sb = sum(1 / x["wbx"] for x in xs)
    rc = RACE[r["rid"]]; top3 = set(rc["order"][:3]); pay = rc["pay"].get("ワイド") or {}
    for x in xs:
        c = (1 / x["o"]) / st
        for me, w, s in ((r["a"], x["wax"], sa), (r["b"], x["wbx"], sb)):
            bets.append((r["y"], np.log(c) - np.log((1 / w) / s), w, pay.get(srt((me, x["x"])), 0)))
Yb = np.array([b[0] for b in bets]); sg = np.array([b[1] for b in bets]); wo = np.array([b[2] for b in bets]); pb = np.array([b[3] for b in bets], dtype=float)
print("  ワイドの組 {}点・当たり {}".format(len(bets), int((pb > 0).sum())))
for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 30), (30, 9999)):
    m0 = (wo >= lo) & (wo < hi)
    qs = np.quantile(sg[m0], [0, .1, .3, .7, .9, 1])
    print("  ワイド {}〜{}倍({}点)".format(lo, hi, m0.sum()))
    for a_, b_ in zip(qs[:-1], qs[1:]):
        m = m0 & (sg >= a_) & (sg <= b_)
        line("    ずれ {:+.2f}〜{:+.2f}".format(a_, b_), [(y, p) for y, p in zip(Yb[m], pb[m])])

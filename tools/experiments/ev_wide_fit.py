"""ワイドの当たる確率を、単勝から出した確率(Harville)とワイドのオッズが表す確率で混ぜて決め、期待値を確かめる
  python tools/experiments/ev_wide_fit.py
  logit P = c + h × logit(Harville の確率) + m × logit(ワイドの市場の確率)。ワイドの市場の確率は 1/(オッズの下限と上限の平均) を
  レースで合計3になるように揃えたもの。c, h, m を2022〜2025年で決め、2026年の期待値ごとの回収率で確かめる"""
import glob, gzip, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_check import pairs, wf, A, B, R          # noqa: E402  (確率の作り方はアプリと同じ)

X, Y, RET, ODL, YR = [], [], [], [], []
for f in sorted(glob.glob(R + "/keiba/days/202*.json")):
    d = json.load(open(f)); y = os.path.basename(f)[:4]
    for rid, r in (d.get("races") or {}).items():
        res = r.get("result") or {}; od = res.get("odds") or {}; wfin = r.get("wide_final")
        pay = (res.get("payout") or {}).get("ワイド") or {}
        if r.get("center") != "jra" or not wfin or not od or not pay or len(res.get("order") or []) < 3:
            continue
        us = [u for u in od if (od[u] or 0) > 1]
        if len(us) < 5:
            continue
        inv = sum(1 / od[u] for u in us); p3 = wf.get(rid) or {}
        cl = lambda v: min(max(v, 1e-3), 0.99)
        z = {u: (1 / od[u] / inv) ** A * (cl(p3.get(u, .5)) / (1 - cl(p3.get(u, .5)))) ** B for u in us}
        s = sum(z.values()); P = pairs({u: v / s for u, v in z.items()})
        ks = [k for k in wfin if k in P]
        mk = {k: 2 / (wfin[k][0] + wfin[k][1]) for k in ks}
        tot = sum(mk.values())
        for k in ks:
            ph = min(max(P[k], 1e-5), 1 - 1e-5); pm = min(max(mk[k] * 3 / tot, 1e-5), 1 - 1e-5)
            X.append((math.log(ph / (1 - ph)), math.log(pm / (1 - pm)))); Y.append(1 if pay.get(k) else 0)
            RET.append(pay.get(k, 0)); ODL.append(wfin[k]); YR.append(y)
X = np.array(X); Y = np.array(Y); RET = np.array(RET); YR = np.array(YR); ODL = np.array(ODL)
print("組", len(Y))


def fit(Xf, Yf):
    """ロジスティック回帰(ニュートン法)"""
    Z = np.column_stack([np.ones(len(Xf)), Xf]); w = np.zeros(Z.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-Z @ w)); g = Z.T @ (Yf - p); H = (Z * (p * (1 - p))[:, None]).T @ Z
        w += np.linalg.solve(H, g)
    return w


tr = YR < "2026"
w = fit(X[tr], Y[tr])
print("c={:.3f} h(単勝から)={:.3f} m(ワイドのオッズから)={:.3f}".format(*w))


def prob(Xs, w):
    return 1 / (1 + np.exp(-(w[0] + Xs @ w[1:])))


te = ~tr
for name, p in (("単勝から(今のアプリ)", 1 / (1 + np.exp(-X[te, 0]))), ("ワイドのオッズだけ", 1 / (1 + np.exp(-X[te, 1]))),
                ("混ぜる", prob(X[te], w))):
    ll = np.mean(Y[te] * np.log(p) + (1 - Y[te]) * np.log(1 - p))
    print("\n■ {} 2026年 対数尤度 {:.5f}  期待値(×オッズの下限)ごとの実際の回収率".format(name, ll))
    ev = p * ODL[te, 0]
    for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 1.1), (1.1, 1.3), (1.3, 99)):
        m = (ev >= lo) & (ev < hi)
        if m.sum():
            print("   {:.1f}〜{:<4} {:7d}組 回収率 {:4.0f}%".format(lo, hi if hi < 99 else "", int(m.sum()), RET[te][m].mean()))
json.dump({"c": round(float(w[0]), 4), "h": round(float(w[1]), 4), "m": round(float(w[2]), 4)},
          open(os.path.join(os.environ.get("KEIBA_WORK", "/tmp"), "wide_fit.json"), "w"))

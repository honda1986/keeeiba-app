"""単勝・ワイドの期待値のための確率を作って確かめる(中央、2024年1月〜、月ごとの3着内の確率と確定の単勝オッズ)
  python tools/experiments/ev_fit.py
  1着の確率 … P ∝ q^a × (p3/(1-p3))^b(q=単勝オッズから出る市場の確率、p3=モデルの3着内の確率)。a, b は2024・2025年で決める
  ワイドの確率 … 1着の確率から Harville(2着・3着も残りの馬の確率の比で決まる)で、2頭とも3着内に入る確率。
               2着・3着は確率を g 乗して平らにする(人気馬を強く見すぎないように)。g も2024・2025年で決める"""
import glob, gzip, json, math, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = json.load(open(f))
    for rid, r in (d.get("races") or {}).items():
        res = r.get("result") or {}
        od = res.get("odds") or {}
        if r.get("center") != "jra" or rid not in wf or len(res.get("order") or []) < 3 or not od:
            continue
        us = [u for u in wf[rid] if (od.get(u) or 0) > 1.0]
        if len(us) < 5:
            continue
        o = np.array([od[u] for u in us]); p3 = np.clip(np.array([wf[rid][u] for u in us]), 1e-3, 0.99)
        q = (1 / o) / (1 / o).sum()
        win = us.index(str(res["order"][0])) if str(res["order"][0]) in us else None
        top3 = [us.index(str(x)) for x in res["order"][:3] if str(x) in us]
        pay = res.get("payout") or {}
        races.append(dict(y=os.path.basename(f)[:4], us=us, o=o, lq=np.log(q), lp=np.log(p3 / (1 - p3)), win=win, top3=top3,
                          tan=pay.get("単勝") or {}, wide=pay.get("ワイド") or {}))
fit = [r for r in races if r["y"] in ("2024", "2025") and r["win"] is not None]


def winp(r, a, b):
    z = a * r["lq"] + b * r["lp"]
    z = np.exp(z - z.max())
    return z / z.sum()


def nll(a, b, rs):
    return -sum(math.log(max(winp(r, a, b)[r["win"]], 1e-12)) for r in rs) / len(rs)


# a, b を粗い格子 → 細かい格子で決める
best = min(((nll(a, b, fit), a, b) for a in np.arange(0.6, 1.41, 0.1) for b in np.arange(-0.2, 0.61, 0.1)))
_, a0, b0 = best
best = min(((nll(a, b, fit), a, b) for a in np.arange(a0 - 0.1, a0 + 0.101, 0.02) for b in np.arange(b0 - 0.1, b0 + 0.101, 0.02)))
_, A, B = best
test = [r for r in races if r["y"] == "2026" and r["win"] is not None]
print("1着の確率 a={:.2f}(市場) b={:.2f}(モデル)".format(A, B))
print("  対数尤度(2026年。大きいほど良い) 市場だけ {:.4f} / 混ぜる {:.4f}".format(-nll(1, 0, test), -nll(A, B, test)))


def harville_pairs(p, g):
    """2頭とも3着内に入る確率の行列。2着・3着は p^g の比で決める"""
    n = len(p); pg = p ** g
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            pj = p[i] * pg[j] / (pg.sum() - pg[i])          # i 1着 → j 2着
            for k in range(n):
                if k in (i, j):
                    continue
                pk = pj * pg[k] / (pg.sum() - pg[i] - pg[j])    # → k 3着
                # {i, j, k} が3着内。3組の2頭に足す
                P[i, j] += pk; P[i, k] += pk; P[j, k] += pk
    return P + P.T


def pair_ll(g, rs):
    s = 0.0; n = 0
    for r in rs:
        P = harville_pairs(winp(r, A, B), g)
        t = set(r["top3"])
        m = len(r["us"])
        for i in range(m):
            for j in range(i + 1, m):
                p = min(max(P[i, j], 1e-6), 1 - 1e-6)
                y = i in t and j in t
                s += math.log(p if y else 1 - p); n += 1
    return s / n


sub = fit[::4]
G = max((pair_ll(g, sub), g) for g in (0.6, 0.7, 0.8, 0.9, 1.0))[1]
print("ワイドの確率 g={:.1f}(1.0 がふつうの Harville)".format(G))

# 確かめ: 予想した確率と、実際の割合(2026年)
print("\n■ 1着の確率(2026年) 予想 → 実際")
bins = [0, .05, .1, .2, .3, .5, 1]
acc = [[0, 0, 0.0] for _ in bins[:-1]]
for r in test:
    for i, p in enumerate(winp(r, A, B)):
        k = next(b for b in range(len(bins) - 1) if p < bins[b + 1])
        acc[k][0] += 1; acc[k][1] += i == r["win"]; acc[k][2] += p
for (lo, hi), (n, w, p) in zip(zip(bins, bins[1:]), acc):
    print("  {:3.0%}〜{:3.0%} {:6d}頭  予想 {:5.1%} → 実際 {:5.1%}".format(lo, hi, n, p / n, w / n))
print("\n■ ワイドの確率(2026年) 予想 → 実際")
bins = [0, .05, .1, .2, .3, .5, 1]
acc = [[0, 0, 0.0] for _ in bins[:-1]]
for r in test:
    P = harville_pairs(winp(r, A, B), G); t = set(r["top3"])
    for i in range(len(r["us"])):
        for j in range(i + 1, len(r["us"])):
            p = P[i, j]; k = next(b for b in range(len(bins) - 1) if p < bins[b + 1])
            acc[k][0] += 1; acc[k][1] += (i in t and j in t); acc[k][2] += p
for (lo, hi), (n, w, p) in zip(zip(bins, bins[1:]), acc):
    print("  {:3.0%}〜{:3.0%} {:6d}組  予想 {:5.1%} → 実際 {:5.1%}".format(lo, hi, n, p / n, w / n))

# 単勝の期待値(確定オッズで)ごとの、実際の回収率
print("\n■ 単勝の期待値ごとの実際の回収率(確定オッズ)   2024  2025  2026 | 計")
edges = [0, .6, .7, .8, .9, 1.0, 1.1, 99]
for lo, hi in zip(edges, edges[1:]):
    cols, tn, tr = [], 0, 0
    for y in ("2024", "2025", "2026"):
        n = ret = 0
        for r in races:
            if r["y"] != y:
                continue
            ev = winp(r, A, B) * r["o"]
            for i in np.where((ev >= lo) & (ev < hi))[0]:
                n += 1; ret += r["tan"].get(r["us"][i], 0)
        cols.append("%4.0f%%" % (ret / max(n, 1))); tn += n; tr += ret
    print("  期待値 {:.1f}〜{:4.1f} {} | {:6d}頭 {:4.0f}%".format(lo, hi if hi < 99 else 9.9, " ".join(cols), tn, tr / max(tn, 1)))
json.dump({"a": round(float(A), 3), "b": round(float(B), 3), "g": G}, open(os.path.join(os.environ.get("KEIBA_WORK", "/tmp"), "ev_params.json"), "w"))

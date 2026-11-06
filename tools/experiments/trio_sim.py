"""組み合わせの確率をそろえて出し、3連複の期待値で絞る(中央、2024年1月〜。◎70%以上のレースの3連複◎-○-x の確定オッズ trio_final)
  python tools/experiments/trio_sim.py
  各馬の強さ w を、Plackett-Luce(強さに比例して1着、残りから強さに比例して2着…)で計算した3着内の確率が
  モデルの3着内の確率(合計3にそろえる)と合うように決める。そこから3頭の組が3着内にそろう確率(3連複)を出し、
  期待値 = 確率 × 確定オッズ で絞る。確率の目盛りの合わせ方・期待値の下限は2024・2025年で決め、2026年で確かめる"""
import glob, json, math, os, random
from itertools import permutations
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")


def pl_top3(w):
    """Plackett-Luce で各馬が3着内に入る確率(厳密)"""
    S = w.sum(); n = len(w)
    p1 = w / S
    den1 = S - w                                   # 1着が j のときの残り
    A = p1[:, None] * w[None, :] / den1[:, None]   # A[j,k] = P(1着 j, 2着 k)
    np.fill_diagonal(A, 0)
    B = S - w[:, None] - w[None, :]
    C = np.where(B > 1e-12, A / np.maximum(B, 1e-12), 0); np.fill_diagonal(C, 0)
    tot = C.sum()
    p3rd = w * (tot - C.sum(1) - C.sum(0))
    return p1 + A.sum(0) + p3rd


def fit(p3):
    """3着内の確率に合う強さ"""
    t = np.clip(np.array(p3, float), .005, .995); t = np.minimum(t * 3 / t.sum(), .995)
    w = t / t.sum()
    for _ in range(200):
        q = pl_top3(w)
        w = w * (t / np.maximum(q, 1e-9)) ** .8
        w /= w.sum()
        if np.abs(q - t).max() < 1e-4:
            break
    return w


def trio_p(w, us):
    S = w.sum(); p = 0.0
    for a, b, c in permutations(us):
        p += w[a] / S * w[b] / (S - w[a]) * w[c] / (S - w[a] - w[b])
    return p


races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res, tf = r.get("pred") or {}, r.get("result") or {}, r.get("trio_final")
        if r.get("center") != "jra" or not p.get("model") or not tf or len(res.get("order") or []) < 3:
            continue
        od = (res.get("odds") or {})
        hs = [h for h in p["horses"] if h.get("p3") is not None and od.get(str(h["umaban"]))]
        h, t, aite = p.get("honmei"), p.get("taikou"), p.get("aite") or []
        if not h or not t or not aite or len(hs) < 6:
            continue
        us = [x["umaban"] for x in hs]; ix = {u: i for i, u in enumerate(us)}
        if h not in ix or t not in ix:
            continue
        w = fit([x["p3"] for x in hs])
        top3 = set(res["order"][:3]); pay = (res.get("payout") or {})
        cands = []
        for u in us:
            if u in (h, t):
                continue
            k = "-".join(str(x) for x in sorted((h, t, u)))
            if tf.get(k):
                cands.append(dict(u=u, odds=tf[k], p=trio_p(w, (ix[h], ix[t], ix[u])), hit=int(top3 == {h, t, u}),
                                  pay=(pay.get("3連複") or {}).get(k, 0)))
        if not cands:
            continue
        hh = {x["umaban"]: x for x in p["horses"]}
        races.append(dict(y=d[:4], d=d, h=h, t=t, a=aite[0], p1=hh[h]["p3"], p2=hh[t]["p3"], cands=cands,
                          ext="距離延長" in (hh[aite[0]].get("axis_ng") or []),
                          wide=(pay.get("ワイド") or {}).get("-".join(str(x) for x in sorted((h, t))), 0)))
print("レース", len(races), {y: sum(1 for r in races if r["y"] == y) for y in YS}, "組", sum(len(r["cands"]) for r in races))

# 1) 確率は当たっているか(◎-○-x の組ごと。2024・2025年 / 2026年)
allc = [(r["y"], c) for r in races for c in r["cands"]]
print("\n■ 1. Plackett-Luce の3連複の確率と、実際に当たった割合")
for lo, hi in ((0, .01), (.01, .03), (.03, .06), (.06, .1), (.1, .15), (.15, .25), (.25, 1)):
    for per, lab in ((("2024", "2025"), "2024-25"), (("2026",), "2026")):
        xs = [c for y, c in allc if y in per and lo <= c["p"] < hi]
        if xs:
            print("  確率 {:3.0%}〜{:3.0%} {:7s} {:5d}組 確率の平均 {:5.1%} / 実際 {:5.1%}".format(lo, hi, lab, len(xs), sum(c["p"] for c in xs) / len(xs), sum(c["hit"] for c in xs) / len(xs)))

# 目盛りの合わせ直し(2024・2025年): logit(実際) = a + b × logit(確率) + c × logit(市場の確率 0.75/オッズ)
lg = lambda v: math.log(min(max(v, 1e-6), 1 - 1e-6) / (1 - min(max(v, 1e-6), 1 - 1e-6)))


def fitlog(X, y):
    X = np.array(X); y = np.array(y, float); wv = np.zeros(X.shape[1])
    for _ in range(60):
        pz = 1 / (1 + np.exp(-X @ wv)); H = (X * (pz * (1 - pz))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]); wv += np.linalg.solve(H, X.T @ (y - pz))
    return wv


tr = [c for y, c in allc if y != "2026"]
W1 = fitlog([[1, lg(c["p"])] for c in tr], [c["hit"] for c in tr])
W2 = fitlog([[1, lg(c["p"]), lg(min(.95, .75 / c["odds"]))] for c in tr], [c["hit"] for c in tr])
print("  合わせ直し(確率だけ) 係数 {} / (確率+市場) 係数 {}".format(" ".join("%.2f" % v for v in W1), " ".join("%.2f" % v for v in W2)))
for r in races:
    for c in r["cands"]:
        c["p1c"] = 1 / (1 + math.exp(-(W1[0] + W1[1] * lg(c["p"]))))
        c["p2c"] = 1 / (1 + math.exp(-(W2[0] + W2[1] * lg(c["p"]) + W2[2] * lg(min(.95, .75 / c["odds"])))))
for key, lab in (("p", "そのまま"), ("p1c", "合わせ直し"), ("p2c", "市場と混ぜて合わせ直し")):
    te = [c for y, c in allc if y == "2026"]
    ll = np.mean([math.log(c[key]) if c["hit"] else math.log(1 - c[key]) for c in te])
    print("  2026年の対数尤度 {:24s} {:.4f}".format(lab, ll))


# 2) 期待値で絞る
def ev(c, key):
    return c[key] * c["odds"]


def yr(rows):
    return " ".join("{:4.0f}".format(sum(v for yy, v, s in rows if yy == y) / max(1, sum(s for yy, v, s in rows if yy == y)) * 100) if any(yy == y for yy, v, s in rows) else "   -" for y in YS)


def summ(rows):
    st = sum(s for _, _, s in rows); rt = sum(v for _, v, s in rows)
    return len(rows), rt / st * 100 if st else 0


now = [(r["y"], r["wide"] if r["ext"] else next((c["pay"] for c in r["cands"] if c["u"] == r["a"]), 0), 100)
       for r in races if r["p1"] >= .8 and r["p1"] - r["p2"] >= .15]
print("\n■ 2. 期待値で絞る(2024・2025年で選ぶ → 2026年)。今の買い方: {}R {:.0f}% [{}]".format(*summ(now), yr(now)))
for key, lab in (("p", "そのまま"), ("p1c", "合わせ直し"), ("p2c", "市場と混ぜて合わせ直し")):
    print("  [{}の確率]".format(lab))
    res = []
    for t1 in (.7, .8):
        for g in (-1, .15):
            for k in (0, .8, 1.0, 1.2, 1.5):
                for how in ("▲だけ", "期待値いちばん", "期待値k以上を全部"):
                    rows = []
                    for r in races:
                        if r["p1"] < t1 or r["p1"] - r["p2"] < g:
                            continue
                        if how == "▲だけ":
                            cs = [c for c in r["cands"] if c["u"] == r["a"] and ev(c, key) >= k]
                        elif how == "期待値いちばん":
                            cs = [max(r["cands"], key=lambda c: ev(c, key))]; cs = [c for c in cs if ev(c, key) >= k]
                        else:
                            cs = [c for c in r["cands"] if ev(c, key) >= max(k, .8)]
                        if cs:
                            rows.append((r["y"], sum(c["pay"] for c in cs), 100 * len(cs)))
                    sel = [x for x in rows if x[0] != "2026"]; te = [x for x in rows if x[0] == "2026"]
                    if len(sel) >= 150:
                        res.append((summ(sel)[1], t1, g, k, how, summ(sel), summ(te), yr(rows), sum(v - s for yy, v, s in te)))
    res.sort(key=lambda z: -z[0])
    for z in res[:6]:
        print("    ◎{:.0%}以上 差{:>4s} 期待値{:.1f}以上 {:14s} 2024-25 {:4d}R {:4.0f}% / 2026 {:3d}R {:4.0f}% [{}] 2026収支{:+,}円".format(
            z[1], "なし" if z[2] < 0 else "15%", z[3], z[4], z[5][0], z[5][1], z[6][0], z[6][1], z[7], int(z[8])))

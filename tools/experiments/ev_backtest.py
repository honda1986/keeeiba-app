"""期待値で買う(オッズを使う)の確かめ。
1着の確率 = モデル(月ごと)と市場(確定の単勝オッズ)を、条件付きロジットで混ぜる(その月より前のレースで係数を決める)。
単勝 … 期待値 = 確率 × オッズ が k 以上の馬を買う
ワイド … 2頭とも3着内の確率を Harville で出し、市場の確率との比 × 払戻率 0.775 を期待値の見積もりにする"""
import gzip, json, math, os, pickle, random, sys
from collections import defaultdict
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_model as T
SP = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
PW = pickle.load(open(SP + "/ev/wf_win.pkl", "rb"))
races = {rc["id"]: (d, rc) for d, rc in T.load_races() if rc["id"] in PW}
RS = []
for rid, (d, rc) in sorted(races.items(), key=lambda z: z[1][0]):
    res = rc["result"]
    od = res.get("odds") or {}
    us = [u for u in PW[rid] if od.get(str(u))]
    if len(us) < 5 or len(us) != len(PW[rid]):
        continue
    inv = np.array([1 / od[str(u)] for u in us])
    RS.append({"rid": rid, "d": d, "u": us, "pm": np.array([PW[rid][u] for u in us]), "mk": inv / inv.sum(),
               "odds": np.array([od[str(u)] for u in us]), "win": res["order"][0], "top3": set(res["order"][:3]),
               "pay": res.get("payout") or {}})
for r in RS:
    r["Z"] = np.stack([np.log(r["pm"] + 1e-9), np.log(r["mk"] + 1e-9)], 1)
    r["wi"] = r["u"].index(r["win"]) if r["win"] in r["u"] else None
print("レース", len(RS), RS[0]["d"], RS[-1]["d"])


def fit(rs, iters=30, w0=None):
    """条件付きロジット: s = a log pm + b log mk。勝ち馬の尤度を最大に"""
    w = np.array([1.0, 1.0]) if w0 is None else w0.copy()
    for _ in range(iters):
        g, H = np.zeros(2), np.zeros((2, 2))
        for r in rs:
            i = r["wi"]
            if i is None:
                continue
            Z = r["Z"]
            s = Z @ w
            p = np.exp(s - s.max()); p /= p.sum()
            ez = p @ Z
            g += Z[i] - ez
            H -= (Z * p[:, None]).T @ Z - np.outer(ez, ez)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-4:
            break
    return w


def comb(r, w):
    Z = np.stack([np.log(r["pm"] + 1e-9), np.log(r["mk"] + 1e-9)], 1)
    s = Z @ w
    p = np.exp(s - s.max())
    return p / p.sum()


# 係数は月ごとに、その月より前の(2024年1月からの)レースで決める。2024年4月から使う
months = sorted({r["d"][:6] for r in RS})
W, w = {}, None
for m in months:
    prev = [r for r in RS if r["d"][:6] < m]
    if len(prev) >= 600:
        w = fit(prev, w0=w)
        W[m] = w
for m in months:
    if m in W and m[4:] in ("04", "07", "10", "01"):
        print("係数", m, "モデル {:.2f} / 市場 {:.2f}".format(*W[m]))
EV = [r for r in RS if r["d"][:6] in W]
for r in EV:
    r["pc"] = comb(r, W[r["d"][:6]])
print("確かめるレース", len(EV), EV[0]["d"], EV[-1]["d"])


def ll(rs, key):
    s = 0.0
    for r in rs:
        i = r["u"].index(r["win"]) if r["win"] in r["u"] else None
        if i is not None:
            s += math.log(r[key][i] + 1e-12)
    return s / len(rs)


print("1着の対数尤度(大きいほど良い) モデル {:.4f} / 市場 {:.4f} / 混ぜる {:.4f}".format(ll(EV, "pm"), ll(EV, "mk"), ll(EV, "pc")))

YEARS = (("2024年4〜12月", "20240401", "20250101"), ("2025年", "20250101", "20260101"), ("2026年", "20260101", "99999999"))


def tansho(r, k, cap, key="pc", best=False):
    """(賭け金, 払戻)"""
    ev = r[key] * r["odds"]
    idx = [i for i in range(len(r["u"])) if ev[i] >= k and r["odds"][i] <= cap]
    if best and idx:
        idx = [max(idx, key=lambda i: ev[i])]
    st = 100 * len(idx)
    rt = sum((r["pay"].get("単勝") or {}).get(str(r["u"][i]), 0) for i in idx)
    return st, rt


def harville_pairs(p, top=6):
    """上位 top 頭の組ごとに、2頭とも3着内の確率(Harville)"""
    n = len(p)
    cand = list(np.argsort(-p)[:top])
    out = {}
    for i, j in combinations(cand, 2):
        s = 0.0
        for k in range(n):
            if k in (i, j):
                continue
            for a, b, c in ((i, j, k), (j, i, k), (i, k, j), (j, k, i), (k, i, j), (k, j, i)):
                pa, pb, pc_ = p[a], p[b], p[c]
                s += pa * pb / (1 - pa) * pc_ / (1 - pa - pb)
        out[(i, j)] = s
    return out


def wide_bets(r, k, key="pc"):
    if "wp" not in r:
        r["wp"] = harville_pairs(r["pc"])
        r["wm"] = {ij: v for ij, v in harville_pairs(r["mk"], top=len(r["u"])).items() if ij in r["wp"] or (ij[1], ij[0]) in r["wp"]}
    st = rt = 0
    for (i, j), pc in r["wp"].items():
        pm = r["wm"].get((i, j)) or r["wm"].get((j, i))
        if not pm:
            continue
        if 0.775 * pc / pm >= k:
            st += 100
            key2 = "-".join(str(x) for x in sorted((r["u"][i], r["u"][j])))
            rt += (r["pay"].get("ワイド") or {}).get(key2, 0)
    return st, rt


def boot(vals, B=400):
    random.seed(1)
    if len(vals) < 30:
        return 0, 0
    tot = [(s, t) for s, t in vals]
    res = []
    for _ in range(B):
        smp = [random.choice(tot) for _ in tot]
        S = sum(s for s, _ in smp)
        res.append(sum(t for _, t in smp) / S if S else 0)
    res.sort()
    return res[int(B * .05)], res[int(B * .95)]


def report(name, fn):
    parts, allv = [], []
    for yn, a, b in YEARS:
        v = [fn(r) for r in EV if a <= r["d"] < b]
        v = [x for x in v if x[0]]
        S, Rt = sum(x[0] for x in v), sum(x[1] for x in v)
        parts.append("{} {:>5}点 {:>4.0f}%".format(yn[:6], S // 100, Rt / S * 100 if S else 0))
        allv += v
    S, Rt = sum(x[0] for x in allv), sum(x[1] for x in allv)
    lo, hi = boot(allv)
    print("  {:<30} ".format(name) + " | ".join(parts) + " | 計 {:>4.0f}%(90%の幅 {:.0f}〜{:.0f})".format(
        Rt / S * 100 if S else 0, lo * 100, hi * 100))


print("\n■ 単勝: 期待値 k 以上の馬を全部(オッズ上限つき)")
for cap in (20, 50, 999):
    for k in (1.0, 1.1, 1.2, 1.3, 1.5):
        report("混ぜた確率 k={:.1f} オッズ{}倍まで".format(k, cap if cap < 999 else "∞"), lambda r: tansho(r, k, cap))
print("\n■ 単勝: モデルだけの確率(市場と混ぜない)")
for k in (1.0, 1.2, 1.5, 2.0):
    report("モデルだけ k={:.1f} オッズ50倍まで".format(k), lambda r: tansho(r, k, 50, "pm"))
print("\n■ 単勝: 1レース1頭(期待値いちばん)")
for k in (1.0, 1.1, 1.2, 1.3):
    report("混ぜた確率 k={:.1f} オッズ50倍まで".format(k), lambda r: tansho(r, k, 50, best=True))
print("\n■ ワイド: 見積もりの期待値 k 以上の組(上位6頭の中)")
for k in (1.0, 1.1, 1.2, 1.3, 1.5):
    report("k={:.1f}".format(k), lambda r: wide_bets(r, k))
pickle.dump({"W": W}, open(SP + "/ev/coef.pkl", "wb"))

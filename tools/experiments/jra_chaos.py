"""中央でも「荒れ度」(1番人気が3着に来ない確率)がオッズから読めるか、買い方に使えるか(2022年〜)
  python tools/experiments/jra_chaos.py
  荒れ度の式(1番人気のオッズとオッズのばらけ方のロジスティック回帰。高知の最終レースと同じ形)は
  2022〜2025年で決め、2026年で確かめる。成績は 2024〜2026年"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import keiba_final as F
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YRS = ("2024", "2025", "2026")
key = lambda c: "-".join(str(z) for z in sorted(c))
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        res = r.get("result") or {}
        if r.get("center") != "jra" or len(res.get("order") or []) < 3:
            continue
        odds = {int(u): v for u, v in (res.get("odds") or {}).items() if v}
        x = F.chaos_x(odds)
        if not x:
            continue
        fav = min((u for u in odds if odds[u] > 1), key=lambda u: odds[u])
        p = r.get("pred") or {}; pay = res.get("payout") or {}
        row = dict(y=d[:4], x=x, out=int(fav not in res["order"][:3]), buy=p.get("verdict") in ("A", "B") and bool(p.get("bets")))
        h, t = p.get("honmei"), p.get("taikou"); a = (p.get("aite") or [None])[0]
        if h and t and a:
            row.update(hfav=h == fav, win=(pay.get("単勝") or {}).get(str(h), 0),
                       wide=(pay.get("ワイド") or {}).get(key([h, t]), 0), trio=(pay.get("3連複") or {}).get(key([h, t, a]), 0))
            if row["buy"]:
                b = p["bets"][0]; row["cur"] = sum((pay.get(b["type"]) or {}).get(key(c), 0) for c in b["combos"])
        rows.append(row)


def fit(X, y, l2=1.0):
    Z = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Z.shape[1]); Rg = l2 * np.eye(len(w)); Rg[0, 0] = 0
    for _ in range(60):
        p = 1 / (1 + np.exp(-Z @ w)); H = (Z * (p * (1 - p))[:, None]).T @ Z + Rg + 1e-6 * np.eye(len(w)); w += np.linalg.solve(H, Z.T @ (y - p) - Rg @ w)
    return w


X = np.array([r["x"] for r in rows]); y = np.array([r["out"] for r in rows], float); yr = np.array([r["y"] for r in rows])
tr = yr < "2026"; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
w = fit((X[tr] - mu) / sd, y[tr]); pz = 1 / (1 + np.exp(-(w[0] + ((X - mu) / sd) @ w[1:])))
for r, c in zip(rows, pz):
    r["c"] = c
print("中央 {}R  係数 {}".format(len(rows), w.round(2)))
print("\n■ 荒れ度ごとの、1番人気が3着に来なかった率")
for lo, hi in ((0, .2), (.2, .3), (.3, .4), (.4, .55), (.55, 1)):
    m = (pz >= lo) & (pz < hi)
    print("  荒れ度 {:3.0%}〜{:3.0%} {:5d}R 実際 {:4.0%} / 見込み {:4.0%}  [{}]".format(lo, hi, int(m.sum()), y[m].mean(), pz[m].mean(),
          " ".join("{} {:.0%}".format(t, y[m & (yr == t)].mean()) for t in ("2022", "2023") + YRS if (m & (yr == t)).sum() >= 20)))


def show(name, m, keys):
    s = []
    for k, lab in keys:
        v = [r[k] for r in m]
        s.append("{} {:4.0f}% [{}]".format(lab, np.mean(v) if v else 0, " ".join(
            "{:4.0f}".format(np.mean([r[k] for r in m if r["y"] == t])) if sum(r["y"] == t for r in m) >= 10 else "   -" for t in YRS)))
    print("  {:20s} {:5d}R ".format(name, len(m)) + " | ".join(s))


buy = [r for r in rows if r["buy"] and r["y"] >= "2024" and "cur" in r]
print("\n■ 買うレース(今の買い方=3連複◎○▲、▲が延長ならワイド)を荒れ度の4等分で  [2024 2025 2026]")
qs = np.quantile([r["c"] for r in buy], [0, .25, .5, .75, 1])
for i in range(4):
    show("荒れ度{:.0%}〜{:.0%}".format(qs[i], qs[i + 1]), [r for r in buy if qs[i] <= r["c"] and (r["c"] < qs[i + 1] or i == 3)],
         (("cur", "今"), ("trio", "3連複"), ("wide", "ワイド")))
print("  (◎が1番人気: {}R / {}R)".format(sum(r["hfav"] for r in buy), len(buy)))
skip = [r for r in rows if not r["buy"] and r["y"] >= "2024" and "hfav" in r]
print("\n■ 見送りレース(◎の確率が足りない等)を荒れ度と◎の人気で分けて  [2024 2025 2026]")
for lo, hi in ((0, .3), (.3, .45), (.45, 1)):
    for hf in (True, False):
        show("荒れ度{:.0%}〜{:.0%} ◎{}".format(lo, hi, "1番人気" if hf else "人気薄"), [r for r in skip if lo <= r["c"] < hi and r["hfav"] == hf],
             (("win", "単勝◎"), ("wide", "ワイド◎○"), ("trio", "3連複◎○▲")))

"""単勝・ワイドの期待値を、確定オッズで確かめる(中央、2022年1月〜)
  python tools/experiments/ev_check.py
  単勝は結果の確定オッズ(result.odds)、ワイドは wide_final(「過去の日を再現」mode=odds で取る)
  確率はアプリと同じ: 1着 P ∝ q^1.00 × (p3/(1-p3))^0.04(p3 が無い2022・2023年は市場の確率だけ)、
  ワイドは Harville(2着・3着は0.7乗)。期待値 = 確率 × 確定オッズの下限。実際の払戻と比べる"""
import glob, gzip, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
A, B, G = 1.0, 0.04, 0.7
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]


def pairs(win, g=G):
    us = list(win); p = np.array([win[u] for u in us]); pg = p ** g; S = pg.sum(); n = len(us); P = {}
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            pij = p[i] * pg[j] / (S - pg[i])
            for k in range(n):
                if k in (i, j):
                    continue
                v = pij * pg[k] / (S - pg[i] - pg[j])
                for a, b in ((i, j), (i, k), (j, k)):
                    key = "-".join(sorted((us[a], us[b]), key=int))
                    P[key] = P.get(key, 0) + v
    return P


rows, tans = [], []
nr = 0
for f in sorted(glob.glob(R + "/keiba/days/202*.json")):
    d = json.load(open(f))
    y = os.path.basename(f)[:4]
    for rid, r in (d.get("races") or {}).items():
        res = r.get("result") or {}
        wfin, od = r.get("wide_final"), res.get("odds") or {}
        pay = (res.get("payout") or {}).get("ワイド") or {}
        if r.get("center") != "jra" or not od or not pay or len(res.get("order") or []) < 3:
            continue
        us = [u for u in od if (od[u] or 0) > 1]
        if len(us) < 5:
            continue
        inv = sum(1 / od[u] for u in us)
        p3 = wf.get(rid) or {}
        z = {u: (1 / od[u] / inv) ** A * ((min(max(p3.get(u, 0.5), 1e-3), 0.99)) / (1 - min(max(p3.get(u, 0.5), 1e-3), 0.99))) ** B
             for u in us}
        s = sum(z.values()); win = {u: v / s for u, v in z.items()}
        tp = (res.get("payout") or {}).get("単勝") or {}
        if p3:                         # 2022・2023年はモデルの確率が無く、期待値がどの馬も払戻率と同じになるので数えない
            for u in us:
                tans.append((y, win[u], od[u], tp.get(u, 0)))
        if not wfin:
            continue
        P = pairs(win)
        nr += 1
        for k, (lo, hi) in wfin.items():
            if k in P:
                rows.append((y, P[k], lo, hi, pay.get(k, 0)))
print("レース", nr, "単勝", len(tans), "頭 / ワイド", len(rows), "組")


def table(title, xs, ev, edges):
    ys = sorted({r[0] for r in xs})
    print("\n■ " + title + "   " + " ".join(ys) + " | 計")
    for lo_e, hi_e in zip(edges, edges[1:]):
        cols, tn, tr = [], 0, 0
        for y in ys:
            s = [r for r in xs if r[0] == y and lo_e <= ev(r) < hi_e]
            cols.append("%4.0f%%" % (sum(r[-1] for r in s) / max(len(s), 1)))
            tn += len(s); tr += sum(r[-1] for r in s)
        print("  {:.2f}〜{:<5} {} | {:7d} {:4.0f}%".format(lo_e, hi_e if hi_e < 99 else "", " ".join(cols), tn, tr / max(tn, 1)))


table("単勝の期待値(確率×確定オッズ)ごとの、実際の回収率(2024年〜)", tans, lambda r: r[1] * r[2], [0, .5, .6, .7, .75, .8, .85, .9, 1.0, 99])
print("\n■ 単勝のオッズの帯ごと(2024年〜。期待値の平均 → 実際の回収率)")
for lo_e, hi_e in ((1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 9999)):
    s = [r for r in tans if lo_e <= r[2] < hi_e]
    if s:
        print("  {:4d}〜{:<4d}倍 {:7d}頭 期待値 {:.2f} → 回収率 {:4.0f}%".format(lo_e, hi_e, len(s), sum(r[1] * r[2] for r in s) / len(s),
                                                                    sum(r[3] for r in s) / len(s)))
if not rows:
    raise SystemExit("ワイドの確定オッズがまだありません(「過去の日を再現」を mode=odds でまわす)")
YS = sorted({r[0] for r in rows})
print("\n■ ワイドの期待値(確率×確定オッズの下限)ごとの、実際の回収率   " + " ".join(YS) + " | 計")
edges = [0, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.5, 99]
for lo_e, hi_e in zip(edges, edges[1:]):
    cols, tn, tr = [], 0, 0
    for y in YS:
        xs = [r for r in rows if r[0] == y and lo_e <= r[1] * r[2] < hi_e]
        cols.append("%4.0f%%" % (sum(r[4] for r in xs) / max(len(xs), 1)))
        tn += len(xs); tr += sum(r[4] for r in xs)
    print("  {:.1f}〜{:<4} {} | {:7d}組 {:4.0f}%".format(lo_e, hi_e if hi_e < 99 else "", " ".join(cols), tn, tr / max(tn, 1)))
print("\n■ 確率の帯ごと(予想 → 実際に当たった割合)")
for lo_e, hi_e in ((0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1)):
    xs = [r for r in rows if lo_e <= r[1] < hi_e]
    if xs:
        print("  {:3.0%}〜{:3.0%} {:7d}組 予想 {:5.1%} → 実際 {:5.1%}".format(lo_e, hi_e, len(xs), sum(r[1] for r in xs) / len(xs),
                                                                     sum(1 for r in xs if r[4]) / len(xs)))

"""券種をレースごとに選ぶと回収率が上がるかの確かめ(中央、2024年1月〜。◎○▲△は月ごとの確率 keiba/model_wf.json.gz)
  python tools/experiments/bet_mix.py
  買い方(券種と点数)× 条件(◎の確率・◎の単勝オッズ・○の確率)の組を、2024・2025年で選び、2026年で確かめる。
  払戻・オッズは確定(◎のオッズは発走直前とほぼ同じとみなす)"""
import glob, gzip, json, os
from itertools import product
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = json.load(open(f))
    for rid, r in (d.get("races") or {}).items():
        if r.get("center") != "jra" or rid not in wf:
            continue
        res = r.get("result") or {}
        pay, od = res.get("payout") or {}, res.get("odds") or {}
        if not all(pay.get(k) for k in ("単勝", "馬連", "ワイド", "馬単", "3連複", "3連単")) or len(res.get("order") or []) < 3:
            continue
        ng = {str(h["umaban"]) for h in ((r.get("pred") or {}).get("horses") or []) if h.get("axis_ng")}
        hs = [u for u, p in sorted(wf[rid].items(), key=lambda x: -x[1]) if od.get(u)]
        ax = [u for u in hs if u not in ng]
        if len(hs) < 6 or len(ax) < 2:
            continue
        h, t = ax[0], ax[1]
        rest = [u for u in hs if u not in (h, t)]
        rows.append(dict(y=os.path.basename(f)[:4], h=h, t=t, a=rest[0], b=rest[1], c=rest[2], n=len(hs),
                         p1=wf[rid][h], p2=wf[rid][t], o1=od[h], o2=od[t], pay=pay))
print("レース", len(rows), {y: sum(1 for r in rows if r["y"] == y) for y in ("2024", "2025", "2026")})

key = lambda typ, c: "-".join(c if typ in ("馬単", "3連単") else sorted(c, key=int))
BETS = {
    "単勝◎": lambda r: [("単勝", (r["h"],))],
    "単勝○": lambda r: [("単勝", (r["t"],))],
    "ワイド◎-○": lambda r: [("ワイド", (r["h"], r["t"]))],
    "ワイド◎-○▲": lambda r: [("ワイド", (r["h"], r["t"])), ("ワイド", (r["h"], r["a"]))],
    "ワイド◎○▲BOX": lambda r: [("ワイド", (r["h"], r["t"])), ("ワイド", (r["h"], r["a"])), ("ワイド", (r["t"], r["a"]))],
    "馬連◎-○": lambda r: [("馬連", (r["h"], r["t"]))],
    "馬連◎-○▲": lambda r: [("馬連", (r["h"], r["t"])), ("馬連", (r["h"], r["a"]))],
    "馬単◎→○": lambda r: [("馬単", (r["h"], r["t"]))],
    "馬単◎⇄○": lambda r: [("馬単", (r["h"], r["t"])), ("馬単", (r["t"], r["h"]))],
    "3連複◎○▲": lambda r: [("3連複", (r["h"], r["t"], r["a"]))],
    "3連複◎○-▲△": lambda r: [("3連複", (r["h"], r["t"], x)) for x in (r["a"], r["b"])],
    "3連複◎○-▲△☆": lambda r: [("3連複", (r["h"], r["t"], x)) for x in (r["a"], r["b"], r["c"])],
    "3連複◎-○▲△": lambda r: [("3連複", (r["h"], x, y)) for x, y in ((r["t"], r["a"]), (r["t"], r["b"]), (r["a"], r["b"]))],
    "3連複◎○▲△BOX": lambda r: [("3連複", c) for c in ((r["h"], r["t"], r["a"]), (r["h"], r["t"], r["b"]), (r["h"], r["a"], r["b"]), (r["t"], r["a"], r["b"]))],
    "3連単◎→○→▲": lambda r: [("3連単", (r["h"], r["t"], r["a"]))],
    "3連単◎→○▲→○▲△": lambda r: [("3連単", (r["h"], x, y)) for x, y in ((r["t"], r["a"]), (r["t"], r["b"]), (r["a"], r["t"]), (r["a"], r["b"]))],
}


def ret(r, bet):
    bs = BETS[bet](r)
    return sum(r["pay"][typ].get(key(typ, c), 0) for typ, c in bs), 100 * len(bs)


for r in rows:
    r["ret"] = {b: ret(r, b) for b in BETS}


def roi(xs, bet):
    s = sum(r["ret"][bet][1] for r in xs)
    return sum(r["ret"][bet][0] for r in xs) / s * 100 if s else 0


def hit(xs, bet):
    return sum(1 for r in xs if r["ret"][bet][0]) / max(len(xs), 1) * 100


SEL = lambda r: r["y"] in ("2024", "2025")
print("\n■ 券種ごと(全レース / ◎80%以上)  2024 2025 2026")
for b in BETS:
    line = []
    for f in (lambda r: True, lambda r: r["p1"] >= .8):
        line.append(" ".join("{:4.0f}%".format(roi([r for r in rows if f(r) and r["y"] == y], b)) for y in ("2024", "2025", "2026"))
                    + " (的中{:3.0f}%)".format(hit([r for r in rows if f(r)], b)))
    print("  {:16s} {}  ‖  {}".format(b, *line))

# 条件: ◎の確率の下限 × ◎の単勝オッズの範囲 × ○の確率の下限
P1 = (0, .5, .6, .7, .8, .85)
OD = ((1, 2), (2, 3), (3, 5), (5, 10), (10, 999), (1, 3), (1, 5), (2, 5), (2, 999), (3, 999), (1, 999))
P2 = (0, .4, .5, .6)
rules = []
for b, t1, (lo, hi), t2 in product(BETS, P1, OD, P2):
    f = lambda r, t1=t1, lo=lo, hi=hi, t2=t2: r["p1"] >= t1 and lo <= r["o1"] < hi and r["p2"] >= t2
    s = [r for r in rows if SEL(r) and f(r)]
    if len(s) < 300:
        continue
    te = [r for r in rows if r["y"] == "2026" and f(r)]
    rules.append(dict(b=b, lab="◎{:.0%}+ ◎{}〜{}倍 ○{:.0%}+".format(t1, lo, hi if hi < 999 else "", t2), f=f,
                      n=len(s), sel=roi(s, b), nt=len(te), test=roi(te, b), ht=hit(te, b)))
rules.sort(key=lambda x: -x["sel"])
print("\n■ 2024・2025年で回収率の高い組(300レース以上)上位30 と、その2026年")
print("  {:16s} {:28s} {:>6s} {:>6s}   {:>6s} {:>6s} {:>5s}".format("買い方", "条件", "R", "選ぶ", "2026R", "2026", "的中"))
for x in rules[:30]:
    print("  {:16s} {:28s} {:6d} {:5.0f}%   {:6d} {:5.0f}% {:4.0f}%".format(x["b"], x["lab"], x["n"], x["sel"], x["nt"], x["test"], x["ht"]))
import numpy as np
a = np.array([x["sel"] for x in rules]); bb = np.array([x["test"] for x in rules])
print("\n  組 {} 個。選ぶ期間の回収率と2026年の回収率の相関 {:.2f}".format(len(rules), np.corrcoef(a, bb)[0, 1]))
for lo_, hi_ in ((0, 80), (80, 90), (90, 100), (100, 999)):
    m = (a >= lo_) & (a < hi_)
    if m.any():
        print("  選ぶ期間 {:3d}〜{:3d}%: {:4d}組 → 2026年の平均 {:5.1f}%".format(lo_, hi_ if hi_ < 999 else 999, int(m.sum()), bb[m].mean()))

# レースごとに選ぶ: 選ぶ期間の回収率が k 以上の組のうち、そのレースに当てはまる中で一番高い1つを買う
print("\n■ レースごとに券種を選ぶ(選ぶ期間で回収率 k 以上の組から、当てはまる一番高い1つ)")
for k in (85, 90, 95, 100, 105):
    use = [x for x in rules if x["sel"] >= k]
    for per, lab in ((SEL, "2024-25(選ぶ)"), (lambda r: r["y"] == "2026", "2026(確かめ)")):
        st = rt = n = 0
        for r in rows:
            if not per(r):
                continue
            x = next((x for x in use if x["f"](r)), None)
            if x:
                g, s = r["ret"][x["b"]]
                rt += g; st += s; n += 1
        print("  k={:3d} {:14s} {:5d}R 回収率 {:5.1f}%".format(k, lab, n, rt / st * 100 if st else 0))
print("\n■ 今の買い方(ワイド◎-○、◎80%以上)", " ".join("{} {:.0f}%".format(y, roi([r for r in rows if r["p1"] >= .8 and r["y"] == y], "ワイド◎-○")) for y in ("2024", "2025", "2026")))

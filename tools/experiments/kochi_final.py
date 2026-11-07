"""高知の最終レース(ファイナルレース)の買い方を確かめる(keiba/kochi。予想は月ごとの確率で作り直した「再現」)
  KEIBA_OUT=keiba/kochi python tools/experiments/kochi_final.py
  1) 最終レースで確率は当たっているか  2) 券種 × ◎の確率の下限 × ◎と○の差 を2024・2025年で選び、2026年で確かめる
  3) 同じことを高知の全レースで選んで、最終レースに当てはめる(数が多い分、選び方が安定する)"""
import glob, json, os
from itertools import product
import numpy as np
OUT = os.environ.get("KEIBA_OUT", "keiba/kochi")
YS = ("2024", "2025", "2026")
key = lambda typ, c: "-".join(str(x) for x in (c if typ in ("馬単", "3連単") else sorted(c)))
races = []
for f in sorted(glob.glob(OUT + "/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    rs = [r for r in (json.load(open(f)).get("races") or {}).values() if r.get("venue") == "高知" and r.get("rno")]
    if not rs:
        continue
    last = max(r["rno"] for r in rs)
    for r in rs:
        p, res = r.get("pred") or {}, r.get("result") or {}
        if not p.get("model") or len(res.get("order") or []) < 3 or not res.get("payout"):
            continue
        hs = {h["umaban"]: h for h in p["horses"] if h.get("p3") is not None}
        mk = {h.get("mark"): u for u, h in hs.items() if h.get("mark") in ("◎", "○", "▲", "△")}
        if not all(m in mk for m in "◎○▲△"):
            continue
        races.append(dict(y=d[:4], d=d, final=r["rno"] == last, h=mk["◎"], t=mk["○"], a=mk["▲"], b=mk["△"], hs=hs,
                          p1=hs[mk["◎"]]["p3"], p2=hs[mk["○"]]["p3"], top3=set(res["order"][:3]), pay=res["payout"], n=len(hs),
                          ext="距離延長" in (hs[mk["▲"]].get("axis_ng") or [])))
fin = [r for r in races if r["final"]]
print("高知 {}R(最終レース {}R) 年ごと {}".format(len(races), len(fin), {y: sum(1 for r in fin if r["y"] == y) for y in YS}))
pay = lambda r, typ, c: (r["pay"].get(typ) or {}).get(key(typ, c), 0)
BETS = {
    "単勝◎": lambda r: [("単勝", (r["h"],))],
    "複勝◎": lambda r: [("複勝", (r["h"],))],
    "ワイド◎-○": lambda r: [("ワイド", (r["h"], r["t"]))],
    "ワイド◎-▲": lambda r: [("ワイド", (r["h"], r["a"]))],
    "ワイド◎○▲BOX": lambda r: [("ワイド", c) for c in ((r["h"], r["t"]), (r["h"], r["a"]), (r["t"], r["a"]))],
    "馬連◎-○": lambda r: [("馬連", (r["h"], r["t"]))],
    "馬単◎→○": lambda r: [("馬単", (r["h"], r["t"]))],
    "3連複◎○▲": lambda r: [("3連複", (r["h"], r["t"], r["a"]))],
    "3連複◎○▲(延長ならワイド)": lambda r: [("ワイド", (r["h"], r["t"]))] if r["ext"] else [("3連複", (r["h"], r["t"], r["a"]))],
    "3連複◎○-▲△": lambda r: [("3連複", (r["h"], r["t"], x)) for x in (r["a"], r["b"])],
    "3連複◎-○▲△": lambda r: [("3連複", (r["h"], x, y)) for x, y in ((r["t"], r["a"]), (r["t"], r["b"]), (r["a"], r["b"]))],
    "3連単◎→○→▲": lambda r: [("3連単", (r["h"], r["t"], r["a"]))],
}
for r in races:
    r["ret"] = {b: (sum(pay(r, t, c) for t, c in f(r)), 100 * len(f(r))) for b, f in BETS.items()}
    r["has"] = {b: all(typ in r["pay"] for typ, _ in f(r)) for b, f in BETS.items()}


def roi(xs, b):
    xs = [r for r in xs if r["has"][b]]
    st = sum(r["ret"][b][1] for r in xs)
    return (sum(r["ret"][b][0] for r in xs) / st * 100) if st else float("nan")


def yr(xs, b):
    return " ".join("{:4.0f}".format(roi([r for r in xs if r["y"] == y], b)) if sum(1 for r in xs if r["y"] == y) >= 5 else "   -" for y in YS)


print("\n■ 1. 確率は当たっているか(3着内 実際 / 見込み)")
for lab, xs in (("最終レース", fin), ("全レース", races)):
    for m, k in (("◎", "h"), ("○", "t"), ("▲", "a")):
        a = sum(r[k] in r["top3"] for r in xs) / len(xs); e = sum(r["hs"][r[k]]["p3"] for r in xs) / len(xs)
        print("  {:6s} {} 実際 {:4.0%} / 見込み {:4.0%}".format(lab, m, a, e))

print("\n■ 2. 最終レース: 券種ごと(全部 / ◎60%以上 / ◎70%以上 / ◎80%以上)  [2024 2025 2026]")
for b in BETS:
    cells = []
    for t1 in (0, .6, .7, .8):
        xs = [r for r in fin if r["p1"] >= t1]
        cells.append("{:3d}R {:4.0f}% [{}]".format(len(xs), roi(xs, b), yr(xs, b)))
    print("  {:22s} {}".format(b, " | ".join(cells)))


def grid(pool, apply_to, min_sel):
    out = []
    for b, t1, g in product(BETS, (0, .5, .6, .7, .75, .8, .85), (-1, .1, .15, .2, .25)):
        f = lambda r: r["p1"] >= t1 and r["p1"] - r["p2"] >= g
        sel = [r for r in pool if r["y"] != "2026" and f(r)]
        if len(sel) < min_sel:
            continue
        te = [r for r in apply_to if r["y"] == "2026" and f(r)]
        te_all = [r for r in apply_to if f(r)]
        out.append((roi(sel, b), b, t1, g, len(sel), len(te), roi(te, b), yr(te_all, b),
                    sum(r["ret"][b][0] - r["ret"][b][1] for r in te if r["has"][b])))
    out.sort(key=lambda x: -(x[0] if x[0] == x[0] else -1))
    return out


for lab, pool, mn in (("最終レースで選ぶ(2年で60R以上)", fin, 60), ("高知の全レースで選び、最終レースに当てはめる(2年で300R以上)", races, 300)):
    res = grid(pool, fin, mn)
    print("\n■ {}  → 最終レースの2026年".format(lab))
    for x in res[:12]:
        print("  {:22s} ◎{:.0%}以上 差{:>4s}  選ぶ期間 {:4d}R {:4.0f}% / 最終レース2026 {:3d}R {:4.0f}% 収支{:+,}円 [最終レース {}]".format(
            x[1], x[2], "なし" if x[3] < 0 else "{:.0%}".format(x[3]), x[4], x[0], x[5], x[6], int(x[8]), x[7]))
    a = np.array([x[0] for x in res if x[6] == x[6]]); t = np.array([x[6] for x in res if x[6] == x[6]])
    if len(a) > 3:
        print("  組 {} 個。選ぶ期間と2026年の相関 {:.2f}。選ぶ期間で100%以上の組の2026年の平均 {:.0f}%".format(
            len(a), np.corrcoef(a, t)[0, 1], t[a >= 100].mean() if (a >= 100).any() else float("nan")))

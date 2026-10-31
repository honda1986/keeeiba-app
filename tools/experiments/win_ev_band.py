"""単勝を、アプリの期待値(確率×オッズ)や勝つ確率で選んで買ったら(中央、2024年1月〜。オッズは確定)
  python tools/experiments/win_ev_band.py
  勝つ確率はアプリと同じ: P ∝ 市場の確率^1.0 × (3着内の確率/(1-3着内の確率))^0.04(keiba/index.html の EVP)"""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
A, B = 1.0, 0.04
YS = ("2024", "2025", "2026")
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    y = os.path.basename(f)[:4]
    for r in (json.load(open(f)).get("races") or {}).values():
        p, res = r.get("pred") or {}, r.get("result") or {}
        od = {int(u): v for u, v in (res.get("odds") or {}).items() if v and v > 1}
        if r.get("center") != "jra" or not p.get("model") or len(od) < 5 or not res.get("order"):
            continue
        p3 = {h["umaban"]: h["p3"] for h in p["horses"] if h.get("p3") is not None}
        inv = sum(1 / v for v in od.values())
        z = {u: (1 / v / inv) ** A * (min(max(p3.get(u, .5), 1e-3), .99) / (1 - min(max(p3.get(u, .5), 1e-3), .99))) ** B for u, v in od.items()}
        s = sum(z.values())
        pay = (res.get("payout") or {}).get("単勝") or {}
        for u, v in od.items():
            P = z[u] / s
            rows.append(dict(y=y, P=P, ev=P * v, odds=v, ret=pay.get(str(u), 0), mark=next((h.get("mark") for h in p["horses"] if h["umaban"] == u), ""),
                             buy=p.get("verdict") in ("A", "B")))
print("馬 {} 頭".format(len(rows)))


def line(lab, xs):
    if not xs:
        print("  {:26s}     0頭".format(lab)); return
    ys = " ".join("{:4.0f}".format(sum(x["ret"] for x in xs if x["y"] == y) / max(1, sum(1 for x in xs if x["y"] == y))) if any(x["y"] == y for x in xs) else "   -" for y in YS)
    print("  {:26s} {:6d}頭 回収率 {:4.0f}% [{}] 的中 {:4.1%} 平均オッズ {:5.1f}".format(
        lab, len(xs), sum(x["ret"] for x in xs) / len(xs), ys, sum(1 for x in xs if x["ret"]) / len(xs), sum(x["odds"] for x in xs) / len(xs)))


print("\n■ アプリの期待値(確率×オッズ)ごと")
for lo, hi in ((0, .7), (.7, .75), (.75, .8), (.8, .85), (.85, .9), (.9, 1.0), (1.0, 1.3), (1.3, 99)):
    line("期待値 {:.2f}〜{}".format(lo, "{:.2f}".format(hi) if hi < 99 else ""), [x for x in rows if lo <= x["ev"] < hi])
print("  (いちばん大きい期待値 {:.2f})".format(max(x["ev"] for x in rows)))
print("\n■ 勝つ確率ごと(アプリの単勝の「確率」)")
for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)):
    line("勝つ確率 {:.0%}〜{:.0%}".format(lo, hi), [x for x in rows if lo <= x["P"] < hi])
line("勝つ確率 30%以上(全部)", [x for x in rows if x["P"] >= .3])
line("  うち◎", [x for x in rows if x["P"] >= .3 and x["mark"] == "◎"])
line("  うち買うレースの◎", [x for x in rows if x["P"] >= .3 and x["mark"] == "◎" and x["buy"]])

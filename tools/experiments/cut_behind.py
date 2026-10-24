"""◎○の次の馬(▲△)が「消し」のレースで、ワイド◎-○はどうか(中央、2024年1月〜、月ごとの確率)
  python tools/experiments/cut_behind.py"""
import glob, gzip, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = json.load(open(f))
    for rid, r in (d.get("races") or {}).items():
        res = r.get("result") or {}
        pay = (res.get("payout") or {}).get("ワイド")
        if r.get("center") != "jra" or rid not in wf or not pay or len(res.get("order") or []) < 3:
            continue
        hs = {str(h["umaban"]): h for h in ((r.get("pred") or {}).get("horses") or [])}
        pr = wf[rid]
        order = sorted(pr.items(), key=lambda kv: -kv[1])
        ok = [kv for kv in order if not (hs.get(kv[0]) or {}).get("axis_ng")]
        if len(order) < 5 or len(ok) < 2:
            continue
        (h, p1), (t, p2) = ok[0], ok[1]
        rest = [(u, p, bool((hs.get(u) or {}).get("cut"))) for u, p in order if u not in (h, t)]
        live = [p for u, p, c in rest if not c]                    # 消しでない残りの馬
        top3 = set(map(str, res["order"][:3]))
        rows.append(dict(y=os.path.basename(f)[:4], p1=p1, p2=p2, c3=rest[0][2], c4=rest[1][2],
                         live1=live[0] if live else 0.0, ret=pay.get("-".join(sorted((h, t), key=int)), 0),
                         both=h in top3 and t in top3))


def show(label, xs):
    cols = []
    for y in ("2024", "2025", "2026"):
        s = [r for r in xs if r["y"] == y]
        cols.append("%4.0f%%" % (sum(r["ret"] for r in s) / max(len(s), 1)))
    n = len(xs)
    print("  {:34s} {} | {:5d}R 的中{:4.0%} 計{:4.0f}%".format(label, " ".join(cols), n,
          sum(r["both"] for r in xs) / max(n, 1), sum(r["ret"] for r in xs) / max(n, 1)))


for title, cond in (("全レース", lambda r: True), ("◎50〜80%(今日の新潟1Rは◎59%)", lambda r: .5 <= r["p1"] < .8),
                    ("今の基準 ◎80%以上", lambda r: r["p1"] >= .8)):
    xs = [r for r in rows if cond(r)]
    print("■ ワイド◎-○ " + title + "   2024  2025  2026 | 計")
    show("すべて", xs)
    show("▲が消し", [r for r in xs if r["c3"]])
    show("▲が消しでない", [r for r in xs if not r["c3"]])
    show("▲△とも消し", [r for r in xs if r["c3"] and r["c4"]])
    show("▲△とも消し・残りの最上位が20%未満", [r for r in xs if r["c3"] and r["c4"] and r["live1"] < .2])
    show("▲△とも消しでない", [r for r in xs if not r["c3"] and not r["c4"]])

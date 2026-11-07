"""○が穴のレースを、3連複◎-○-全流し(◎○が3着内なら3頭目は何でも当たり。点数は出走頭数-2)で買ったら(中央。2024〜2026年)
  python tools/experiments/trio_flow.py
  ○が穴 = ○の確定人気が6番人気以下 / 4番人気以下 / ○の3着内の確率30%未満。買うレースと全レース(◎の確率で分ける)で、
  今の買い方・ワイド◎-○・馬連◎-○ とくらべる(1レースあたりの賭け金は買い方ごとに違うので、回収率と収支で見る)"""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for r in (json.load(open(f)).get("races") or {}).values():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or not p.get("honmei") or len(res.get("order") or []) < 3:
            continue
        h, t = p["honmei"], p["taikou"]
        n = len([v for v in (res.get("odds") or {}).values() if v])
        if n < 5:
            continue
        pay = res["payout"]; top3 = res["order"][:3]
        both = h in top3 and t in top3
        hp = {x["umaban"]: x.get("p3") or 0 for x in p["horses"]}
        cur = None
        if p.get("verdict") in ("A", "B") and p.get("bets"):
            b = p["bets"][0]; cur = sum((pay.get(b["type"]) or {}).get(srt(c), 0) for c in b["combos"])
        rows.append(dict(y=d[:4], buy=cur is not None, cur=cur, n=n, p1=hp.get(h, 0), p2=hp.get(t, 0),
                         nk=int((res.get("ninki") or {}).get(str(t)) or 99), both=both,
                         flow=(pay.get("3連複") or {}).get(srt(top3), 0) if both else 0,
                         wide=(pay.get("ワイド") or {}).get(srt([h, t]), 0), umaren=(pay.get("馬連") or {}).get(srt([h, t]), 0)))
print("中央の予想したレース {}R(買うレース {}R)".format(len(rows), sum(r["buy"] for r in rows)))


def line(name, m):
    if not m:
        return
    def roi(ms, key, cost):
        b = sum(cost(r) for r in ms); return sum(r[key] for r in ms) / b * 100 if b else 0
    def yrs(key, cost):
        return " ".join("{:4.0f}".format(roi([r for r in m if r["y"] == y], key, cost)) if sum(r["y"] == y for r in m) >= 8 else "   -" for y in YS)
    flow_cost = lambda r: 100 * (r["n"] - 2); one = lambda r: 100
    fb = sum(flow_cost(r) for r in m)
    print("  {:30s} {:4d}R ◎○が3着内 {:3.0%}  全流し {:4.0f}% [{}] {:+9,.0f}円 (1R平均{:,.0f}円) | ワイド {:4.0f}% [{}] | 馬連 {:4.0f}%{}".format(
        name, len(m), np.mean([r["both"] for r in m]), roi(m, "flow", flow_cost), yrs("flow", flow_cost),
        sum(r["flow"] for r in m) - fb, fb / len(m), roi(m, "wide", one), yrs("wide", one), roi(m, "umaren", one),
        " | 今の買い方 {:4.0f}% [{}]".format(roi(m, "cur", one), yrs("cur", one)) if all(r["buy"] for r in m) else ""))


ANA = (("○が6番人気以下", lambda r: r["nk"] >= 6), ("○が4番人気以下", lambda r: r["nk"] >= 4), ("○の確率30%未満", lambda r: r["p2"] < .3))
print("\n■ 買うレース(◎80%以上・◎と○の差15%以上)  回収率 [2024 2025 2026]")
B = [r for r in rows if r["buy"]]
line("買うレース全部", B)
for name, fn in ANA:
    line(name, [r for r in B if fn(r)])
for lo, hi, lab in ((.8, 1.01, "◎80%以上"), (.6, .8, "◎60〜80%"), (0, .6, "◎60%未満")):
    print("\n■ 全レース(見送りも)・{}".format(lab))
    A = [r for r in rows if lo <= r["p1"] < hi]
    line("全部", A)
    for name, fn in ANA:
        line(name, [r for r in A if fn(r)])

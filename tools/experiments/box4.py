"""◎○▲△の4頭ボックスで買ったら(中央。2024〜2026年)
  python tools/experiments/box4.py
  3連複ボックス(4点)・ワイドボックス(6点)・馬連ボックス(6点)・3連単ボックス(24点)を、今の買い方(3連複◎-○-▲の1点。
  ▲が延長ならワイド◎-○)とくらべる。▲△は◎○の次に確率が高い2頭(アプリと同じ)。買うレースと全レース(◎の確率で分ける)"""
import glob, json, os
from itertools import combinations
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for r in (json.load(open(f)).get("races") or {}).values():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or not p.get("honmei") or len(p.get("aite") or []) < 2 or len(res.get("order") or []) < 3:
            continue
        box = [p["honmei"], p["taikou"]] + p["aite"][:2]
        pay = res["payout"]; top3 = res["order"][:3]
        inb = set(top3) <= set(box)
        cur = None
        if p.get("verdict") in ("A", "B") and p.get("bets"):
            b = p["bets"][0]; cur = sum((pay.get(b["type"]) or {}).get(srt(c), 0) for c in b["combos"])
        hp = {x["umaban"]: x.get("p3") or 0 for x in p["horses"]}
        rows.append(dict(y=d[:4], buy=cur is not None, cur=cur, p1=hp.get(p["honmei"], 0),
                         trio=(pay.get("3連複") or {}).get(srt(top3), 0) if inb else 0,
                         tri=(pay.get("3連単") or {}).get("-".join(map(str, top3)), 0) if inb else 0,
                         wide=sum((pay.get("ワイド") or {}).get(srt(c), 0) for c in combinations(box, 2)),
                         umaren=sum((pay.get("馬連") or {}).get(srt(c), 0) for c in combinations(box, 2)),
                         hit3=inb))
print("中央の予想したレース {}R(買うレース {}R)".format(len(rows), sum(r["buy"] for r in rows)))
BETS = (("3連複ボックス 4点", "trio", 4), ("ワイドボックス 6点", "wide", 6), ("馬連ボックス 6点", "umaren", 6), ("3連単ボックス 24点", "tri", 24))


def show(title, m, cur=False):
    print("\n■ {}  {}R  回収率 [2024 2025 2026] / 収支 / 当たり".format(title, len(m)))
    items = ((("今の買い方 1点", "cur", 1),) if cur else ()) + BETS
    for name, key, pts in items:
        v = lambda ms: sum(r[key] for r in ms) / (100 * pts * len(ms)) * 100 if ms else 0
        hit = np.mean([r[key] > 0 for r in m])
        big = sorted((r[key] for r in m), reverse=True)
        ex3 = sum(big[3:]) / (100 * pts * len(m)) * 100
        print("  {:20s} {:4.0f}% [{}]  {:+10,.0f}円  {:3.0%}  (大きい3つを除くと {:3.0f}%)".format(
            name, v(m), " ".join("{:4.0f}".format(v([r for r in m if r["y"] == y])) for y in YS),
            sum(r[key] for r in m) - 100 * pts * len(m), hit, ex3))


show("買うレース(◎80%以上・◎と○の差15%以上)", [r for r in rows if r["buy"]], cur=True)
for lo, hi, lab in ((.8, 1.01, "◎80%以上(見送りも)"), (.6, .8, "◎60〜80%"), (0, .6, "◎60%未満")):
    show("全レース・" + lab, [r for r in rows if lo <= r["p1"] < hi])

"""単勝◎の的中率・回収率を、◎の確率・人気・オッズ・差・クラスで分けて年ごとに出す(中央、2024年1月〜)
  python tools/experiments/tan_honmei.py
  ◎は月ごとの確率(keiba/model_wf.json.gz)で、軸にできる馬のうち確率1番。払戻・人気・オッズは確定"""
import glob, gzip, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = json.load(open(f))
    for rid, r in (d.get("races") or {}).items():
        if r.get("center") != "jra" or rid not in wf:
            continue
        res = r.get("result") or {}
        pay = (res.get("payout") or {}).get("単勝")
        odds, ninki = res.get("odds") or {}, res.get("ninki") or {}
        if not pay or not odds or not res.get("order"):
            continue
        pr = wf[rid]
        ng = {str(h["umaban"]) for h in ((r.get("pred") or {}).get("horses") or []) if h.get("axis_ng")}
        hs = sorted(((u, p) for u, p in pr.items() if u in odds), key=lambda x: -x[1])
        if len(hs) < 5:
            continue
        ax = [x for x in hs if x[0] not in ng] or hs
        hon = ax[0][0]
        rows.append(dict(y=os.path.basename(f)[:4], hon=hon, p1=pr[hon], p2=ax[1][1] if len(ax) > 1 else 0,
                         odds=odds.get(hon), ninki=ninki.get(hon), pay=pay, cls=r.get("cls"),
                         fav=min(ninki, key=lambda u: ninki[u]) if ninki else None))


def show(label, xs, pick=lambda r: r["hon"]):
    cols = []
    for y in ("2024", "2025", "2026", None):
        s = [r for r in xs if y is None or r["y"] == y]
        n = len(s)
        hit = sum(1 for r in s if pick(r) in r["pay"])
        ret = sum(r["pay"].get(pick(r), 0) for r in s)
        cols.append(f"{n:5d}R {hit / max(n, 1) * 100:4.0f}% {ret / max(n, 1):4.0f}%")
    print(f"{label:28s}" + " | ".join(cols))


print(" " * 28 + "2024(R 的中 回収)    | 2025                | 2026                | 計")
show("1番人気(市場)", rows, lambda r: r["fav"])
show("単勝◎ 全レース", rows)
show("今の買う基準 ◎60+ ○40-50", [r for r in rows if r["p1"] >= .6 and .4 <= r["p2"] < .5])
for t in (.5, .6, .7, .8):
    show(f"◎{int(t * 100)}%以上", [r for r in rows if r["p1"] >= t])
for a, b in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 99)):
    show(f"◎が{a}〜{b}番人気", [r for r in rows if r["ninki"] and a <= r["ninki"] <= b])
for a, b in ((1, 2), (2, 3), (3, 5), (5, 8), (8, 15), (15, 999)):
    show(f"◎が{a}〜{b}倍", [r for r in rows if r["odds"] and a <= r["odds"] < b])
for g in (.1, .2, .3):
    show(f"◎−○ {int(g * 100)}ポイント以上", [r for r in rows if r["p1"] - r["p2"] >= g])
show("◎60%以上・2〜3番人気", [r for r in rows if r["p1"] >= .6 and r["ninki"] in (2, 3)])
for c, name in ((6, "G3"), (7, "G2"), (8, "G1")):
    show(f"◎60%以上・{name}", [r for r in rows if r["p1"] >= .6 and r["cls"] == c])

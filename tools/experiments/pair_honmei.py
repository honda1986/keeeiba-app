"""ワイド・馬連の◎-○1点の的中率・回収率を、◎○の確率・人気・差・頭数・クラスで分けて年ごとに出す(中央、2024年1月〜)
  python tools/experiments/pair_honmei.py
  ◎○は月ごとの確率(keiba/model_wf.json.gz)で、軸にできる馬のうち確率の上位2頭。払戻・人気は確定"""
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
        pay, ninki = res.get("payout") or {}, res.get("ninki") or {}
        if not pay.get("ワイド") or not pay.get("馬連") or not ninki or len(res.get("order") or []) < 3:
            continue
        pr = wf[rid]
        ng = {str(h["umaban"]) for h in ((r.get("pred") or {}).get("horses") or []) if h.get("axis_ng")}
        hs = sorted(((u, p) for u, p in pr.items() if u in ninki), key=lambda x: -x[1])
        ax = [x for x in hs if x[0] not in ng] or hs
        if len(hs) < 5 or len(ax) < 2:
            continue
        (h, p1), (t, p2) = ax[0], ax[1]
        p3 = max(p for u, p in hs if u not in (h, t))
        fav = sorted(ninki, key=lambda u: ninki[u])[:2]
        rows.append(dict(y=os.path.basename(f)[:4], m=os.path.basename(f)[4:6], pair=(h, t), fav=tuple(fav),
                         p1=p1, p2=p2, gap=p2 - p3, n=len(hs), cls=r.get("cls"), surf=r.get("surface"),
                         nk=sorted((ninki[h], ninki[t])), pay=pay))


def ret(r, typ, pair):
    return r["pay"][typ].get("-".join(sorted(pair, key=int)), 0)


def show(label, xs, pair=lambda r: r["pair"]):
    out = []
    for typ in ("ワイド", "馬連"):
        cols = []
        for y in ("2024", "2025", "2026", None):
            s = [r for r in xs if y is None or r["y"] == y]
            n = len(s)
            cols.append(f"{sum(ret(r, typ, pair(r)) for r in s) / max(n, 1):4.0f}%")
        hit = sum(1 for r in xs if ret(r, typ, pair(r))) / max(len(xs), 1) * 100
        out.append(" ".join(cols) + f" (的中{hit:3.0f}%)")
    print(f"{label:26s}{len(xs):5d}R  " + "  ‖  ".join(out))


print(" " * 26 + "       ワイド 2024 2025 2026  計             ‖  馬連 2024 2025 2026  計")
show("1・2番人気(市場)", rows, lambda r: r["fav"])
show("◎-○ 全レース", rows)
show("今の買う基準 ◎60+ ○40-50", [r for r in rows if r["p1"] >= .6 and .4 <= r["p2"] < .5])
for t in (.5, .6, .65, .7):
    show(f"○{int(t * 100)}%以上", [r for r in rows if r["p2"] >= t])
for t in (.6, .7, .8):
    show(f"◎{int(t * 100)}%以上", [r for r in rows if r["p1"] >= t])
for g in (.1, .2, .3):
    show(f"○と3番手の差 {int(g * 100)}ポイント以上", [r for r in rows if r["gap"] >= g])
show("◎○が1・2番人気", [r for r in rows if r["nk"] == [1, 2]])
show("◎○に1番人気がいない", [r for r in rows if 1 not in r["nk"]])
show("◎○のどちらかが4番人気以下", [r for r in rows if r["nk"][1] >= 4])
for a, b in ((5, 10), (11, 14), (15, 18)):
    show(f"{a}〜{b}頭", [r for r in rows if a <= r["n"] <= b])
show("芝", [r for r in rows if r["surf"] == "芝"])
show("ダ", [r for r in rows if r["surf"] == "ダ"])
for c, name in ((1, "新馬・未勝利"), (2, "1勝"), (3, "2勝"), (4, "3勝"), (5, "OP・L"), (6, "G3"), (7, "G2"), (8, "G1")):
    show(name, [r for r in rows if r["cls"] == c])
show("冬(1〜2月)を除く", [r for r in rows if r["m"] not in ("01", "02")])

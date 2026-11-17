"""過去のG1で ◎○▲ はどれくらい当たったか(中央 2022〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/g1_marks.py
  PRED … fresh_years.py(YEARS=2022,...,2026)の3着内の確率(その年を除いて学んだモデル。5年とも同じ作り方)。
  ◎○ = 軸にできる馬の確率上位2頭、▲ = 残りの確率1位(アプリと同じ印の付け方)。G1 = クラス 8(障害は入らない)。
  あわせて、アプリに出ている印(日別ファイルの予想。2024年からはその月より前だけで学んだモデル、2022・2023年はモデルなしの評価)も数える。
  比べるもの: G2・G3、全レース。1番人気・2番人気・3番人気(確定オッズ)"""
import json, os, pickle, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba as K
YS = ("2022", "2023", "2024", "2025", "2026")
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
p3 = np.load(os.environ["PRED"])
info = {rc["id"]: rc for d, rc in races}
by = defaultdict(list)
for i, r in enumerate(rows):
    if not np.isnan(p3[i]):
        by[r["rid"]].append(i)
P3 = {}
for rid, ix in by.items():
    tot = p3[ix].sum(); want = min(3, len(ix))
    for i in ix:
        P3[(rid, rows[i]["u"])] = min(0.99, p3[i] * want / tot)
jg = T.group(T.of_center(rows, "jra"))
pick = T.pick_model(0.0, -1.0, P3, True)
buy = T.pick_model(0.0, -1.0, P3, True, t1=0.80, g12=0.15)
srt = lambda us: "-".join(str(x) for x in sorted(us))


def cls_of(rid):
    """日別ファイルのクラス(8 = G1、7 = G2、6 = G3)"""
    d = K.load_day(rows_date[rid])
    return (d["races"].get(rid) or {}).get("cls")


rows_date = {r["rid"]: r["date"] for r in rows}
_cls = {}
for d in sorted({rows_date[rid] for rid in jg}):
    try:
        day = json.load(open(os.path.join(K.OUT, "days", d + ".json"), encoding="utf-8"))
    except OSError:
        continue
    for rid, rc in day["races"].items():
        _cls[rid] = (rc.get("cls"), rc.get("name"), rc.get("pred"), rc.get("result"))

R = []
for rid, rs in jg.items():
    if rid not in info or rid not in _cls or (rid, rs[0]["u"]) not in P3:
        continue
    s = pick(rid, rs, None)
    res = info[rid]["result"]; order = res.get("order") or []
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    if not s or not s[2] or len(order) < 3 or len(od) < 5:
        continue
    hon, tai, rest = s; a = rest[0]
    pay = res.get("payout") or {}
    favs = sorted(od, key=lambda u: (od[u], u))
    cls, name, apred, _ = _cls[rid]
    top3 = set(order[:3])
    R.append(dict(rid=rid, y=rs[0]["date"][:4], d=rs[0]["date"], name=name, cls=cls, hon=hon, tai=tai, a=a, order=order, top3=top3,
                  ph=P3[(rid, hon)], pt=P3[(rid, tai)], pa=P3[(rid, a)], buy=buy(rid, rs, None) is not None,
                  tan=od[hon] * 100 if order[0] == hon else 0, wide=(pay.get("ワイド") or {}).get(srt((hon, tai)), 0),
                  umaren=(pay.get("馬連") or {}).get(srt((hon, tai)), 0), trio=(pay.get("3連複") or {}).get(srt((hon, tai, a)), 0),
                  nk=favs.index(hon) + 1 if hon in favs else None, f=favs[:3], n=len(od), apred=apred))
print("中央 {}R / G1 {}R / G2 {}R / G3 {}R".format(len(R), sum(r["cls"] == 8 for r in R), sum(r["cls"] == 7 for r in R), sum(r["cls"] == 6 for r in R)))


def pm(p, n):
    return 196 * (p * (1 - p) / max(n, 1)) ** 0.5


def show(title, sel):
    n = len(sel)
    if not n:
        return
    f = lambda k: np.mean([k(r) for r in sel])
    print("\n■ {}: {}R".format(title, n))
    print("  ◎ 1着 {:5.1%}(±{:.0f}) 3着内 {:5.1%}(見込み {:4.1%}) / ○ 3着内 {:5.1%}(見込み {:4.1%}) / ▲ 3着内 {:5.1%}(見込み {:4.1%})".format(
        f(lambda r: r["order"][0] == r["hon"]), pm(f(lambda r: r["order"][0] == r["hon"]), n), f(lambda r: r["hon"] in r["top3"]), f(lambda r: r["ph"]),
        f(lambda r: r["tai"] in r["top3"]), f(lambda r: r["pt"]), f(lambda r: r["a"] in r["top3"]), f(lambda r: r["pa"])))
    for nm, hit, pay in (("単勝 ◎", lambda r: r["order"][0] == r["hon"], "tan"), ("馬連 ◎-○", lambda r: set(r["order"][:2]) == {r["hon"], r["tai"]}, "umaren"),
                         ("ワイド ◎-○", lambda r: r["hon"] in r["top3"] and r["tai"] in r["top3"], "wide"),
                         ("3連複 ◎-○-▲", lambda r: {r["hon"], r["tai"], r["a"]} == r["top3"], "trio")):
        h = f(hit)
        yr = " ".join("{:3.0f}".format(np.mean([r[pay] for r in sel if r["y"] == y])) if any(r["y"] == y for r in sel) else "  -" for y in YS)
        print("  {:12s} 的中 {:5.1%}(±{:4.1f}) {:3d}回 回収率 {:6.1f}% [{}]".format(nm, h, pm(h, n), int(round(h * n)), f(lambda r: r[pay]), yr))
    print("  ◎が1番人気 {:.0%} / ◎の確率 平均 {:.0%} / 1番人気の3着内 {:.1%}・2番人気 {:.1%}・3番人気 {:.1%} / 今の買う基準(◎80%以上・差15%以上)を満たす {}R".format(
        f(lambda r: r["nk"] == 1), f(lambda r: r["ph"]), f(lambda r: r["f"][0] in r["top3"]), f(lambda r: r["f"][1] in r["top3"]), f(lambda r: r["f"][2] in r["top3"]),
        sum(r["buy"] for r in sel)))


G1 = sorted([r for r in R if r["cls"] == 8], key=lambda r: r["d"])
show("G1", G1)
show("G2", [r for r in R if r["cls"] == 7])
show("G3", [r for r in R if r["cls"] == 6])
show("全レース", R)
print("\n■ G1 の年ごと")
for y in YS:
    s = [r for r in G1 if r["y"] == y]
    if s:
        print("  {} {:2d}R ◎1着 {} ◎3着内 {} ○3着内 {} ▲3着内 {} ワイド◎-○ {} 3連複◎○▲ {}".format(
            y, len(s), sum(r["order"][0] == r["hon"] for r in s), sum(r["hon"] in r["top3"] for r in s), sum(r["tai"] in r["top3"] for r in s),
            sum(r["a"] in r["top3"] for r in s), sum(r["wide"] > 0 for r in s), sum(r["trio"] > 0 for r in s)))
print("\n■ G1 のレースごと(◎-○-▲ / 結果 / ◎の人気)")
for r in G1:
    mk = lambda u: "◎" if u == r["hon"] else "○" if u == r["tai"] else "▲" if u == r["a"] else ""
    print("  {} {:14s} ◎{:2d}({:2.0f}%・{}人) ○{:2d} ▲{:2d} | 結果 {} | {}{}".format(
        r["d"], r["name"] or "", r["hon"], 100 * r["ph"], r["nk"], r["tai"], r["a"], "-".join("{}{}".format(u, mk(u)) for u in r["order"][:3]),
        "3連複的中 {}円".format(r["trio"]) if r["trio"] else "ワイド的中 {}円".format(r["wide"]) if r["wide"] else "", " 買う" if r["buy"] else ""))
# アプリに出ている印(日別ファイルの予想)
print("\n■ アプリに出ている印で数える(日別ファイルの予想。◎○▲は馬の印)")
for lab, ys in (("2022・2023年(モデルなしの評価)", ("2022", "2023")), ("2024年〜(その月より前だけで学んだモデル)", ("2024", "2025", "2026"))):
    n = hon1 = hon3 = tai3 = a3 = wd = tr = 0
    for r in G1:
        if r["y"] not in ys:
            continue
        hs = {h.get("mark"): h["umaban"] for h in ((r["apred"] or {}).get("horses") or []) if h.get("mark") in ("◎", "○", "▲")}
        if len(hs) < 3:
            continue
        n += 1; hon1 += r["order"][0] == hs["◎"]; hon3 += hs["◎"] in r["top3"]; tai3 += hs["○"] in r["top3"]; a3 += hs["▲"] in r["top3"]
        wd += hs["◎"] in r["top3"] and hs["○"] in r["top3"]; tr += {hs["◎"], hs["○"], hs["▲"]} == r["top3"]
    if n:
        print("  {}: {}R ◎1着 {}({:.0%}) ◎3着内 {}({:.0%}) ○3着内 {}({:.0%}) ▲3着内 {}({:.0%}) ◎○が3着内 {}({:.0%}) ◎○▲が3着内 {}({:.0%})".format(
            lab, n, hon1, hon1 / n, hon3, hon3 / n, tai3, tai3 / n, a3, a3 / n, wd, wd / n, tr, tr / n))

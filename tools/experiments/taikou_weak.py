"""○が穴すぎないか: ○は「軸にできる馬(延長・条件替わりなどでない)」の確率の2番目なので、確率の高い馬が軸不可だと
確率の低い馬が○になる。買うレース(中央。2024〜2026年)を、○の確率の順位・○の確率・○の人気で分けて、今の買い方の成績を出す
  python tools/experiments/taikou_weak.py"""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or p.get("verdict") not in ("A", "B") or not p.get("bets") or len(res.get("order") or []) < 3:
            continue
        hs = sorted(p["horses"], key=lambda h: -(h.get("p3") or 0))
        us = [h["umaban"] for h in hs]
        h, t = p["honmei"], p["taikou"]
        b = p["bets"][0]
        ret = sum((res["payout"].get(b["type"]) or {}).get(srt(c), 0) for c in b["combos"])
        tp = next(x["p3"] for x in hs if x["umaban"] == t)
        rows.append(dict(y=d[:4], rid=rid, rank=us.index(t) + 1, tp=tp, nk=int((res.get("ninki") or {}).get(str(t)) or 99),
                         ret=ret, typ=b["type"], t3=t in res["order"][:3], tpp=tp))


def show(name, m):
    v = [r["ret"] for r in m]
    if not v:
        return
    print("  {:28s} {:4d}R 回収率 {:4.0f}% [{}]  ○が3着内 {:3.0%}(見込み {:3.0%})".format(name, len(v), np.mean(v),
          " ".join("{:4.0f}".format(np.mean([r["ret"] for r in m if r["y"] == y])) if sum(r["y"] == y for r in m) >= 5 else "   -" for y in YS),
          np.mean([r["t3"] for r in m]), np.mean([r["tp"] for r in m])))


print("買うレース {}R".format(len(rows)))
show("全部", rows)
print("\n■ ○の確率の順位(◎を除いて何番目の馬か。2=確率どおり)")
for k in (2, 3, 4, 5):
    show("○が確率{}位{}".format(k, "以下" if k == 5 else ""), [r for r in rows if (r["rank"] >= 5 if k == 5 else r["rank"] == k)])
print("\n■ ○の3着内の確率")
for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1)):
    show("○ {:.0%}〜{:.0%}".format(lo, hi), [r for r in rows if lo <= r["tp"] < hi])
print("\n■ ○の人気(確定)")
for lo, hi in ((1, 3), (3, 6), (6, 9), (9, 99)):
    show("○ {}〜{}番人気".format(lo, hi - 1 if hi < 99 else ""), [r for r in rows if lo <= r["nk"] < hi])
print("\n■ 券種ごと(3連複◎○▲ / ▲延長でワイド◎-○)と ○の確率")
for typ in ("3連複", "ワイド"):
    for lo, hi in ((0, .3), (.3, .45), (.45, 1)):
        show("{} ○ {:.0%}〜{:.0%}".format(typ, lo, hi), [r for r in rows if r["typ"] == typ and lo <= r["tp"] < hi])


# ■ ◎○を選ぶときに「軸にしない」条件を変えたら(保存してある各馬の確率と軸不可の理由から作り直す。買う基準・買い方は今と同じ)
import sys
sys.path.insert(0, R)
import keiba_logic as L
SEL = L.model_select("jra")
ALL = {"距離延長", "条件替わり", "能力不明", "長期休養明け"}


def sim(excl_hon, excl_tai, max_rank=99):
    out = []
    for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
        d = os.path.basename(f)[:8]
        for r in (json.load(open(f)).get("races") or {}).values():
            p, res = r.get("pred") or {}, r.get("result") or {}
            if r.get("center") != "jra" or not p.get("model") or not p.get("honmei") or len(res.get("order") or []) < 3:
                continue
            hs = sorted(p["horses"], key=lambda h: -(h.get("p3") or 0))
            if len(hs) < 5:
                continue
            ng = lambda h, ex: set(h.get("axis_ng") or ()) & ex
            hon = next((h for h in hs if not ng(h, excl_hon)), None)
            tai = next((h for h in hs if h is not hon and not ng(h, excl_tai)), None)
            if not hon or not tai:
                continue
            rest = [h for h in hs if h is not hon and h is not tai]
            if not L.select_ok(SEL, hon["p3"], tai["p3"], rest[0]["p3"], len(hs)):
                continue
            if [h for h in hs if h is not hon].index(tai) + 1 > max_rank:
                continue
            a = rest[0]
            if "距離延長" in (a.get("axis_ng") or ()):
                ret = (res["payout"].get("ワイド") or {}).get(srt([hon["umaban"], tai["umaban"]]), 0)
            else:
                ret = (res["payout"].get("3連複") or {}).get(srt([hon["umaban"], tai["umaban"], a["umaban"]]), 0)
            out.append((d[:4], ret))
    return out


print("\n■ 軸にしない条件を変えたら(買う基準・買い方は今と同じ)  [2024 2025 2026] 収支は1点100円")
for name, eh, et, mr in (("今(延長・条件替わり・能力不明・長期休養明け)", ALL, ALL, 99),
                         ("○だけ確率3位までに限る(4位以下なら見送り)", ALL, ALL, 2 + 1),
                         ("○だけ確率4位までに限る", ALL, ALL, 4),
                         ("条件替わりは軸にしてよい", ALL - {"条件替わり"}, ALL - {"条件替わり"}, 99),
                         ("○は延長だけ外す", ALL, {"距離延長"}, 99),
                         ("○は何も外さない(確率の2番目)", ALL, set(), 99)):
    v = sim(eh, et, mr)
    print("  {:40s} {:4d}R {:4.0f}% [{}]  {:+,}円".format(name, len(v), np.mean([x for _, x in v]),
          " ".join("{:4.0f}".format(np.mean([x for y, x in v if y == yy])) for yy in YS), sum(x for _, x in v) - 100 * len(v)))

"""3着内の確率が 80%以上の馬の単勝を買ったらどうなるか(中央 2022〜2026年。その年を除いて学んだモデル)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/tan_p80.py
  PRED … fresh_years.py(YEARS=2022,...,2026)の3着内の確率。レースの中で合計3にそろえ 0.99 で頭打ち(アプリ・pick_model と同じ)。
  払戻は確定の単勝の払戻(無ければ確定オッズ)。境目を 70〜90% で動かし、オッズの帯・人気・頭数でも分ける
  (複勝は、記録に払戻が残っているレースが少ないので出さない)"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
info = {rc["id"]: rc for d, rc in races}
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
p3 = np.load(os.environ["PRED"])
by = defaultdict(list)
for i, r in enumerate(rows):
    if not np.isnan(p3[i]) and r["f"].get("center") == 0:
        by[r["rid"]].append(i)
H = []
for rid, ix in by.items():
    rc = info.get(rid)
    if not rc:
        continue
    res = rc["result"]; order = res.get("order") or []; pay = res.get("payout") or {}
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    if len(order) < 3 or len(od) < 5:
        continue
    tot = p3[ix].sum(); want = min(3, len(ix))
    favs = sorted(od, key=lambda u: (od[u], u))
    ps = sorted((min(0.99, p3[i] * want / tot) for i in ix), reverse=True)
    for i in ix:
        u = rows[i]["u"]
        if u not in od:
            continue
        p = min(0.99, p3[i] * want / tot)
        top = p >= ps[0] and len(ps) > 1
        tan = (pay.get("単勝") or {}).get(str(u)); fuku = (pay.get("複勝") or {}).get(str(u))
        H.append(dict(y=rows[i]["date"][:4], p=p, odds=od[u], nk=favs.index(u) + 1, n=len(od),
                      tan=(tan if tan is not None else (od[u] * 100 if order[0] == u else 0)) if order[0] == u else 0,
                      fuku=(fuku or 0) if u in order[:3] else 0, in3=u in order[:3],
                      top=top, gap=(p - ps[1]) if top else None))
print("中央 {}頭 ({:.0f}分)".format(len(H), (time.time() - t0) / 60), flush=True)


def show(name, sel, key="tan"):
    if len(sel) < 30:
        print("  {:34s} {:5d}頭(少ない)".format(name, len(sel))); return
    pay = np.array([x[key] for x in sel], float); y = np.array([x["y"] for x in sel])
    hit = np.mean([x[key] > 0 for x in sel])
    top = np.sort(pay)[-3:].sum()
    print("  {:34s} {:5d}頭 的中 {:5.1%} 平均オッズ {:4.1f}倍 回収率 {:5.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
        name, len(sel), hit, np.median([x["odds"] for x in sel]), pay.mean(), (pay.sum() - top) / len(sel),
        " ".join("{:3.0f}".format(pay[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))


print("\n■ 単勝: 3着内の確率の境目ごと  [ ] は 2022〜2026年の回収率(平均オッズは中央値)")
for th in (0.70, 0.75, 0.80, 0.85, 0.90):
    show("{:.0%}以上".format(th), [x for x in H if x["p"] >= th])
show("(物差し)1番人気", [x for x in H if x["nk"] == 1])
show("(物差し)全部の馬", H)
s80 = [x for x in H if x["p"] >= 0.80]
print("\n  80%以上の馬の 3着内に来た割合 {:.1%}(見込みの平均 {:.1%})/ 1着 {:.1%} / 1番人気 {:.1%}".format(
    np.mean([x["in3"] for x in s80]), np.mean([x["p"] for x in s80]), np.mean([x["tan"] > 0 for x in s80]), np.mean([x["nk"] == 1 for x in s80])))
print("\n■ 単勝 80%以上を、オッズ・人気・頭数で分ける")
for lo, hi in ((1, 1.5), (1.5, 2), (2, 3), (3, 1e9)):
    show("オッズ {:.1f}〜{}倍".format(lo, "{:.1f}".format(hi) if hi < 1e9 else ""), [x for x in s80 if lo <= x["odds"] < hi])
show("1番人気", [x for x in s80 if x["nk"] == 1])
show("2番人気以下", [x for x in s80 if x["nk"] >= 2])
show("12頭以下", [x for x in s80 if x["n"] <= 12])
show("13頭以上", [x for x in s80 if x["n"] >= 13])
# 複勝は、記録に払戻が残っているレースが少ない(2026年の一部だけ)ので出さない

# 2番目に確率の高い馬(○)との差で分ける(◎ = レースで確率がいちばん高い馬)
GAPS = ((0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0))


def gap_table(title, sel):
    print("\n■ " + title)
    for lo, hi in GAPS:
        g = [x for x in sel if lo <= x["gap"] < hi]
        if len(g) < 30:
            print("  差 {:2.0f}〜{:3.0f}%  {:5d}頭(少ない)".format(100 * lo, 100 * hi, len(g))); continue
        show("差 {:2.0f}〜{:3.0f}%(3着内 {:4.1%})".format(100 * lo, 100 * hi, np.mean([x["in3"] for x in g])), g)


hon = [x for x in H if x["top"]]
gap_table("単勝 ◎(確率80%以上)を、○との確率の差で分ける", [x for x in hon if x["p"] >= 0.80])
gap_table("単勝 ◎(確率70〜80%)を、○との確率の差で分ける", [x for x in hon if 0.70 <= x["p"] < 0.80])
gap_table("単勝 ◎(全部)を、○との確率の差で分ける", hon)

# 「80%以上・差30%以上で、オッズが2倍以上なら 56% × 2倍 でプラス?」: オッズの帯ごとに、実際の1着の割合を見る
print("\n■ 単勝 ◎80%以上・○との差30%以上を、確定オッズで分ける(1着の割合 × オッズ が 1 を超えるか)")
g30 = [x for x in hon if x["p"] >= 0.80 and x["gap"] >= 0.30]
for lo, hi in ((1, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 1e9), (2.0, 1e9)):
    show("オッズ {:.1f}〜{}倍".format(lo, "{:.1f}".format(hi) if hi < 1e9 else ""), [x for x in g30 if lo <= x["odds"] < hi])
print("\n■ くらべる: ◎80%以上(差は問わない)をオッズで分ける")
for lo, hi in ((2.0, 2.5), (2.5, 1e9), (2.0, 1e9)):
    show("オッズ {:.1f}〜{}倍".format(lo, "{:.1f}".format(hi) if hi < 1e9 else ""), [x for x in hon if x["p"] >= 0.80 and lo <= x["odds"] < hi])
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

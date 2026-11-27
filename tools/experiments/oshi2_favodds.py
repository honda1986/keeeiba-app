"""後押しの買い目(1番人気の後押しが弱いレースで、後押し〈2頭とも3着内の式〉上5%の馬 × 2番人気 の馬連とワイド)を、
1番人気のオッズで分けたらどうなるか(中央 2022〜2026年。後押しはその年を除いて作った式。人気・払戻は確定)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/oshi2_favodds.py
  1番人気が 2.0倍以下のレースで「買う」か「買わない」か。帯(〜1.5・1.5〜2.0・2.0〜3.0・3.0〜)でも見る。
  境目は tools/train_push.py と同じ(その年を除いて作った後押しの、単勝20倍以下の馬の中の下20%・上5%)"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import train_push as TP
import keiba_logic as L
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
D = TP.build(rows, races, log=lambda *a: None)
del rows, races
w, q, bw, bp = TP.loyo(D, log=lambda *a: None)
h = D["H"]; sizes = D["sizes"]; m20 = h["odds"] <= 20
fcut, xcut = float(np.quantile(w[m20], float(L.OSHI2["fav_q"]))), float(np.quantile(q[m20], float(L.OSHI2["x_q"])))
X = []
st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
for g, (s0, n) in enumerate(zip(st, sizes)):
    us = h["u"][s0:s0 + n].astype(int); od = {int(u): float(o) for u, o in zip(us, h["odds"][s0:s0 + n])}
    pw = {int(u): float(v) for u, v in zip(us, w[s0:s0 + n])}; p2 = {int(u): float(v) for u, v in zip(us, q[s0:s0 + n])}
    pay = D["info"][D["rid"][g]]["result"].get("payout") or {}
    for pk in L.oshi2_picks(pw, p2, od, fcut, xcut):
        key = "-".join(str(x) for x in sorted((pk["u"], pk["mate"])))
        X.append(dict(y=D["year"][g], fo=od[pk["fav"]], uma=(pay.get("馬連") or {}).get(key, 0), wide=(pay.get("ワイド") or {}).get(key, 0),
                      fav_in=pk["fav"] in (D["info"][D["rid"][g]]["result"].get("order") or [])[:3]))
print("後押しの買い目 {}頭 ({:.0f}分)".format(len(X), (time.time() - t0) / 60), flush=True)


def show(name, sel):
    if not sel:
        print("  {:28s} 0頭".format(name)); return
    y = np.array([x["y"] for x in sel])
    out = []
    for k in ("uma", "wide"):
        r = np.array([x[k] for x in sel], float); top = np.sort(r)[-3:].sum()
        out.append("{} {:5.1f}%(除3 {:5.1f}%)[{}]".format("馬連" if k == "uma" else "ワイド", r.mean(), (r.sum() - top) / len(r),
                   " ".join("{:3.0f}".format(r[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))
    both = np.array([x["uma"] + x["wide"] for x in sel], float)
    print("  {:28s} {:5d}頭(年に{:4.0f}) 1番人気が3着内 {:4.0%} | {} | {} | 2点合計 {:5.1f}%".format(
        name, len(sel), len(sel) / 4.9, np.mean([x["fav_in"] for x in sel]), out[0], out[1], both.mean() / 2))


print("\n■ 1番人気のオッズで分ける  [ ] は 2022〜2026年の回収率")
show("全部(今の買い方)", X)
show("1番人気 2.0倍以下(買う時)", [x for x in X if x["fo"] <= 2.0])
show("1番人気 2.0倍より上", [x for x in X if x["fo"] > 2.0])
print()
for lo, hi in ((1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)):
    show("1番人気 {:.1f}〜{}倍".format(lo, "{:.1f}".format(hi) if hi < 99 else ""), [x for x in X if lo < x["fo"] <= hi] if lo > 1 else [x for x in X if x["fo"] <= hi])
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

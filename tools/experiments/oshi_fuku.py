"""後押しが強い馬の複勝を買ったらどうなるか(中央 2022〜2026年。後押しはその年を除いて作った式)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/oshi_fuku.py
  後押しは2つ: 1着の式(push。後押しの買い目に使っている)と、2頭とも3着内の式(push2。期待値ワイドに使っている。3着内に近い)。
  それぞれ「単勝20倍以下の馬の中の上5%・10%・20%」と、上1割をオッズの帯で分けたもの、1番人気を除いたものを、複勝の払戻(確定)で数える。
  複勝の払戻は 2026-11-27 に取り直した(それまでは残していなかった)。払戻の無いレースは数えない"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import pickle
import train_model as T
import train_push as TP
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
D = TP.build(rows, races, log=lambda *a: None)
del rows, races
w, q, bw, bp = TP.loyo(D, log=lambda *a: None)
h = D["H"]; sizes = D["sizes"]; hy = D["hyear"]
odds = h["odds"]; u = h["u"].astype(int)
rid_of = np.repeat(np.array(D["rid"]), sizes)
fuku = np.full(len(u), np.nan)
fav = np.zeros(len(u), bool)
st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
for g, (s0, n) in enumerate(zip(st, sizes)):
    pay = (D["info"][D["rid"][g]]["result"].get("payout") or {}).get("複勝")
    o = odds[s0:s0 + n]; fav[s0 + int(np.argmin(o))] = True
    if pay:
        for j in range(n):
            fuku[s0 + j] = pay.get(str(u[s0 + j]), 0) / 100.0
have = ~np.isnan(fuku)
print("中央 {}頭・複勝の払戻があるのは {}頭({:.0%})  年ごと {} ({:.0f}分)".format(
    len(u), have.sum(), have.mean(), " ".join("{} {:.0%}".format(y, have[hy == y].mean()) for y in YS), (time.time() - t0) / 60), flush=True)


def show(name, m):
    m = m & have
    n = int(m.sum())
    if n < 50:
        print("  {:42s} {:6d}頭(少ない)".format(name, n)); return
    r = fuku[m]; y = hy[m]; top = np.sort(r)[-3:].sum()
    print("  {:42s} {:6d}頭 3着内 {:5.1%} 単勝 {:4.1f}倍 回収率 {:6.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
        name, n, np.mean(r > 0), np.median(odds[m]), 100 * r.mean(), 100 * (r.sum() - top) / n,
        " ".join("{:3.0f}".format(100 * r[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))


m20 = odds <= 20
print("\n■ 複勝  [ ] は 2022〜2026年の回収率(単勝は中央値)")
show("(物差し)全部の馬", np.ones(len(u), bool))
show("(物差し)単勝20倍以下の馬", m20)
show("(物差し)1番人気", fav)
for nm, v in (("1着の式(push)", w), ("2頭とも3着内の式(push2)", q)):
    print("\n ◆ " + nm)
    for qq in (0.95, 0.9, 0.8):
        c = np.quantile(v[m20], qq)
        show("上{:.0f}%(単勝20倍以下)".format(100 * (1 - qq)), m20 & (v >= c))
        show("上{:.0f}%(単勝20倍以下・1番人気を除く)".format(100 * (1 - qq)), m20 & (v >= c) & ~fav)
    show("下20%(単勝20倍以下)", m20 & (v <= np.quantile(v[m20], 0.2)))
    c = np.quantile(v[m20], 0.9)
    for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 20)):
        show("上10% × 単勝 {}〜{}倍".format(lo, hi), (v >= c) & (odds >= lo) & (odds < hi))
    c = np.quantile(v, 0.9)
    show("上10%(全部の馬の中で。20倍超えも)", v >= c)
    show("上10% × 単勝 20〜50倍", (v >= c) & (odds >= 20) & (odds < 50))
# もっと絞る: 上2〜3%、上5% をオッズの帯で(2頭とも3着内の式)
print("\n ◆ 2頭とも3着内の式(push2)をもっと絞る")
for qq in (0.97, 0.98, 0.99):
    c = np.quantile(q[m20], qq)
    show("上{:.0f}%(単勝20倍以下)".format(100 * (1 - qq)), m20 & (q >= c))
c = np.quantile(q[m20], 0.95)
for lo, hi in ((1, 4), (4, 8), (8, 14), (14, 20.01)):
    show("上5% × 単勝 {}〜{}倍".format(lo, int(hi)), (q >= c) & (odds >= lo) & (odds < hi))
# 2つの式の両方が強い
cw = np.quantile(w[m20], 0.9); cq = np.quantile(q[m20], 0.9)
show("1着の式も2頭とも3着内の式も上10%", m20 & (w >= cw) & (q >= cq))
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

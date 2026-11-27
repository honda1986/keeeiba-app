"""1番人気の後押しが弱いレースで、後押しが強い馬を買ったらどうなるか(中央 2022〜2026年。後押しはその年を除いて作った式)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/oshi_weakfav.py
  1番人気(確定オッズ)の後押しが「弱い」= 単勝20倍以下の馬の中で下20% / 下1/3(1着の式 push・2頭とも3着内の式 push2 のそれぞれで)。
  後押しが強い馬 x = 1番人気でない・単勝20倍以下・後押しが上5% / 上10%(買う式は 1着の式・2頭とも3着内の式)。
  x の 単勝・複勝、x と1番人気の 馬連・ワイド(今の後押しの買い目の形)、x と2番人気(x が2番人気なら3番人気)の 馬連・ワイド を、
  1番人気が弱いレースと、そうでないレースでくらべる。払戻は確定"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import train_push as TP
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
D = TP.build(rows, races, log=lambda *a: None)
del rows, races
w, q, bw, bp = TP.loyo(D, log=lambda *a: None)
h = D["H"]; sizes = D["sizes"]; odds = h["odds"]; uu = h["u"].astype(int)
m20 = odds <= 20
cut = {nm: {k: float(np.quantile(v[m20], k)) for k in (0.2, 1 / 3, 0.9, 0.95)} for nm, v in (("w", w), ("q", q))}
srt = lambda a, b: "{}-{}".format(min(a, b), max(a, b))
X = []          # 後押しが強い馬 1頭ごと
st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
for g, (s0, n) in enumerate(zip(st, sizes)):
    rc = D["info"][D["rid"][g]]; res = rc["result"]; pay = res.get("payout") or {}; order = res.get("order") or []
    o = odds[s0:s0 + n]; us = uu[s0:s0 + n]
    rk = np.argsort(o, kind="stable"); f1, f2, f3 = us[rk[0]], us[rk[1]], us[rk[2]]
    jf = s0 + rk[0]
    for j in range(n):
        u = us[j]
        if u == f1 or o[j] > 20:
            continue
        sw, sq = w[s0 + j], q[s0 + j]
        if sw < cut["w"][0.9] and sq < cut["q"][0.9]:
            continue
        mate2 = f2 if u != f2 else f3
        X.append(dict(y=D["year"][g], sw=sw, sq=sq, fw=w[jf], fq=q[jf], od=o[j],
                      tan=o[j] * 100 if order and order[0] == u else 0, fuku=(pay.get("複勝") or {}).get(str(u), 0),
                      ur1=(pay.get("馬連") or {}).get(srt(u, f1), 0), wd1=(pay.get("ワイド") or {}).get(srt(u, f1), 0),
                      ur2=(pay.get("馬連") or {}).get(srt(u, mate2), 0), wd2=(pay.get("ワイド") or {}).get(srt(u, mate2), 0),
                      fin=int(f1 in order[:3])))
print("後押しが強い馬(どちらかの式で上10%・1番人気でない・20倍以下) {}頭 ({:.0f}分)".format(len(X), (time.time() - t0) / 60), flush=True)
BETS = (("単勝 x", "tan"), ("複勝 x", "fuku"), ("馬連 x-1番人気(今の形)", "ur1"), ("ワイド x-1番人気", "wd1"),
        ("馬連 x-2番人気", "ur2"), ("ワイド x-2番人気", "wd2"))


def table(title, sel):
    if len(sel) < 100:
        print("\n■ {}: {}頭(少ない)".format(title, len(sel))); return
    print("\n■ {}: {}頭(1番人気が3着外 {:.0%})  [ ] は 2022〜2026年".format(title, len(sel), 1 - np.mean([x["fin"] for x in sel])))
    y = np.array([x["y"] for x in sel])
    for nm, k in BETS:
        r = np.array([x[k] for x in sel], float); top = np.sort(r)[-3:].sum()
        print("  {:24s} 的中 {:5.1%} 回収率 {:6.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
            nm, np.mean(r > 0), r.mean(), (r.sum() - top) / len(r),
            " ".join("{:3.0f}".format(r[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))


for buy, bk in (("1着の式", "sw"), ("2頭とも3着内の式", "sq")):
    ck = "w" if bk == "sw" else "q"
    for top, tq in (("上5%", 0.95), ("上10%", 0.9)):
        strong = [x for x in X if x[bk] >= cut[ck][tq]]
        print("\n==================== 後押しが強い馬 = {}の{} ====================".format(buy, top))
        table("全部", strong)
        for fnm, fk, fc in (("1着の式", "fw", "w"), ("2頭とも3着内の式", "fq", "q")):
            for lab, k in (("下20%", 0.2), ("下1/3", 1 / 3)):
                table("1番人気の後押し({})が{}".format(fnm, lab), [x for x in strong if x[fk] <= cut[fc][k]])
            table("1番人気の後押し({})が下1/3でない".format(fnm), [x for x in strong if x[fk] > cut[fc][1 / 3]])
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

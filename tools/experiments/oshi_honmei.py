"""後押しの買い目の相手を、1番人気でなく ◎ にしたらどうなるか(中央 2022〜2026年。その年を除いて学んだモデルと後押しの式)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/oshi_honmei.py
  PRED … fresh_years.py(YEARS=2022,...,2026)の3着内の確率。◎○ はアプリと同じ(軸にできる馬の確率上位2頭)。
  後押しの馬 x … 後押し(1着の式)が上5%・1番人気でない・単勝20倍以下(keiba_logic.oshi_picks。今の買い目と同じ)。
  人気・オッズ・払戻は確定のもの。くらべる買い方:
   今の形: 馬連 x-1番人気 / ◎にする: 馬連 x-◎(x が ◎ なら ○)/ 両方(2点)/ ワイド・馬単・単勝 も
  ◎が1番人気のレースはどちらも同じ組になるので、◎が1番人気でないレースだけでもくらべる"""
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
p3 = np.load(os.environ["PRED"])
D = TP.build(rows, races)
w, q, bw, bp = TP.loyo(D, log=lambda *a: None)
h = D["H"]; sizes = D["sizes"]
cut = float(np.quantile(w[h["odds"] <= 20], float(L.OSHI["q"])))
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
anyr = T.pick_model(0.0, -1.0, P3, True)
info = D["info"]
srt = lambda us: "-".join(str(x) for x in sorted(us))
print("準備 ({:.0f}分)".format((time.time() - t0) / 60), flush=True)

R = []
st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
for g, (s0, n) in enumerate(zip(st, sizes)):
    rid = D["rid"][g]
    if rid not in jg or rid not in info:
        continue
    s = anyr(rid, jg[rid], None)
    if not s:
        continue
    hon, tai = s[0], s[1]
    us = h["u"][s0:s0 + n].astype(int); od = {int(u): float(o) for u, o in zip(us, h["odds"][s0:s0 + n])}
    push = {int(u): float(v) for u, v in zip(us, w[s0:s0 + n])}
    res = info[rid]["result"]; order = res.get("order") or []
    if len(order) < 3:
        continue
    for pk in L.oshi_picks(push, od, cut):
        R.append(dict(y=D["year"][g], x=pk["u"], fav=pk["fav"], hon=hon, tai=tai, od=od, order=order, pay=res.get("payout") or {},
                      honfav=hon == pk["fav"], xhon=pk["u"] == hon, ph=P3.get((rid, hon), 0)))
print("後押しの馬 {}頭(レース {}) ◎が1番人気 {:.1%} / x が ◎ {:.1%} ({:.0f}分)".format(
    len(R), len({id(r["pay"]) for r in R}), np.mean([r["honfav"] for r in R]), np.mean([r["xhon"] for r in R]), (time.time() - t0) / 60), flush=True)


def pt(r, typ, us):
    key = "-".join(str(x) for x in us) if typ in ("馬単", "3連単") else srt(us)
    if typ == "単勝":
        return r["od"][us[0]] * 100 if r["order"][0] == us[0] else 0
    return (r["pay"].get(typ) or {}).get(key, 0)


mate = lambda r: r["tai"] if r["xhon"] else r["hon"]          # x の相手の ◎(x が ◎ なら ○)
BETS = [
    ("馬連 x-1番人気(今の形)", lambda r: [("馬連", [r["x"], r["fav"]])]),
    ("馬連 x-◎(x が◎なら○)", lambda r: [("馬連", [r["x"], mate(r)])]),
    ("馬連 x-◎ と x-1番人気(2点。同じなら1点)", lambda r: [("馬連", [r["x"], m]) for m in dict.fromkeys([r["fav"], mate(r)])]),
    ("ワイド x-1番人気", lambda r: [("ワイド", [r["x"], r["fav"]])]),
    ("ワイド x-◎(x が◎なら○)", lambda r: [("ワイド", [r["x"], mate(r)])]),
    ("馬単 x→1番人気", lambda r: [("馬単", [r["x"], r["fav"]])]),
    ("馬単 x→◎(x が◎なら○)", lambda r: [("馬単", [r["x"], mate(r)])]),
    ("単勝 x", lambda r: [("単勝", [r["x"]])]),
]


def table(title, sel):
    print("\n■ {}: 後押しの馬 {}頭  [ ] は 2022〜2026年の回収率".format(title, len(sel)))
    for name, f in BETS:
        n = stake = hit = 0; pay = 0.0; yp = defaultdict(float); ys = defaultdict(int); big = []
        for r in sel:
            b = f(r)
            v = [pt(r, t, us) for t, us in b]
            n += 1; stake += len(b); hit += any(v); pay += sum(v); yp[r["y"]] += sum(v); ys[r["y"]] += len(b); big += v
        if not n:
            continue
        top3 = sorted(big)[-3:]
        print("  {:38s} {:5d}点 的中 {:5.1%} 回収率 {:6.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
            name, stake, hit / n, pay / stake, (pay - sum(top3)) / stake,
            " ".join("{:3.0f}".format(yp[y] / ys[y]) if ys[y] else "  -" for y in YS)))


table("全部(今の後押しの買い目の対象)", R)
table("◎が1番人気でないとき(組が変わるところ)", [r for r in R if not r["honfav"]])
table("  そのうち x が ◎ でないとき", [r for r in R if not r["honfav"] and not r["xhon"]])
table("  そのうち x が ◎ のとき(相手は○)", [r for r in R if not r["honfav"] and r["xhon"]])
for lo, hi in ((0, 0.6), (0.6, 0.75), (0.75, 1.01)):
    table("◎が1番人気でない・◎の確率 {:.0%}〜{:.0%}".format(lo, min(hi, 1)), [r for r in R if not r["honfav"] and lo <= r["ph"] < hi])
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

"""1番人気が 3.0倍以下なのに 3着に来ない(飛ぶ)レースは予想できるか。できるなら、1番人気を外して買えば儲かるか
(中央 2022〜2026年。その年を除いて学んだモデル・後押しの式・飛ぶ確率の式)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/fav_fly.py
  1番人気は確定オッズで決める。飛ぶ確率はロジスティック回帰で、材料を足していって当たり方(AUC)をくらべる:
   A. オッズだけ(今のアプリの荒れ度と同じ: log(1番人気のオッズ)・オッズのばらけ方)
   B. A + 2番人気のオッズ・頭数
   C. B + モデル(1番人気の3着内の確率・モデルで何番目か・◎か)
   D. C + 後押し(1番人気の1着の式・2頭とも3着内の式の値。過去の成績が良く見える馬は買われすぎ)
  式はその年を除いた年で作る。飛ぶ確率の高いレースで、1番人気を外した買い目(モデルの確率の順に、1番人気でない馬)を買う"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import train_push as TP
import keiba_final as F
YS = ("2022", "2023", "2024", "2025", "2026")
MAX_ODDS = 3.0
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
p3 = np.load(os.environ["PRED"])
D = TP.build(rows, races)
w, q, bw, bp = TP.loyo(D, log=lambda *a: None)
h = D["H"]; sizes = D["sizes"]
PW, PQ = {}, {}
k0 = 0
for g, rid in enumerate(D["rid"]):
    for j in range(sizes[g]):
        PW[(rid, int(h["u"][k0 + j]))] = w[k0 + j]; PQ[(rid, int(h["u"][k0 + j]))] = q[k0 + j]
    k0 += sizes[g]
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
R = []
for rid, rs in jg.items():
    if rid not in info or (rid, rs[0]["u"]) not in PW:
        continue
    res = info[rid]["result"]; order = res.get("order") or []
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    if len(order) < 3 or len(od) < 5:
        continue
    favs = sorted(od, key=lambda u: (od[u], u)); fav = favs[0]
    if od[fav] > MAX_ODDS or (rid, fav) not in P3:
        continue
    s = anyr(rid, rs, None)
    mp = sorted((u for u in od if (rid, u) in P3), key=lambda u: -P3[(rid, u)])
    if not s or len(mp) < 5:
        continue
    cx = F.chaos_x(od)
    R.append(dict(rid=rid, y=rs[0]["date"][:4], fly=int(fav not in order[:3]), fav=fav, od=od, order=order, pay=res.get("payout") or {},
                  x=[cx[0], cx[1], np.log(od[favs[1]]), len(od),
                     P3[(rid, fav)], mp.index(fav) + 1, int(s[0] == fav),
                     PW.get((rid, fav), 0.0), PQ.get((rid, fav), 0.0)],
                  alt=[u for u in mp if u != fav], f2=favs[1]))
Y = np.array([r["fly"] for r in R], float); X = np.array([r["x"] for r in R], float); YR = np.array([r["y"] for r in R])
print("中央 1番人気 {:.1f}倍以下 {}R。飛んだ(3着外) {:.1%}  年ごと {} ({:.0f}分)".format(
    MAX_ODDS, len(R), Y.mean(), " ".join("{} {:.1%}".format(y, Y[YR == y].mean()) for y in YS), (time.time() - t0) / 60), flush=True)


def auc(p, y):
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


SETS = {"A オッズだけ(今の荒れ度)": [0, 1], "B +2番人気のオッズ・頭数": [0, 1, 2, 3], "C +モデル": [0, 1, 2, 3, 4, 5, 6],
        "D +後押し": [0, 1, 2, 3, 4, 5, 6, 7, 8]}
CV = {}
print("\n■ 飛ぶかの当たり方(AUC。0.5 = 当てずっぽう、1 = 全部当たる。式はその年を除いて作る)")
for name, cols in SETS.items():
    Xs = X[:, cols]; cv = np.zeros(len(Y))
    for yy in YS:
        tr = YR != yy
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-9
        wt = F.fit_logit((Xs[tr] - mu) / sd, Y[tr])
        cv[YR == yy] = 1 / (1 + np.exp(-(wt[0] + ((Xs[YR == yy] - mu) / sd) @ wt[1:])))
    CV[name] = cv
    ll = -np.mean(Y * np.log(cv) + (1 - Y) * np.log(1 - cv))
    print("  {:26s} AUC {:.3f}  年ごと [{}]  対数損失 {:.4f}".format(name, auc(cv, Y), " ".join("{:.3f}".format(auc(cv[YR == y], Y[YR == y])) for y in YS), ll))
best = CV["D +後押し"]
print("\n■ 飛ぶ確率(D)の帯ごとの、実際に飛んだ割合  [ ] は 2022〜2026年")
for lo, hi in ((0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)):
    m = (best >= lo) & (best < hi)
    if m.sum():
        print("  {:3.0f}〜{:3.0f}%  {:5d}R 見込み {:5.1%} 実際 {:5.1%} [{}]".format(100 * lo, 100 * min(hi, 1), m.sum(), best[m].mean(), Y[m].mean(),
              " ".join("{:4.0%}".format(Y[m & (YR == y)].mean()) if (m & (YR == y)).any() else "   -" for y in YS)))
a = CV["A オッズだけ(今の荒れ度)"]
for nm, p in (("A(オッズだけ)", a), ("D(+モデル・後押し)", best)):
    top = p >= np.quantile(p, 0.9)
    print("  {} の上1割 {}R: 実際に飛んだ {:.1%}".format(nm, top.sum(), Y[top].mean()))


def pt(r, typ, us):
    if typ == "単勝":
        return r["od"][us[0]] * 100 if r["order"][0] == us[0] else 0
    return (r["pay"].get(typ) or {}).get(srt(us), 0)


BETS = [
    ("単勝 1番人気でないモデル1位", lambda r: [("単勝", r["alt"][:1])]),
    ("単勝 2番人気", lambda r: [("単勝", [r["f2"]])]),
    ("単勝 1番人気(くらべる)", lambda r: [("単勝", [r["fav"]])]),
    ("ワイド 1番人気を外したモデル上位2頭", lambda r: [("ワイド", r["alt"][:2])]),
    ("馬連 1番人気を外したモデル上位2頭", lambda r: [("馬連", r["alt"][:2])]),
    ("3連複 1番人気を外したモデル上位3頭", lambda r: [("3連複", r["alt"][:3])]),
    ("3連複 1番人気を外したモデル上位4頭のBOX(4点)", lambda r: [("3連複", list(c)) for c in
        [(a, b, c) for i, a in enumerate(r["alt"][:4]) for j, b in enumerate(r["alt"][:4]) if j > i for k, c in enumerate(r["alt"][:4]) if k > j]]),
]


def bet_table(title, sel):
    print("\n■ {}: {}R(実際に飛んだ {:.1%})  [ ] は 2022〜2026年の回収率".format(title, len(sel), np.mean([r["fly"] for r in sel]) if sel else 0))
    for name, f in BETS:
        st = n = hit = 0; pay = 0.0; yp = defaultdict(float); ys = defaultdict(int); big = []
        for r in sel:
            b = f(r); v = [pt(r, t, us) for t, us in b]
            n += 1; st += len(b); hit += any(v); pay += sum(v); yp[r["y"]] += sum(v); ys[r["y"]] += len(b); big += v
        if st:
            print("  {:36s} {:5d}点 的中 {:5.1%} 回収率 {:6.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
                name, st, hit / n, pay / st, (pay - sum(sorted(big)[-3:])) / st,
                " ".join("{:3.0f}".format(yp[y] / ys[y]) if ys[y] else "  -" for y in YS)))


bet_table("全部(1番人気 3.0倍以下)", R)
for th in (0.35, 0.40, 0.45):
    bet_table("飛ぶ確率(D) {:.0%}以上".format(th), [r for r, p in zip(R, best) if p >= th])
bet_table("飛ぶ確率(A オッズだけ) 40%以上", [r for r, p in zip(R, a) if p >= 0.40])
# 反対に、飛びにくいレースで1番人気の単勝を買ったら
for th in (0.20, 0.15):
    bet_table("飛ぶ確率(D) {:.0%}未満(1番人気が来そう)".format(th), [r for r, p in zip(R, best) if p < th])
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

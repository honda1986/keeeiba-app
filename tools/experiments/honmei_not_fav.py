"""◎が1番人気でないとき、買い方で回収率は良くなるか(中央 2022〜2026年。その年を除いて学んだモデル)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/honmei_not_fav.py
  PRED … fresh_years.py(YEARS=2022,...,2026)の3着内の確率。後押しも、その年を除いて作った式(tools/train_push.py の loyo)。
  ◎○ = 軸にできる馬の確率上位2頭(アプリと同じ)、▲ = 残りの確率1位。人気は確定の単勝オッズの順。
  レースを「◎が何番人気か」で分け、券種・組み合わせごとに 的中・回収率(年ごと)を出す:
   単勝(◎ / 1番人気)、ワイド・馬連・馬単(◎と○ / ◎と1番人気)、3連複(◎○▲ / ◎○と1番人気 / ◎と1番人気とモデルの次の馬 など)
  さらに ◎の確率・1番人気のオッズ・後押しの強さで分ける。今の買うレース(◎80%以上・◎と○の差15%以上)の中でも見る"""
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
p3 = np.load(os.environ["PRED"])
D = TP.build(rows, races)
w, q, bw, bp = TP.loyo(D, log=lambda *a: None)
h = D["H"]; sizes = D["sizes"]
PW = {}
k0 = 0
for g, rid in enumerate(D["rid"]):
    for j in range(sizes[g]):
        PW[(rid, int(h["u"][k0 + j]))] = w[k0 + j]
    k0 += sizes[g]
cut_hi, cut_lo = np.quantile(w[h["odds"] <= 20], 0.8), np.quantile(w[h["odds"] <= 20], 0.2)     # 後押しが強い(上2割)・弱い(下2割)
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
info = D["info"]
anyr = T.pick_model(0.0, -1.0, P3, True)
buy = T.pick_model(0.0, -1.0, P3, True, t1=0.80, g12=0.15)
srt = lambda us: "-".join(str(x) for x in sorted(us))

R = []
for rid, rs in jg.items():
    if rid not in info or (rid, rs[0]["u"]) not in PW:
        continue
    s = anyr(rid, rs, None)
    res = info[rid]["result"]; order = res.get("order") or []
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    if not s or len(s[2]) < 2 or len(order) < 3 or len(od) < 5 or s[0] not in od or s[1] not in od:
        continue
    favs = sorted(od, key=lambda u: (od[u], u))
    hon, tai, rest = s
    rest = [x for x in rest if x in od]
    if len(rest) < 2:
        continue
    R.append(dict(rid=rid, y=rs[0]["date"][:4], hon=hon, tai=tai, a=rest[0], b=rest[1], rest=rest, f1=favs[0], f2=favs[1], f3=favs[2],
                  nk=favs.index(hon) + 1, od=od, order=order, pay=res.get("payout") or {}, ph=P3[(rid, hon)], buy=buy(rid, rs, None) is not None,
                  wh=PW.get((rid, hon), 0.0), wf=PW.get((rid, favs[0]), 0.0), wide_ng=T.trio_wide(info[rid], rest[0])))
print("中央 {}R ({:.0f}分)。◎が1番人気 {:.1%} / 2番人気 {:.1%} / 3番人気 {:.1%} / 4番人気以下 {:.1%}".format(
    len(R), (time.time() - t0) / 60, *[np.mean([c(r["nk"]) for r in R]) for c in (lambda n: n == 1, lambda n: n == 2, lambda n: n == 3, lambda n: n >= 4)]), flush=True)


def pt(r, typ, us):
    """1点の払戻(100円あたり)。馬単・3連単は us の順、ほかは小さい順"""
    key = "-".join(str(x) for x in us) if typ in ("馬単", "3連単") else srt(us)
    if typ == "単勝":
        return r["od"][us[0]] * 100 if r["order"][0] == us[0] else 0
    return (r["pay"].get(typ) or {}).get(key, 0)


def other(r, ex, k=1):
    """モデルの確率が高い順に、ex に入っていない馬を k 頭"""
    c = [r["hon"], r["tai"]] + r["rest"]
    return [x for x in c if x not in ex][:k]


# 買い方: r -> [(券種, 馬番…)](点数ぶん)。買えない形なら None
BETS = [
    ("単勝 ◎", lambda r: [("単勝", [r["hon"]])]),
    ("単勝 1番人気", lambda r: [("単勝", [r["f1"]])]),
    ("ワイド ◎-○", lambda r: [("ワイド", [r["hon"], r["tai"]])]),
    ("ワイド ◎-1番人気", lambda r: [("ワイド", [r["hon"], r["f1"]])] if r["hon"] != r["f1"] else None),
    ("ワイド ◎-1番人気 と ◎-○(2点)", lambda r: [("ワイド", [r["hon"], r["f1"]]), ("ワイド", [r["hon"], r["tai"]])] if r["f1"] not in (r["hon"], r["tai"]) else None),
    ("ワイド 1番人気-2番人気", lambda r: [("ワイド", [r["f1"], r["f2"]])]),
    ("ワイド 1番人気-モデルの次の馬(◎以外)", lambda r: [("ワイド", [r["f1"], other(r, (r["hon"], r["f1"]))[0]])]),
    ("馬連 ◎-○", lambda r: [("馬連", [r["hon"], r["tai"]])]),
    ("馬連 ◎-1番人気", lambda r: [("馬連", [r["hon"], r["f1"]])] if r["hon"] != r["f1"] else None),
    ("馬単 1番人気→◎", lambda r: [("馬単", [r["f1"], r["hon"]])] if r["hon"] != r["f1"] else None),
    ("馬単 ◎→1番人気", lambda r: [("馬単", [r["hon"], r["f1"]])] if r["hon"] != r["f1"] else None),
    ("3連複 ◎-○-▲(今の形)", lambda r: [("3連複", [r["hon"], r["tai"], r["a"]])]),
    ("3連複 ◎-○-1番人気(1番人気が◎○なら▲)", lambda r: [("3連複", [r["hon"], r["tai"], r["f1"] if r["f1"] not in (r["hon"], r["tai"]) else r["a"]])]),
    ("3連複 ◎-1番人気-モデルの次の馬", lambda r: [("3連複", [r["hon"], r["f1"]] + other(r, (r["hon"], r["f1"])))] if r["hon"] != r["f1"] else None),
    ("3連複 ◎-1番人気-モデルの次の2頭(2点)", lambda r: [("3連複", [r["hon"], r["f1"], x]) for x in other(r, (r["hon"], r["f1"]), 2)] if r["hon"] != r["f1"] else None),
    ("3連複 ◎-1番人気-2番人気(◎が2番人気なら3番人気)", lambda r: [("3連複", [r["hon"], r["f1"], r["f2"] if r["f2"] != r["hon"] else r["f3"]])] if r["hon"] != r["f1"] else None),
    ("3連複 1番人気-2番人気-◎以外のモデルの次の馬", lambda r: [("3連複", [r["f1"], r["f2"]] + other(r, (r["hon"], r["f1"], r["f2"])))]),
    ("3連単 1番人気→◎→モデルの次の馬", lambda r: [("3連単", [r["f1"], r["hon"]] + other(r, (r["hon"], r["f1"])))] if r["hon"] != r["f1"] else None),
]


def table(title, sel, bets=BETS, min_n=150):
    if len(sel) < min_n:
        print("\n■ {}: {}R(少ないので出さない)".format(title, len(sel))); return
    print("\n■ {}: {}R  [ ] は 2022〜2026年の回収率".format(title, len(sel)))
    for name, f in bets:
        n = st = 0; hit = 0; pay = 0.0; yp = defaultdict(float); ys = defaultdict(int)
        for r in sel:
            b = f(r)
            if not b:
                continue
            v = [pt(r, t, us) for t, us in b]
            n += 1; st += len(b); hit += any(v); pay += sum(v); yp[r["y"]] += sum(v); ys[r["y"]] += len(b)
        if n >= min_n // 2:
            print("  {:42s} {:5d}R 的中 {:5.1%} 回収率 {:6.1f}% [{}]".format(name, n, hit / n, pay / st,
                  " ".join("{:3.0f}".format(yp[y] / ys[y]) if ys[y] else "  -" for y in YS)))


table("全レース", R)
table("◎が1番人気", [r for r in R if r["nk"] == 1])
NF = [r for r in R if r["nk"] > 1]
table("◎が1番人気でない(全部)", NF)
table("◎が2番人気", [r for r in R if r["nk"] == 2])
table("◎が3番人気", [r for r in R if r["nk"] == 3])
table("◎が4番人気以下", [r for r in R if r["nk"] >= 4])
# ◎の確率・1番人気のオッズ・後押しで分ける(◎が1番人気でないレース)
SHORT = [b for b in BETS if b[0] in ("単勝 ◎", "単勝 1番人気", "ワイド ◎-○", "ワイド ◎-1番人気", "馬連 ◎-1番人気", "3連複 ◎-○-▲(今の形)",
                                    "3連複 ◎-1番人気-モデルの次の馬", "3連複 ◎-1番人気-モデルの次の2頭(2点)", "馬単 1番人気→◎")]
table("◎が1番人気でない・◎の確率 70%以上", [r for r in NF if r["ph"] >= .70], SHORT)
table("◎が1番人気でない・◎の確率 55〜70%", [r for r in NF if .55 <= r["ph"] < .70], SHORT)
table("◎が1番人気でない・◎の確率 55%未満", [r for r in NF if r["ph"] < .55], SHORT)
table("◎が1番人気でない・1番人気が 2.0倍以下", [r for r in NF if r["od"][r["f1"]] <= 2.0], SHORT)
table("◎が1番人気でない・1番人気が 2.0〜3.0倍", [r for r in NF if 2.0 < r["od"][r["f1"]] <= 3.0], SHORT)
table("◎が1番人気でない・1番人気が 3.0倍より上", [r for r in NF if r["od"][r["f1"]] > 3.0], SHORT)
table("◎が1番人気でない・◎の後押しが強い(上2割)", [r for r in NF if r["wh"] >= cut_hi], SHORT)
table("◎が1番人気でない・◎の後押しがふつう", [r for r in NF if cut_lo < r["wh"] < cut_hi], SHORT)
table("◎が1番人気でない・◎の後押しが弱い(下2割)", [r for r in NF if r["wh"] <= cut_lo], SHORT)
table("◎が1番人気でない・1番人気の後押しが弱い(下2割)", [r for r in NF if r["wf"] <= cut_lo], SHORT)
table("◎が1番人気でない・1番人気の後押しが強い(上2割)", [r for r in NF if r["wf"] >= cut_hi], SHORT)
table("◎が1番人気でない・◎の単勝 3〜6倍", [r for r in NF if 3 <= r["od"][r["hon"]] < 6], SHORT)
table("◎が1番人気でない・◎の単勝 6〜10倍", [r for r in NF if 6 <= r["od"][r["hon"]] < 10], SHORT)
table("◎が1番人気でない・◎の単勝 10倍以上", [r for r in NF if r["od"][r["hon"]] >= 10], SHORT)
# 今の買うレースの中で
NOW = ("今の買い方(3連複◎-○-▲、▲が延長ならワイド◎-○)", lambda r: [("ワイド", [r["hon"], r["tai"]])] if r["wide_ng"] else [("3連複", [r["hon"], r["tai"], r["a"]])])
BB = [r for r in R if r["buy"]]
table("今の買うレース(◎80%以上・◎と○の差15%以上)", BB, [NOW] + BETS, 100)
table("今の買うレース・◎が1番人気", [r for r in BB if r["nk"] == 1], [NOW] + BETS, 100)
table("今の買うレース・◎が1番人気でない", [r for r in BB if r["nk"] > 1], [NOW] + BETS, 60)
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

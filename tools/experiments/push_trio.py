"""後押し(keiba_push)で、今の買い方(3連複◎-○-▲)の回収率は上がるか。
「ワイドの的中率が上がれば、3連複の回収率も上がる」は本当か(中央 2022〜2026年)。
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/push_trio.py
  PRED … fresh_years.py(YEARS=2022,...,2026)の3着内の確率(その年を除いて学んだモデル)。
  後押しも、その年を除いて作った式のもの(tools/train_push.py の loyo)。
  1. 今の買い方(◎80%以上・◎と○の差15%以上のレースで 3連複◎-○-▲、▲が距離延長ならワイド◎-○)を、後押しの強さで3つに分ける
  2. 組の選び方を変えて「2頭とも3着内」の的中率を上げたとき、ワイドと3連複の回収率はどうなるか
     (◎○ = モデルのまま / 3連複のオッズでいちばん来そうな組 / それに後押しを足した組。3頭目もモデル・オッズ・後押しで選ぶ)
  3. 今の買い方の ▲ を、後押しで選び直したら"""
import gc, os, pickle, sys, time
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
w, q, bw, bp = TP.loyo(D)
h = D["H"]; sizes = D["sizes"]
PW, PQ = {}, {}                        # (rid, 馬番) -> 後押し(1着の式 / 2頭とも3着内の式)
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
info = D["info"]
buy = T.pick_model(0.0, -1.0, P3, True, t1=0.80, g12=0.15)      # 今の買うレース(◎80%以上・◎と○の差15%以上)
anyr = T.pick_model(0.0, -1.0, P3, True)                        # 基準なし(全レースの◎○)
srt = lambda us: "-".join(str(x) for x in sorted(us))
print("準備 ({:.0f}分)".format((time.time() - t0) / 60), flush=True)


def pays(rid):
    pay = info[rid]["result"].get("payout") or {}
    return pay.get("ワイド") or {}, pay.get("3連複") or {}


def show(name, recs):
    """recs = [(年, 的中か, 100円あたりの払戻)]"""
    if not recs:
        print("  {:44s}    0R".format(name)); return
    y = np.array([r[0] for r in recs]); hit = np.array([r[1] for r in recs], float); pay = np.array([r[2] for r in recs], float)
    print("  {:44s} {:5d}R 的中 {:5.1%} 回収率 {:6.1f}% [{}]".format(name, len(recs), hit.mean(), pay.mean(),
          " ".join("{:3.0f}".format(pay[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))


# ---------------------------------------------------------------- 1. 今の買い方を後押しで分ける
B = []
for rid, rs in jg.items():
    if rid not in info or (rid, rs[0]["u"]) not in PQ:
        continue
    s = buy(rid, rs, None)
    if not s or not s[2]:
        continue
    hh, tt, rest = s; a = rest[0]
    if any((rid, u) not in PQ for u in (hh, tt, a)):
        continue
    wp, tp_ = pays(rid); wide = T.trio_wide(info[rid], a)
    v = wp.get(srt((hh, tt)), 0) if wide else tp_.get(srt((hh, tt, a)), 0)
    B.append(dict(rid=rid, y=rs[0]["date"][:4], h=hh, t=tt, a=a, rest=rest, wide=wide, pay=v,
                  q2=PQ[(rid, hh)] + PQ[(rid, tt)], q3=(PQ[(rid, hh)] + PQ[(rid, tt)] + PQ[(rid, a)]),
                  qa=PQ[(rid, a)], wh=PW[(rid, hh)], wpay=wp.get(srt((hh, tt)), 0), tpay=tp_.get(srt((hh, tt, a)), 0)))
print("\n■ 1. 今の買い方(3連複◎-○-▲、▲が距離延長ならワイド◎-○)を後押しで3つに分ける  [ ] は 2022〜2026年の回収率")
show("今の買い方(全部)", [(b["y"], b["pay"] > 0, b["pay"]) for b in B])
for nm, key in (("◎○▲の後押し(3頭の合計)", "q3"), ("◎○の後押し(2頭の合計)", "q2"), ("▲の後押し", "qa"), ("◎の後押し(1着の式)", "wh")):
    v = np.array([b[key] for b in B]); c1, c2 = np.quantile(v, [1 / 3, 2 / 3])
    for lab, m in (("弱い", v <= c1), ("ふつう", (v > c1) & (v < c2)), ("強い", v >= c2)):
        show("  {} {}".format(nm, lab), [(b["y"], b["pay"] > 0, b["pay"]) for b, k in zip(B, m) if k])
v = np.array([b["q3"] for b in B])
for qq in (0.5, 0.25):
    show("  ◎○▲の後押しが上 {:.0%}".format(1 - qq), [(b["y"], b["pay"] > 0, b["pay"]) for b, k in zip(B, v >= np.quantile(v, qq)) if k])
    show("  ◎○▲の後押しが下 {:.0%}".format(1 - qq), [(b["y"], b["pay"] > 0, b["pay"]) for b, k in zip(B, v <= np.quantile(v, 1 - qq)) if k])
show("  (同じレースで ワイド◎-○ を買うと)", [(b["y"], b["wpay"] > 0, b["wpay"]) for b in B])

# ---------------------------------------------------------------- 2. 組の選び方を変える
print("\n■ 2. 「2頭とも3着内」の的中率を上げると、ワイドと3連複の回収率はどうなるか")


def market_pair(rid, us, adj=0.0):
    """3連複のオッズでいちばん来そうな組(adj>0 なら、後押しを足した順で)"""
    tp = info[rid].get("trio_pair_final") or {}; best = None
    for k, s in tp.items():
        a, b = (int(t) for t in k.split("-"))
        if a not in us or b not in us:
            continue
        sc = float(TP.lg(np.array([min(L.SOLID["pt"] * s, 0.98)]))[0]) + adj * (PQ[(rid, a)] + PQ[(rid, b)])
        if best is None or sc > best[0]:
            best = (sc, a, b, L.SOLID["pt"] * s)
    return best


def third(rid, us, a, b, how):
    """3頭目: model = モデルの確率 / market = 3連複のオッズ(a・b それぞれとの組の強さの合計)/ push = market に後押しを足す"""
    tp = info[rid].get("trio_pair_final") or {}
    cand = [x for x in us if x not in (a, b)]
    if not cand:
        return None
    if how == "model":
        return max(cand, key=lambda x: P3.get((rid, x), 0))
    st = lambda x: tp.get(srt((a, x)), 0) + tp.get(srt((b, x)), 0)
    if how == "market":
        return max(cand, key=st)
    return max(cand, key=lambda x: np.log(max(st(x), 1e-6)) + PQ[(rid, x)])


SETS = {"今の買うレース": {b["rid"] for b in B}, "全レース": None}
for sname, sel in SETS.items():
    print("  -- {}".format(sname))
    R = defaultdict(list)
    for rid, rs in jg.items():
        if rid not in info or (rid, rs[0]["u"]) not in PQ or (sel is not None and rid not in sel):
            continue
        us = {r["u"] for r in rs if (rid, r["u"]) in PQ}
        if len(us) < 5:
            continue
        y = rs[0]["date"][:4]; wp, tp_ = pays(rid)
        s = anyr(rid, rs, None)
        mk = market_pair(rid, us); mp = market_pair(rid, us, 1.0)
        pairs = {"モデルの◎○": (s[0], s[1]) if s and s[0] in us and s[1] in us else None,
                 "オッズでいちばん来そうな組": (mk[1], mk[2]) if mk else None,
                 "オッズ+後押しでいちばん来そうな組": (mp[1], mp[2]) if mp else None}
        for pn, pr in pairs.items():
            if not pr:
                continue
            a, b = pr
            R[(pn, "ワイド")].append((y, wp.get(srt((a, b)), 0) > 0, wp.get(srt((a, b)), 0)))
            for hn in ("model", "market", "push"):
                x = third(rid, us, a, b, hn)
                if x is not None:
                    R[(pn, "3連複 3頭目=" + {"model": "モデル", "market": "オッズ", "push": "オッズ+後押し"}[hn])].append((y, tp_.get(srt((a, b, x)), 0) > 0, tp_.get(srt((a, b, x)), 0)))
    for (pn, bn), recs in R.items():
        show("{} / {}".format(pn, bn), recs)

# ---------------------------------------------------------------- 3. 今の買い方の ▲ を後押しで選び直す
print("\n■ 3. 今の買うレースで、◎○はそのまま、▲を選び直す(3連複◎-○-▲。距離延長でワイドに替えるのは今と同じ判断を▲ごとに)")
for nm, keyf in (("今の▲(モデルの3番手)", lambda b, x: P3.get((b["rid"], x), 0)),
                 ("モデルの確率 + 後押し(上位4頭の中で)", lambda b, x: np.log(max(P3.get((b["rid"], x), 1e-6), 1e-6)) + PQ.get((b["rid"], x), 0)),
                 ("後押しがいちばん強い(上位4頭の中で)", lambda b, x: PQ.get((b["rid"], x), -9))):
    recs = []
    for b in B:
        cand = [x for x in b["rest"][:4] if (b["rid"], x) in PQ]
        if not cand:
            continue
        x = max(cand, key=lambda u: keyf(b, u)); wp, tp_ = pays(b["rid"])
        v = wp.get(srt((b["h"], b["t"])), 0) if T.trio_wide(info[b["rid"]], x) else tp_.get(srt((b["h"], b["t"], x)), 0)
        recs.append((b["y"], v > 0, v))
    show(nm, recs)
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

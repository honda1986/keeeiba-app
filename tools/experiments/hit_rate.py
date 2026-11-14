"""1着の的中率・ワイドの的中率は、どうすればどこまで上がるか(中央 2022〜2026年。その年を除いて学んだモデル)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy OUTDIR=/path/to/dir python tools/experiments/hit_rate.py
  PRED … fresh_years.py を YEARS=2022,...,2026 で回して作った3着内の確率。
  1着を当てるモデル(材料は同じ。答えを「1着か」に変えて学ぶ。種1〜4の平均)も、その年を除いて作る(OUTDIR に保存)。
  くらべるもの:
   1着 … 今の◎ / 3着内の確率1位 / 1着の確率1位 / 1番人気(単勝オッズ) / モデルと人気が同じ馬のレースだけ / 自信のあるレースだけ
   ワイド … 今の◎-○ / 3着内の確率の上位2頭 / ワイドの1番人気の組 / 3連複から見ていちばん来そうな組 / 自信のあるレースだけ
  的中率・買えるレースの割合・回収率(確定の払戻)を出す。オッズを使う選び方は、買う時点のオッズでは少し変わる"""
import gc, os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
OUT = os.environ.get("OUTDIR", "/tmp/hit_rate"); os.makedirs(OUT, exist_ok=True)
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
info = {rc["id"]: {"id": rc["id"], "center": rc.get("center"), "result": rc["result"], "trio_wide_u": T.trio_wide_us(rc),
                   "tp": rc.get("trio_pair_final") or {}, "wf": rc.get("wide_final") or {}} for d, rc in races}
del races; gc.collect()
p3 = np.load(os.environ["PRED"])
dts = np.array([r["date"] for r in rows])
f = os.path.join(OUT, "pwin.npy")
if os.path.exists(f):
    pw1 = np.load(f)
else:
    feats = sorted({k for r in rows for k in r["f"]})
    X = np.array([[M._num(r["f"].get(k)) for k in feats] for r in rows])
    y1 = np.array([int((info[r["rid"]]["result"]["order"] or [None])[0] == r["u"]) for r in rows])
    cat = [feats.index(k) for k in ("venue", "style") if k in feats]
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.7,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1}
    tr, va = dts < T.VALID_FROM, (dts >= T.VALID_FROM) & (dts < T.TEST_FROM)
    its = {}
    for s in (1, 2, 3, 4):
        dtr = lgb.Dataset(X[tr], y1[tr], categorical_feature=cat); dva = lgb.Dataset(X[va], y1[va], categorical_feature=cat, reference=dtr)
        its[s] = lgb.train(dict(params, seed=s), dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)]).best_iteration
    print("1着のモデルの回数", its, flush=True)
    pw1 = np.full(len(rows), np.nan)
    for yy in YS:
        m_pr = np.array([d[:4] == yy for d in dts]); m_tr = ~m_pr
        pw1[m_pr] = np.mean([lgb.train(dict(params, seed=s), lgb.Dataset(X[m_tr], y1[m_tr], categorical_feature=cat), its[s]).predict(X[m_pr]) for s in its], axis=0)
        print("{}年 ({:.0f}分)".format(yy, (time.time() - t0) / 60), flush=True)
    np.save(f, pw1); del X; gc.collect()

by = defaultdict(list)
for i, r in enumerate(rows):
    if not np.isnan(p3[i]):
        by[r["rid"]].append(i)
P3, P1 = {}, {}
for rid, ix in by.items():
    tot = p3[ix].sum(); want = min(3, len(ix)); t1 = pw1[ix].sum()
    for i in ix:
        P3[(rid, rows[i]["u"])] = min(0.99, p3[i] * want / tot); P1[(rid, rows[i]["u"])] = pw1[i] / t1
jg = T.group(T.of_center(rows, "jra"))
pick_all = T.pick_model(0.0, -1.0, P3, True)          # 今の◎○(軸にできる馬の上位2頭。基準なし)
srt = lambda c: "-".join(str(x) for x in sorted(c))
mid = lambda v: (v[0] + v[1]) / 2
R = []
for rid, rs in jg.items():
    if (rid, rs[0]["u"]) not in P3:
        continue
    x = info[rid]; res = x["result"]; order = res.get("order") or []
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    if len(order) < 3 or len(od) < 5:
        continue
    us = [r["u"] for r in rs if r["u"] in od]
    if len(us) < 5:
        continue
    q = {u: 1 / od[u] for u in us}; qs = sum(q.values()); q = {u: v / qs for u, v in q.items()}
    s_ = pick_all(rid, rs, None)
    top3 = set(order[:3]); pay = res.get("payout") or {}
    by3 = sorted(us, key=lambda u: -P3[(rid, u)]); by1 = sorted(us, key=lambda u: -P1[(rid, u)]); byq = sorted(us, key=lambda u: -q[u])
    wf = {k: v for k, v in x["wf"].items() if v[0] > 1}; tp = x["tp"]
    wk = min(wf, key=lambda k: mid(wf[k])) if wf else None                      # ワイドの1番人気の組
    tk = max((k for k in tp if k in wf), key=lambda k: tp[k], default=None)       # 3連複から見ていちばん来そうな組
    R.append(dict(y=rs[0]["date"][:4], rid=rid, win=order[0], top3=top3, od=od, q=q, pay=pay, wf=wf, tp=tp,
                  hon=s_[0] if s_ else None, tai=s_[1] if s_ else None, by3=by3, by1=by1, byq=byq, wk=wk, tk=tk,
                  p3=lambda u, rid=rid: P3[(rid, u)], p1=lambda u, rid=rid: P1[(rid, u)]))
N = len(R)
print("中央 {}R ({:.0f}分)".format(N, (time.time() - t0) / 60))


def line(name, sel, hit, payf):
    """sel = レースのリスト、hit(r) = 当たりか、payf(r) = 100円あたりの払戻"""
    if not sel:
        print("  {:44s}    0R".format(name)); return
    h = [hit(r) for r in sel]; p = [payf(r) for r in sel]
    yh = [np.mean([hit(r) for r in sel if r["y"] == y]) if any(r["y"] == y for r in sel) else 0 for y in YS]
    print("  {:44s} {:6d}R({:5.1%}) 的中 {:5.1%} [{}] 回収率 {:4.0f}%".format(name, len(sel), len(sel) / N, np.mean(h),
          " ".join("{:2.0f}".format(v * 100) for v in yh), np.mean(p)))


win_hit = lambda key: (lambda r: r[key] == r["win"]) if isinstance(key, str) else None
tan = lambda u_of: (lambda r: r["od"][u_of(r)] * 100 if u_of(r) == r["win"] else 0)
print("\n■ 1着を当てる(単勝1点)  レース数(全体に対する割合) 的中 [2022〜2026] 回収率")
line("今の◎(軸にできる馬で3着内の確率1位)", [r for r in R if r["hon"]], lambda r: r["hon"] == r["win"], tan(lambda r: r["hon"]))
line("3着内の確率1位(軸の決まりなし)", R, lambda r: r["by3"][0] == r["win"], tan(lambda r: r["by3"][0]))
line("1着の確率1位(1着を当てるモデル)", R, lambda r: r["by1"][0] == r["win"], tan(lambda r: r["by1"][0]))
line("1番人気(単勝オッズ)", R, lambda r: r["byq"][0] == r["win"], tan(lambda r: r["byq"][0]))
same = [r for r in R if r["by1"][0] == r["byq"][0]]
line("1着のモデルと1番人気が同じ馬のレースだけ", same, lambda r: r["byq"][0] == r["win"], tan(lambda r: r["byq"][0]))
line("  違うレースで 1着のモデルの馬", [r for r in R if r["by1"][0] != r["byq"][0]], lambda r: r["by1"][0] == r["win"], tan(lambda r: r["by1"][0]))
line("  違うレースで 1番人気", [r for r in R if r["by1"][0] != r["byq"][0]], lambda r: r["byq"][0] == r["win"], tan(lambda r: r["byq"][0]))
print("  -- 自信のあるレースだけ(1着のモデルの確率で)")
for th in (.25, .30, .35, .40, .50):
    line("1着の確率 {:.0%} 以上".format(th), [r for r in R if r["p1"](r["by1"][0]) >= th], lambda r: r["by1"][0] == r["win"], tan(lambda r: r["by1"][0]))
print("  -- 自信のあるレースだけ(1番人気の単勝オッズで)")
for th in (3.0, 2.5, 2.0, 1.7, 1.5):
    line("1番人気が {:.1f}倍 以下".format(th), [r for r in R if r["od"][r["byq"][0]] <= th], lambda r: r["byq"][0] == r["win"], tan(lambda r: r["byq"][0]))
print("  -- モデルと人気の両方(同じ馬で、1着の確率と単勝オッズ)")
for pth, oth in ((.30, 3.0), (.35, 2.5), (.40, 2.0), (.45, 1.8)):
    line("同じ馬・確率 {:.0%} 以上・{:.1f}倍 以下".format(pth, oth), [r for r in same if r["p1"](r["by1"][0]) >= pth and r["od"][r["byq"][0]] <= oth],
         lambda r: r["byq"][0] == r["win"], tan(lambda r: r["byq"][0]))

wd = lambda pair_of: (lambda r: r["pay"].get("ワイド", {}).get(srt(pair_of(r)), 0))
whit = lambda pair_of: (lambda r: set(pair_of(r)) <= r["top3"])
kpair = lambda k: [int(x) for x in k.split("-")]
print("\n■ ワイドを当てる(1点)  レース数(割合) 的中 [2022〜2026] 回収率")
line("今の◎-○(軸にできる馬の上位2頭)", [r for r in R if r["hon"] and r["tai"]], whit(lambda r: (r["hon"], r["tai"])), wd(lambda r: (r["hon"], r["tai"])))
line("3着内の確率の上位2頭(軸の決まりなし)", R, whit(lambda r: r["by3"][:2]), wd(lambda r: r["by3"][:2]))
line("単勝の1・2番人気", R, whit(lambda r: r["byq"][:2]), wd(lambda r: r["byq"][:2]))
Wk = [r for r in R if r["wk"]]; Tk = [r for r in R if r["tk"]]
line("ワイドの1番人気の組(いちばんオッズが低い組)", Wk, whit(lambda r: kpair(r["wk"])), wd(lambda r: kpair(r["wk"])))
line("3連複から見ていちばん来そうな組", Tk, whit(lambda r: kpair(r["tk"])), wd(lambda r: kpair(r["tk"])))
line("モデルの上位2頭 = ワイドの1番人気の組 のレースだけ", [r for r in Wk if set(r["by3"][:2]) == set(kpair(r["wk"]))], whit(lambda r: r["by3"][:2]), wd(lambda r: r["by3"][:2]))
print("  -- 自信のあるレースだけ(今の◎の3着内の確率で。今の◎-○)")
for th in (.60, .70, .80, .85):
    line("◎ {:.0%} 以上".format(th), [r for r in R if r["hon"] and r["tai"] and r["p3"](r["hon"]) >= th], whit(lambda r: (r["hon"], r["tai"])), wd(lambda r: (r["hon"], r["tai"])))
print("  -- 自信のあるレースだけ(◎と○の両方の確率で。今の◎-○)")
for a, b in ((.60, .50), (.70, .50), (.70, .60), (.80, .60)):
    line("◎ {:.0%}・○ {:.0%} 以上".format(a, b), [r for r in R if r["hon"] and r["tai"] and r["p3"](r["hon"]) >= a and r["p3"](r["tai"]) >= b],
         whit(lambda r: (r["hon"], r["tai"])), wd(lambda r: (r["hon"], r["tai"])))
print("  -- 自信のあるレースだけ(ワイドの1番人気の組のオッズで)")
for th in (3.0, 2.5, 2.0, 1.7, 1.5):
    line("いちばん低いワイドが {:.1f}倍 以下(下限)".format(th), [r for r in Wk if r["wf"][r["wk"]][0] <= th], whit(lambda r: kpair(r["wk"])), wd(lambda r: kpair(r["wk"])))
print("  -- 自信のあるレースだけ(3連複から見た確率 = 0.75×組の強さ で)")
for th in (.30, .40, .50, .60):
    line("来そうな組の確率 {:.0%} 以上".format(th), [r for r in Tk if 0.75 * r["tp"][r["tk"]] >= th], whit(lambda r: kpair(r["tk"])), wd(lambda r: kpair(r["tk"])))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

# ---- アプリに入れやすい選び方(今ある3着内の確率とオッズだけで決める。1着のモデルは使わない)
print("\n■ アプリに入れる候補: 堅い単勝(1番人気の単勝オッズと、モデルの見立て)")
fav = lambda r: r["byq"][0]
for oth in (2.5, 2.0, 1.8, 1.7, 1.5):
    S = [r for r in R if r["od"][fav(r)] <= oth]
    line("1番人気 {:.1f}倍以下".format(oth), S, lambda r: fav(r) == r["win"], tan(fav))
    line("  + 3着内の確率1位も同じ馬", [r for r in S if r["by3"][0] == fav(r)], lambda r: fav(r) == r["win"], tan(fav))
    line("  + その馬の3着内の確率 70%以上", [r for r in S if r["p3"](fav(r)) >= .70], lambda r: fav(r) == r["win"], tan(fav))
    line("  - モデルは別の馬が1位", [r for r in S if r["by3"][0] != fav(r)], lambda r: fav(r) == r["win"], tan(fav))
print("\n■ アプリに入れる候補: 堅いワイド(3連複から見ていちばん来そうな組の確率と、モデルの見立て)")
tkp = lambda r: 0.75 * r["tp"][r["tk"]]
for pth in (.35, .40, .45, .50, .55):
    S = [r for r in Tk if tkp(r) >= pth]
    line("来そうな組の確率 {:.0%}以上".format(pth), S, whit(lambda r: kpair(r["tk"])), wd(lambda r: kpair(r["tk"])))
    line("  + 2頭ともモデルの3着内の確率の上位3頭", [r for r in S if set(kpair(r["tk"])) <= set(r["by3"][:3])], whit(lambda r: kpair(r["tk"])), wd(lambda r: kpair(r["tk"])))
    line("  - どちらかがモデルの上位3頭にいない", [r for r in S if not set(kpair(r["tk"])) <= set(r["by3"][:3])], whit(lambda r: kpair(r["tk"])), wd(lambda r: kpair(r["tk"])))
# 見込みの確率と実際(合っているか)
print("\n■ 3連複から見た確率は合っているか(いちばん来そうな組)")
for a, b in ((0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1)):
    S = [r for r in Tk if a <= tkp(r) < b]
    if S:
        print("  見込み {:.0%}〜{:.0%}: {:5d}R 見込みの平均 {:.1%} 実際 {:.1%}".format(a, b, len(S), np.mean([tkp(r) for r in S]), np.mean([set(kpair(r["tk"])) <= r["top3"] for r in S])))
print("■ 1日あたり(中央の開催日 {}日)".format(len({r["rid"][:4] + r["rid"][6:10] + r["y"] for r in R})))

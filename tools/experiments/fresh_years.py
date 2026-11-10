"""今の買い方を、買い方選びに一度も使っていない 2022・2023年で確かめる(中央)
  KEIBA_CACHE=/path/to/cache.pkl OUTDIR=/path/to/dir python tools/experiments/fresh_years.py
  買う基準(◎80%以上・◎と○の差15%以上)・3連複◎○▲・▲が延長ならワイド は、どれも 2024〜2026年の成績を見て決めた。
  2022年と2023年は、その年を除いたほかの年で学んだモデル(種1〜4の平均。回数は本番と同じ決め方)で確率を作り、同じ買い方を当てはめる。
  先の年で学んだモデルを前の年に使うので本番の順とは逆だが、買い方はこの2年の結果を一度も見ていないので、
  「たまたま良く見える買い方を選んだだけか」の確かめになる。2022年の前半は騎手・厩舎の成績がまだ少ない"""
import gc, os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
OUT = os.environ.get("OUTDIR", "/tmp/fresh_years"); os.makedirs(OUT, exist_ok=True)
SEEDS = [1, 2, 3, 4]
# YEARS=2022,2023,2024,2025,2026 にすると、5年とも「その年を除いて学んだモデル」でそろえて比べられる
YEARS = os.environ.get("YEARS", "2022,2023").split(",")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
races_by_id = {rc["id"]: {"id": rc["id"], "center": rc.get("center"), "result": rc["result"], "trio_wide_u": T.trio_wide_us(rc)} for d, rc in races}
del races; gc.collect()
feats = sorted({k for r in rows for k in r["f"]})
y = np.array([r["y"] for r in rows]); dts = np.array([r["date"] for r in rows])
X = np.array([[M._num(r["f"].get(k)) for k in feats] for r in rows])
cat = [feats.index(k) for k in ("venue", "style") if k in feats]
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.7,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1}
tr, va = dts < T.VALID_FROM, (dts >= T.VALID_FROM) & (dts < T.TEST_FROM)
print("材料 {}頭 ({:.0f}秒)".format(len(rows), time.time() - t0), flush=True)
f = os.path.join(OUT, "pred.npy")
if os.path.exists(f):
    pred = np.load(f)
else:
    iters = {}
    for s in SEEDS:      # 回数は本番と同じ(調整期間の早期終了)
        dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cat); dva = lgb.Dataset(X[va], y[va], categorical_feature=cat, reference=dtr)
        iters[s] = lgb.train(dict(params, seed=s), dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)]).best_iteration
    print("回数", iters, flush=True)
    pred = np.full(len(rows), np.nan)
    for yy in YEARS:
        m_pr = np.array([d[:4] == yy for d in dts]); m_tr = ~m_pr
        pred[m_pr] = np.mean([lgb.train(dict(params, seed=s), lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), iters[s]).predict(X[m_pr])
                              for s in SEEDS], axis=0)
        print("{}年 学習{}頭 → {}頭 ({:.0f}分)".format(yy, m_tr.sum(), m_pr.sum(), (time.time() - t0) / 60), flush=True)
    np.save(f, pred)

by = defaultdict(list)
for r, v in zip(rows, pred):
    if not np.isnan(v):
        by[r["rid"]].append((r["u"], v))
pw = {}
for rid, us in by.items():
    tot = sum(v for _, v in us); want = min(3, len(us))
    for u, v in us:
        pw[(rid, u)] = min(0.99, v * want / tot)
jra = T.of_center(rows, "jra"); jg = T.group(jra)
rdate = {rid: rs[0]["date"] for rid, rs in jg.items()}
SEL = T.fixed_selects()["jra"]
srt = lambda c: "-".join(str(x) for x in sorted(c))


def per_race(t1=.8, g12=.15, bet="now"):
    pick = T.pick_sel(dict(SEL, t1=t1, g12=g12), pw); out = {}
    for rid, rs in jg.items():
        if (rid, rs[0]["u"]) not in pw:
            continue
        s_ = pick(rid, rs, None)
        if not s_ or not s_[2]:
            continue
        pay = races_by_id[rid]["result"].get("payout") or {}
        w = pay.get("ワイド", {}).get(srt(s_[:2]), 0); t = pay.get("3連複", {}).get(srt([s_[0], s_[1], s_[2][0]]), 0)
        out[rid] = {"now": w if T.trio_wide(races_by_id[rid], s_[2][0]) else t, "trio": t, "wide": w}[bet]
    return out


def show(name, pr_):
    v = np.array([pr_[r] for r in sorted(pr_)], dtype=float)
    if not len(v):
        return
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(5000)])
    hy = defaultdict(list)
    for r, x in pr_.items():
        hy[rdate[r][:4] + ("前" if rdate[r][4:6] <= "06" else "後")].append(x)
    print("  {:30s} {:4d}R {:4.0f}% 的中 {:3.0%}  95%の幅 {:.0f}〜{:.0f}%・100%を超える見込み {:.0%}  半年ごと [{}]".format(
        name, len(v), v.mean(), (v > 0).mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 100).mean(),
        " ".join("{} {:.0f}".format(k, np.mean(x)) for k, x in sorted(hy.items()))))


print("\n■ {}年(その年を除いて学んだモデル)に今の買い方を当てはめる。2022・2023年は買い方選びに使っていない年".format("・".join(YEARS)))
show("今の買い方(▲が延長ならワイド)", per_race())
show("全部 3連複◎○▲", per_race(bet="trio"))
show("全部 ワイド◎-○", per_race(bet="wide"))
for yy in YEARS:
    for b, nm in (("now", "今の買い方"), ("trio", "3連複"), ("wide", "ワイド")):
        show("  {}年 {}".format(yy, nm), {r: x for r, x in per_race(bet=b).items() if rdate[r][:4] == yy})
print("\n■ 基準を変えると(今の買い方のまま)")
for t1 in (.70, .75, .80, .85):
    for g12 in (0, .15, .25):
        show("◎{:.0%}以上・差{:.0%}以上".format(t1, g12), per_race(t1, g12))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

"""乱数の種を変えた何本かのモデルの確率を平均したら、予想と今の買い方の成績は安定するか(中央。月ごとに学び直して2024〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl OUTDIR=/path/to/dir python tools/experiments/ensemble_wf.py
  種 1〜8 で、train_model.py と同じ学び方(早期終了で回数を決め、2024年1月から月ごとに学び直す)の確率を作り、OUTDIR に保存する
  (途中で止まっても、保存した種は飛ばす)。1本ずつ・4本の平均(A=種1〜4、B=種5〜8)・8本の平均で、
  今の基準(◎80%以上・◎と○の差15%以上)の回収率と、2024・2025年で選び直した基準の2026年の回収率をくらべる"""
import gc, os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
OUT = os.environ.get("OUTDIR", "/tmp/ensemble_wf")
os.makedirs(OUT, exist_ok=True)
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5,6,7,8").split(",")]
t0 = time.time()
races = T.load_races()
if os.path.exists(T.CACHE):
    norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
else:
    norm = T.l3f_norms(races); sp = T.speed_table(races, T.VALID_FROM)
    rows, jt_all = T.build(races, norm, sp)
    pickle.dump((norm, rows, jt_all, sp), open(T.CACHE, "wb"))
races_by_id = {rc["id"]: (rc if d >= T.VALID_FROM else {"id": rc["id"], "center": rc.get("center"), "result": rc["result"],
                                                       "trio_wide_u": T.trio_wide_us(rc)}) for d, rc in races}
del races; gc.collect()
feats = sorted({k for r in rows for k in r["f"]})
y = np.array([r["y"] for r in rows]); dts = np.array([r["date"] for r in rows])
X = np.array([[M._num(r["f"].get(k)) for k in feats] for r in rows])
cat = [feats.index(k) for k in ("venue", "style") if k in feats]
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.7,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1}
tr, va = dts < T.VALID_FROM, (dts >= T.VALID_FROM) & (dts < T.TEST_FROM)
months = [m for m in sorted({d[:6] for d in dts}) if m >= T.WF_FROM]
print("材料 {}頭・{}個 ({:.0f}秒)".format(len(rows), len(feats), time.time() - t0), flush=True)

raw = {}
for s in SEEDS:
    f = os.path.join(OUT, "seed{}.npy".format(s))
    if os.path.exists(f):
        raw[s] = np.load(f); continue
    pr = dict(params, seed=s)
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], categorical_feature=cat, reference=dtr)
    it = lgb.train(pr, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)]).best_iteration
    pred = np.full(len(rows), np.nan)
    for mth in months:
        m_tr, m_pr = dts < mth + "01", np.array([d[:6] == mth for d in dts])
        pred[m_pr] = lgb.train(pr, lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), it).predict(X[m_pr])
    np.save(f, pred); raw[s] = pred
    print("種{} 回数{} ({:.0f}分)".format(s, it, (time.time() - t0) / 60), flush=True)


def norm_race(pred):
    by = defaultdict(list)
    for r, v in zip(rows, pred):
        if not np.isnan(v):
            by[r["rid"]].append((r["u"], v))
    out = {}
    for rid, us in by.items():
        tot = sum(v for _, v in us); want = min(3, len(us))
        for u, v in us:
            out[(rid, u)] = min(0.99, v * want / tot)
    return out


jra = T.of_center(rows, "jra"); jg = T.group(jra)
rdate = {rid: rs[0]["date"] for rid, rs in jg.items()}
SEL = T.fixed_selects()["jra"]


def per_race(pw, t1, g12):
    pick = T.pick_sel(dict(SEL, t1=t1, g12=g12), pw); out = {}
    for rid, rs in jg.items():
        s_ = pick(rid, rs, None)
        if not s_ or not s_[2]:
            continue
        b = "wide" if T.trio_wide(races_by_id[rid], s_[2][0]) else "trio"
        pay = (races_by_id[rid]["result"].get("payout") or {}).get("3連複" if b == "trio" else "ワイド") or {}
        out[rid] = pay.get("-".join(str(x) for x in sorted([s_[0], s_[1], s_[2][0]] if b == "trio" else [s_[0], s_[1]])), 0)
    return out


PER = (("2024", "20240101", "20250101"), ("2025", "20250101", "20260101"), ("2026/1-8", "20260101", "20260901"), ("2026/9-", "20260901", "99999999"))


def roi(pr_, a="0", b="9"):
    v = [x for rid, x in pr_.items() if a <= rdate[rid] < b]
    return (np.mean(v) if v else 0.0), len(v)


def line(name, pw, t1=.8, g12=.15):
    pr_ = per_race(pw, t1, g12)
    ll = []
    for yy in ("2024", "2025", "2026"):
        sel = [i for i, r in enumerate(rows) if r["date"][:4] == yy and (r["rid"], r["u"]) in pw]
        p = np.clip(np.array([pw[(rows[i]["rid"], rows[i]["u"])] for i in sel]), 1e-4, 1 - 1e-4); yt = y[sel]
        ll.append(float(np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))))
    print("  {:18s} [{}] 全体 {:4.0f}%({}R)  対数尤度 [{}]".format(name, " ".join("{:4.0f}%".format(roi(pr_, a, b)[0]) for _, a, b in PER),
          roi(pr_)[0], len(pr_), " ".join("{:.5f}".format(x) for x in ll)), flush=True)
    return pr_


print("\n■ 今の基準(◎80%以上・◎と○の差15%以上)  回収率 [2024 2025 2026/1-8 2026/9-] 全体 / 対数尤度 [2024 2025 2026]")
single = {s: line("1本 種{}".format(s), norm_race(raw[s])) for s in SEEDS}
ens = {}
groups = [("A 種1〜4の平均", [s for s in SEEDS if s <= 4]), ("B 種5〜8の平均", [s for s in SEEDS if s >= 5]), ("8本の平均", SEEDS)]
for name, ss in groups:
    if ss:
        ens[name] = norm_race(np.mean([raw[s] for s in ss], axis=0))
        line(name, ens[name])
v1 = [roi(p)[0] for p in single.values()]
print("  1本ずつの全体の回収率: 平均 {:.0f}%・幅 {:.0f}〜{:.0f}%".format(np.mean(v1), min(v1), max(v1)))
if "A 種1〜4の平均" in ens and "B 種5〜8の平均" in ens:
    a_, b_ = set(per_race(ens["A 種1〜4の平均"], .8, .15)), set(per_race(ens["B 種5〜8の平均"], .8, .15))
    s1, s2 = set(single[SEEDS[0]]), set(single[SEEDS[1]])
    print("  買うレースの重なり: 1本どうし(種{}・{}) 同じ {} / 片方だけ {}  |  4本の平均どうし(A・B) 同じ {} / 片方だけ {}".format(
        SEEDS[0], SEEDS[1], len(s1 & s2), len(s1 ^ s2), len(a_ & b_), len(a_ ^ b_)))

print("\n■ 基準を2024・2025年で選び直して2026年で確かめる(◎の下限 × ◎と○の差の下限。2024・2025年で300R以上)")
for name, pw in ens.items():
    grid = []
    for t1 in (.70, .75, .80, .85):
        for g12 in (0, .05, .10, .15, .20, .25):
            pr_ = per_race(pw, t1, g12)
            r24, n24 = roi(pr_, "20240101", "20250101"); r25, n25 = roi(pr_, "20250101", "20260101"); r26, n26 = roi(pr_, "20260101", "99999999")
            if n24 + n25 >= 300:
                grid.append(((r24 * n24 + r25 * n25) / (n24 + n25), min(r24, r25), t1, g12, r24, r25, r26, n26))
    grid.sort(key=lambda g: -g[0])
    print("  " + name)
    for g in grid[:5]:
        print("    ◎{:.0%}以上・差{:.0%}以上  2024・2025 {:4.0f}%(低い年 {:4.0f}%) → 2026 {:4.0f}%({}R)".format(g[2], g[3], g[0], g[1], g[6], g[7]))
    cur = next(g for g in grid if g[2] == .8 and g[3] == .15) if any(g[2] == .8 and g[3] == .15 for g in grid) else None
    if cur:
        print("    (今の基準 ◎80%以上・差15%以上  2024・2025 {:4.0f}% → 2026 {:4.0f}%({}R))".format(cur[0], cur[6], cur[7]))

# 4本の平均のぶれ: 8本から4本を選ぶ70通りすべてで、今の基準の回収率がどれだけ散らばるか(1本ずつの散らばりとくらべる)
from itertools import combinations
if len(SEEDS) >= 8:
    v4 = []
    for ss in combinations(SEEDS, 4):
        v4.append(roi(per_race(norm_race(np.mean([raw[s] for s in ss], axis=0)), .8, .15))[0])
    print("\n■ 4本の平均の組み合わせ {}通り: 全体の回収率 平均 {:.0f}%・幅 {:.0f}〜{:.0f}%・標準偏差 {:.1f}(1本ずつは {:.1f})".format(
        len(v4), np.mean(v4), min(v4), max(v4), np.std(v4), np.std(v1)))
# たまたまの幅: レースを選び直して(ブートストラップ)、回収率がどこまで動くか
rng = np.random.default_rng(0)
for name, pw in ens.items():
    pr_ = per_race(pw, .8, .15); rids = sorted(pr_)
    for lab, a, b in (("全体", "0", "9"), ("2026年", "20260101", "99999999")):
        v = np.array([pr_[r] for r in rids if a <= rdate[r] < b], dtype=float)
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(5000)])
        print("  {:14s} {:6s} {:4.0f}%({}R) 95%の幅 {:.0f}〜{:.0f}%・100%を超える見込み {:.0%}・的中 {:.0%}".format(
            name, lab, v.mean(), len(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 100).mean(), (v > 0).mean()))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

"""前走のレースの強さ(「負けた相手のその後」)をモデルの材料に足したら(中央。月ごとに学び直して2024〜2026年で確かめる)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/race_level_wf.py
  材料(その日より前に分かることだけ): 中央の前走〜3走前のそれぞれで、一緒に走った馬が、そのあと(今回より前)の次の1走で
  3着内に来た率(頭数で平均に寄せる)。前走は、先着された相手・先着した相手に分けた率と、先着された相手が次に勝った率、数も。
  今の材料だけのモデルと、足したモデルを、学習の仕方(train_model.py と同じ)・月ごとの学び直し(2024年1月〜)をそろえてくらべる。
  当たり方(対数尤度・AUC)と、今の買い方(◎80%以上・◎と○の差15%以上、3連複◎○▲、▲が延長ならワイド◎-○)の回収率を年ごとに"""
import bisect, gc, json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
t0 = time.time()
races = T.load_races()
print("レース", len(races), races[0][0], "〜", races[-1][0], flush=True)
if os.path.exists(T.CACHE):
    norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
else:
    norm = T.l3f_norms(races); sp = T.speed_table(races, T.VALID_FROM)
    rows, jt_all = T.build(races, norm, sp)
    pickle.dump((norm, rows, jt_all, sp), open(T.CACHE, "wb"))
print("材料 {}頭 ({:.0f}秒)".format(len(rows), time.time() - t0), flush=True)

# ---- 前走のレースの強さ
runs, field, hid_of = defaultdict(list), {}, {}
for d, rc in races:
    order = rc["result"]["order"]; rank = {u: i + 1 for i, u in enumerate(order)}
    hm = {h["umaban"]: h.get("horse_id") for h in rc["horses"] if h.get("horse_id")}
    field[rc["id"]] = [(hm[u], rank[u]) for u in order if u in hm]
    for u in order:
        if u in hm:
            runs[hm[u]].append((d, rc["id"], rank[u]))
    for u, h in hm.items():
        hid_of[(rc["id"], u)] = h
for v in runs.values():
    v.sort()
dates = {h: [x[0] for x in v] for h, v in runs.items()}
allr = [x[2] for v in runs.values() for x in v]
B3, BW = np.mean([r <= 3 for r in allr]), np.mean([r == 1 for r in allr])


def level_feats(T_, hid):
    ds = dates.get(hid, [])
    i = bisect.bisect_left(ds, T_)
    prev = runs[hid][max(0, i - 3):i][::-1]
    out, lvs = {}, []
    for j, (dx, rx, rkx) in enumerate(prev):
        a_h = a_c = ah = ac = aw = bh = bc = 0
        for o, rk in field.get(rx, []):
            if o == hid:
                continue
            od = dates[o]; k = bisect.bisect_right(od, dx)
            if k < len(od) and od[k] < T_:
                r2 = runs[o][k][2]
                a_c += 1; a_h += r2 <= 3
                if rk < rkx:
                    ac += 1; ah += r2 <= 3; aw += r2 == 1
                else:
                    bc += 1; bh += r2 <= 3
        lv = (a_h + 5 * B3) / (a_c + 5)
        lvs.append(lv)
        if j == 0:
            out.update(rl_last=lv, rl_last_cnt=a_c, rl_ahead=(ah + 3 * B3) / (ac + 3) if ac else None,
                       rl_behind=(bh + 5 * B3) / (bc + 5) if bc else None, rl_winahead=(aw + 3 * BW) / (ac + 3) if ac else None)
    if lvs:
        out["rl_mean3"] = float(np.mean(lvs)); out["rl_max3"] = max(lvs)
    return out


NEW = ["rl_last", "rl_last_cnt", "rl_ahead", "rl_behind", "rl_winahead", "rl_mean3", "rl_max3", "rl_last_rank", "rl_last_dmean"]
extra = {}
for rid, rs in T.group(rows).items():
    fs = {r["u"]: level_feats(r["date"], hid_of.get((rid, r["u"]))) if hid_of.get((rid, r["u"])) else {} for r in rs}
    vals = sorted([f["rl_last"] for f in fs.values() if "rl_last" in f], reverse=True)
    mean = np.mean(vals) if vals else None
    for u, f in fs.items():
        if "rl_last" in f:
            f["rl_last_rank"] = vals.index(f["rl_last"]) + 1; f["rl_last_dmean"] = f["rl_last"] - mean
        extra[(rid, u)] = f
print("強さの材料 {:.0f}秒。前走の強さが出せた馬 {:.0%}".format(time.time() - t0, np.mean([("rl_last" in f) for f in extra.values()])), flush=True)
races_by_id = {rc["id"]: (rc if d >= T.VALID_FROM else {"id": rc["id"], "center": rc.get("center"), "result": rc["result"],
                                                       "trio_wide_u": T.trio_wide_us(rc)}) for d, rc in races}
del races; gc.collect()

base_feats = sorted({k for r in rows for k in r["f"]})
y = np.array([r["y"] for r in rows]); dts = np.array([r["date"] for r in rows])
Xb = np.array([[M._num(r["f"].get(k)) for k in base_feats] for r in rows])
Xn = np.column_stack([Xb, np.array([[M._num(extra.get((r["rid"], r["u"]), {}).get(k)) for k in NEW] for r in rows])])
cat = [base_feats.index(k) for k in ("venue", "style") if k in base_feats]
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.7,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 1}
tr, va = dts < T.VALID_FROM, (dts >= T.VALID_FROM) & (dts < T.TEST_FROM)


def norm_race(pred):
    """月ごとの確率を、レースで合計3(頭数が少なければ頭数)にそろえる(アプリ・model_wf と同じ)。確率の無い行は除く"""
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


SEEDS = [int(x) for x in os.environ.get("SEEDS", "1").split(",")]
VARIANTS = (("今の材料", Xb), ("+前走の強さ", Xn))
sels = T.fixed_selects()
jra = T.of_center(rows, "jra")
jra_g = T.group(jra)


def walk(X, seed, show_imp=False):
    """早期終了で学習の回数を決め(train_model.py と同じ)、2024年1月から月ごとに学び直した確率"""
    pr = dict(params, seed=seed)
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], categorical_feature=cat, reference=dtr)
    bst = lgb.train(pr, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    it = bst.best_iteration
    if show_imp:
        imp = dict(zip(base_feats + NEW, bst.feature_importance("gain")))
        order = sorted(imp, key=lambda k: -imp[k])
        print("  効き方の順位(全{}個): ".format(len(order)) + " / ".join("{} {}位".format(k, order.index(k) + 1) for k in NEW), flush=True)
    pred = np.full(len(rows), np.nan)
    for mth in [m for m in sorted({d[:6] for d in dts}) if m >= T.WF_FROM]:
        m_tr, m_pr = dts < mth + "01", np.array([d[:6] == mth for d in dts])
        b = lgb.train(pr, lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), it)
        pred[m_pr] = b.predict(X[m_pr])
    return norm_race(pred), it


def per_race(pw):
    """今の買い方で買ったレース -> 払戻(1点100円)"""
    pick = T.pick_sel(sels["jra"], pw); out = {}
    for rid, rs in jra_g.items():
        s_ = pick(rid, rs, None)
        if not s_ or not s_[2]:
            continue
        b = "wide" if T.trio_wide(races_by_id[rid], s_[2][0]) else "trio"
        pay = (races_by_id[rid]["result"].get("payout") or {}).get("3連複" if b == "trio" else "ワイド") or {}
        out[rid] = pay.get("-".join(str(x) for x in sorted([s_[0], s_[1], s_[2][0]] if b == "trio" else [s_[0], s_[1]])), 0)
    return out


PERIODS = (("2024", "20240101", "20250101"), ("2025", "20250101", "20260101"), ("2026/1-8", "20260101", "20260901"), ("2026/9-", "20260901", "99999999"))
YRS = ("2024", "2025", "2026")
summ = {v: [] for v, _ in VARIANTS}
print("\n■ 種ごと: 今の買い方の回収率 [2024 2025 2026/1-8 2026/9-] 全体 / 対数尤度 [2024 2025 2026]", flush=True)
for seed in SEEDS:
    for name, X in VARIANTS:
        pw, it = walk(X, seed, show_imp=(seed == SEEDS[0] and name != "今の材料"))
        pr_ = per_race(pw)
        rid_date = {rid: rs[0]["date"] for rid, rs in jra_g.items()}
        rois = []
        for _, a_, b_ in PERIODS:
            v = [x for rid, x in pr_.items() if a_ <= rid_date[rid] < b_]
            rois.append((np.mean(v) if v else 0, len(v)))
        lls = []
        for yy in YRS:
            sel = [i for i, r in enumerate(rows) if r["date"][:4] == yy and (r["rid"], r["u"]) in pw]
            p = np.clip(np.array([pw[(rows[i]["rid"], rows[i]["u"])] for i in sel]), 1e-4, 1 - 1e-4); yt = y[sel]
            lls.append(float(np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))))
        tot = np.mean(list(pr_.values()))
        summ[name].append(dict(rois=rois, lls=lls, tot=tot, n=len(pr_), per=pr_))
        print("  種{} {:10s} 回数{:4d} 回収率 [{}] 全体 {:4.0f}%({}R) / 対数尤度 [{}]  ({:.0f}分)".format(
            seed, name, it, " ".join("{:4.0f}%({:3d}R)".format(r, n) for r, n in rois), tot, len(pr_),
            " ".join("{:.5f}".format(x) for x in lls), (time.time() - t0) / 60), flush=True)
print("\n■ 種の平均")
for name, _ in VARIANTS:
    S = summ[name]
    print("  {:10s} 回収率 [{}] 全体 {:4.0f}%(種ごとの幅 {:.0f}〜{:.0f}%) / 対数尤度 [{}]".format(
        name, " ".join("{:4.0f}%".format(np.mean([x["rois"][k][0] for x in S])) for k in range(len(PERIODS))),
        np.mean([x["tot"] for x in S]), min(x["tot"] for x in S), max(x["tot"] for x in S),
        " ".join("{:.5f}".format(np.mean([x["lls"][k] for x in S])) for k in range(3))))
# 種ごとに、両方が買ったレースでの差
d = [b_["tot"] - a_["tot"] for a_, b_ in zip(summ["今の材料"], summ["+前走の強さ"])]
print("  回収率の差(+前走の強さ − 今の材料)の種ごと: {}".format(" ".join("{:+.1f}".format(x) for x in d)))
# 今の材料どうし: 種が違うと買うレースがどれだけ変わるか
if len(SEEDS) >= 2:
    a0, a1 = set(summ["今の材料"][0]["per"]), set(summ["今の材料"][1]["per"])
    print("  今の材料で種1と種2の買うレース: 同じ {}R / 片方だけ {}R".format(len(a0 & a1), len(a0 ^ a1)))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

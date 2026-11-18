"""自前のコース傾向(全コース。keiba_model.course_own)を入れると、3着内の確率と買い方は良くなるか(中央)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/course_own.py
  材料は train_model.build で作り直したもの(cown_style・cown_waku・cown_gate・cown_n が入っている)。
  くらべる材料の組:
   今      … 記事の表(keiba_course.json。78コース。2021〜2025年の集計)の course_style_idx・course_waku_idx・course_kata。自前は入れない
   +自前   … 今 + 自前のコース傾向
   自前だけ … 記事の表の3つを外して、自前のコース傾向だけ
  年ごとに、その年より前だけで学んだモデルでその年を確かめる(2024・2025・2026年。学習の最後の3か月で回数を決める)。
  記事の表は 2025年までのレースには先の結果が混ざるので、混ざらないのは 2026年だけ。
  見るもの: 対数尤度・AUC(全部 / 記事の表に無いコース)、◎○▲の3着内、今の買い方(◎80%以上・◎と○の差15%以上で
  3連複◎-○-▲、▲が距離延長ならワイド◎-○)と、全レースの3連複◎-○-▲・ワイド◎-○の回収率"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import keiba_logic as L
import lightgbm as lgb
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1").split(",")]      # 種を変えた何本かの平均(アプリと同じ。SEEDS=1,2,3,4)
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
rows = [r for r in rows if r["f"].get("center") == 0]
info = {rc["id"]: {"result": rc["result"], "trio_wide_u": T.trio_wide_us(rc), "exact": bool((L.course_of(rc) or {}).get("exact")),
                   "course": "{}{}{}".format(rc.get("venue"), rc.get("surface"), rc.get("dist"))} for d, rc in races}
del races
feats = sorted({k for r in rows for k in r["f"]})
X = np.array([[M._num(r["f"].get(k)) for k in feats] for r in rows], dtype=np.float32)
y = np.array([r["y"] for r in rows])
dates = np.array([r["date"] for r in rows])
rids = np.array([r["rid"] for r in rows])
exact = np.array([info[r["rid"]]["exact"] for r in rows])
OWN = [k for k in feats if k.startswith("cown_")]
EXT = ["course_style_idx", "course_waku_idx", "course_kata"]
SETS = {"今(記事の表)": [k for k in feats if k not in OWN], "+自前": feats, "自前だけ": [k for k in feats if k not in EXT]}
print("中央 馬 {} / 特徴量 {} / 自前 {} / 自前の脚質がある {:.1%} / 記事の表の脚質がある {:.1%} / 記事の表に無いコースの馬 {:.1%} ({:.0f}分)".format(
    len(rows), len(feats), OWN, np.mean(~np.isnan(X[:, feats.index("cown_style")])), np.mean(~np.isnan(X[:, feats.index("course_style_idx")])),
    np.mean(~exact), (time.time() - t0) / 60), flush=True)
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.7,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 1}


def ll(p, yy):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


srt = lambda us: "-".join(str(x) for x in sorted(us))
RES = defaultdict(dict)
for Y in ("2024", "2025", "2026"):
    y0 = Y + "0101"
    vstart = str(int(Y) - 1) + "1001"
    tr = dates < vstart; va = (dates >= vstart) & (dates < y0); te = (dates >= y0) & (dates < str(int(Y) + 1) + "0101")
    for name, fs in SETS.items():
        ix = [feats.index(k) for k in fs]
        cat = [fs.index(k) for k in ("venue", "style") if k in fs]
        ps, imps = [], []
        for sd in SEEDS:
            dtr = lgb.Dataset(X[tr][:, ix], y[tr], categorical_feature=cat, free_raw_data=False)
            dva = lgb.Dataset(X[va][:, ix], y[va], categorical_feature=cat, reference=dtr)
            b = lgb.train(dict(params, seed=sd), dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
            b2 = lgb.train(dict(params, seed=sd), lgb.Dataset(X[tr | va][:, ix], y[tr | va], categorical_feature=cat), int(b.best_iteration * 1.1))
            ps.append(b2.predict(X[te][:, ix])); imps.append(b2.feature_importance("gain"))
        p = np.mean(ps, axis=0)
        # レースで合計3にそろえる(アプリと同じ)
        P3 = {}
        by = defaultdict(list)
        for i, (rid, r) in enumerate(zip(rids[te], [rows[j] for j in np.where(te)[0]])):
            by[rid].append((r["u"], p[i]))
        for rid, us in by.items():
            tot = sum(v for _, v in us); want = min(3, len(us))
            for u, v in us:
                P3[(rid, u)] = min(0.99, v * want / tot)
        q = np.array([P3[(rid, rows[j]["u"])] for j, rid in zip(np.where(te)[0], rids[te])])
        r = RES[Y][name] = {"ll": ll(q, y[te]), "auc": T.auc(y[te], q), "ll_noex": ll(q[~exact[te]], y[te][~exact[te]]), "auc_noex": T.auc(y[te][~exact[te]], q[~exact[te]]),
                            "imp": dict(zip(fs, np.mean(imps, axis=0)))}
        # 印と買い方
        jg = T.group([rows[j] for j in np.where(te)[0]])
        anyr = T.pick_model(0.0, -1.0, P3, True); buy = T.pick_model(0.0, -1.0, P3, True, t1=0.80, g12=0.15)
        c = defaultdict(float)
        for rid, rs in jg.items():
            s = anyr(rid, rs, None)
            if not s or not s[2]:
                continue
            res = info[rid]["result"]; top3 = set(res["order"][:3]); pay = res.get("payout") or {}
            h, t, a = s[0], s[1], s[2][0]
            c["n"] += 1; c["h3"] += h in top3; c["t3"] += t in top3; c["a3"] += a in top3; c["h1"] += res["order"][0] == h
            c["wide"] += (pay.get("ワイド") or {}).get(srt((h, t)), 0); c["trio"] += (pay.get("3連複") or {}).get(srt((h, t, a)), 0)
            c["wh"] += (pay.get("ワイド") or {}).get(srt((h, t)), 0) > 0; c["th"] += (pay.get("3連複") or {}).get(srt((h, t, a)), 0) > 0
            if buy(rid, rs, None):
                wide = T.trio_wide(info[rid], a)
                v = (pay.get("ワイド") or {}).get(srt((h, t)), 0) if wide else (pay.get("3連複") or {}).get(srt((h, t, a)), 0)
                c["bn"] += 1; c["bret"] += v; c["bh"] += v > 0
        r.update(c)
        print("{} {:10s} 対数尤度 {:.5f} AUC {:.4f} | 表に無いコース {:.5f} {:.4f} | ◎1着 {:.1%} ◎3着内 {:.1%} ○ {:.1%} ▲ {:.1%} | ワイド◎-○ {:.1%}・{:.1f}% 3連複◎○▲ {:.1%}・{:.1f}% | 今の買い方 {}R 的中 {:.1%} 回収率 {:.1f}% ({:.0f}分)".format(
            Y, name, r["ll"], r["auc"], r["ll_noex"], r["auc_noex"], c["h1"] / c["n"], c["h3"] / c["n"], c["t3"] / c["n"], c["a3"] / c["n"],
            c["wh"] / c["n"], c["wide"] / c["n"], c["th"] / c["n"], c["trio"] / c["n"], int(c["bn"]), c["bh"] / max(c["bn"], 1), c["bret"] / max(c["bn"], 1),
            (time.time() - t0) / 60), flush=True)
print("\n■ 3年まとめ")
for name in SETS:
    rs = [RES[Y][name] for Y in RES]
    n = sum(r["n"] for r in rs); bn = sum(r["bn"] for r in rs)
    print("  {:10s} 対数尤度(平均) {:.5f} AUC {:.4f} | 表に無いコース {:.5f} | ◎3着内 {:.1%} ワイド◎-○ {:.1f}% 3連複◎○▲ {:.1f}% | 今の買い方 {}R 回収率 {:.1f}% [{}]".format(
        name, np.mean([r["ll"] for r in rs]), np.mean([r["auc"] for r in rs]), np.mean([r["ll_noex"] for r in rs]),
        sum(r["h3"] for r in rs) / n, sum(r["wide"] for r in rs) / n, sum(r["trio"] for r in rs) / n, int(bn), sum(r["bret"] for r in rs) / max(bn, 1),
        " ".join("{:.0f}".format(r["bret"] / max(r["bn"], 1)) for r in rs)))
print("\n■ 2026年の +自前 で効いている度合い(gain の順位)")
imp = RES["2026"]["+自前"]["imp"]
order = sorted(imp, key=lambda k: -imp[k])
print("  " + " / ".join("{} {}位".format(k, order.index(k) + 1) for k in OWN + EXT))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

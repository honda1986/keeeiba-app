"""当日の偏りの材料を足した3着内のモデル(月ごと。2024年1月〜)。train_model と同じ学び方"""
import gzip, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
SP = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
_, rows, _, _ = pickle.load(open(T.CACHE, "rb"))
B = pickle.load(open(SP + "/feat.pkl", "rb"))
for r in rows:
    r["f"] = dict(r["f"], **(B.get((r["rid"], r["u"])) or {}))
feats = sorted({k for r in rows for k in r["f"]})
print("特徴量", len(feats), flush=True)
X = np.empty((len(rows), len(feats)))
for i, r in enumerate(rows):
    X[i] = [M._num(r["f"].get(k)) for k in feats]
y = np.array([r["y"] for r in rows])
dates = np.array([r["date"] for r in rows])
cat = [feats.index(k) for k in ("venue", "style") if k in feats]
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 1}
tr, va = dates < T.VALID_FROM, (dates >= T.VALID_FROM) & (dates < T.TEST_FROM)
bst = lgb.train(params, lgb.Dataset(X[tr], y[tr], categorical_feature=cat), 3000,
                valid_sets=[lgb.Dataset(X[va], y[va], categorical_feature=cat)], callbacks=[lgb.early_stopping(150, verbose=False)])
it = bst.best_iteration
imp = sorted(zip(bst.feature_importance("gain"), feats), reverse=True)
print("反復", it, "偏りの材料の順位", [(k, i + 1, int(v)) for i, (v, k) in enumerate(imp) if k.startswith(("bias", "fit_", "prev_"))], flush=True)
wf = {}
for mth in sorted({d[:6] for d in dates if d >= "20240101"}):
    t0 = time.time()
    m_tr, m_pr = dates < mth + "01", np.array([d[:6] == mth for d in dates])
    b = lgb.train(params, lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), it)
    pm = b.predict(X[m_pr])
    byr = {}
    for r, v in zip([r for r, k in zip(rows, m_pr) if k], pm):
        byr.setdefault(r["rid"], {})[str(r["u"])] = float(v)
    for rid, d in byr.items():
        tot = sum(d.values()); want = min(3, len(d))
        wf[rid] = {u: round(min(0.99, v * want / tot), 4) for u, v in d.items()}
    print(mth, len(byr), "レース {:.0f}秒".format(time.time() - t0), flush=True)
with gzip.open(SP + "/wf_bias.json.gz", "wt") as f:
    json.dump({"probs": wf}, f)
print("保存")

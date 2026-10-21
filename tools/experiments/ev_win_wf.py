"""1着になる確率のモデル(月ごとに、その月より前だけで学ぶ)。特徴量は3着内のモデルと同じ
  KEIBA_CACHE=<train_model の特徴量の表> KEIBA_WORK=<作業フォルダ> python tools/experiments/ev_win_wf.py
  そのあと python tools/experiments/ev_backtest.py(期待値で買ったときの年ごとの成績)"""
import os, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
SP = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
_, rows, _, _ = pickle.load(open(T.CACHE, "rb"))
os.makedirs(SP + "/ev", exist_ok=True)
order = {rc["id"]: rc["result"]["order"] for _, rc in T.load_races()}
feats = sorted({k for r in rows for k in r["f"]})
X = np.empty((len(rows), len(feats)))
for i, r in enumerate(rows):
    X[i] = [M._num(r["f"].get(k)) for k in feats]
yw = np.array([1 if order[r["rid"]][0] == r["u"] else 0 for r in rows])
dates = np.array([r["date"] for r in rows])
cat = [feats.index(k) for k in ("venue", "style") if k in feats]
params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "seed": 1}
tr, va = dates < T.VALID_FROM, (dates >= T.VALID_FROM) & (dates < T.TEST_FROM)
bst = lgb.train(params, lgb.Dataset(X[tr], yw[tr], categorical_feature=cat), 3000,
                valid_sets=[lgb.Dataset(X[va], yw[va], categorical_feature=cat)], callbacks=[lgb.early_stopping(150, verbose=False)])
it = bst.best_iteration
print("反復", it, flush=True)
P = {}
for mth in sorted({d[:6] for d in dates if d >= "20240101"}):
    t0 = time.time()
    m_tr, m_pr = dates < mth + "01", np.array([d[:6] == mth for d in dates])
    b = lgb.train(params, lgb.Dataset(X[m_tr], yw[m_tr], categorical_feature=cat), it)
    pm = b.predict(X[m_pr])
    by = {}
    for r, v in zip([r for r, k in zip(rows, m_pr) if k], pm):
        by.setdefault(r["rid"], {})[r["u"]] = float(v)
    for rid, d in by.items():
        s = sum(d.values())
        P[rid] = {u: v / s for u, v in d.items()}
    print(mth, len(by), "レース {:.0f}秒".format(time.time() - t0), flush=True)
pickle.dump(P, open(SP + "/ev/wf_win.pkl", "wb"))
ys = [yw[i] for i, r in enumerate(rows) if r["rid"] in P]
ps = [P[r["rid"]][r["u"]] for r in rows if r["rid"] in P]
print("1着のAUC(月ごと)", round(T.auc(ys, ps), 4))

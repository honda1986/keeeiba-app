"""騎手の得意な競馬場・コース・距離帯(それまでの成績)を材料に足したときを、月ごとの学び直しで比べる(中央)
  KEIBA_WORK=<作業フォルダ> python tools/experiments/jockey_review.py build   # 材料を作る(今のコードで)
  KEIBA_WORK=<作業フォルダ> python tools/experiments/jockey_review.py wf      # base(新しい材料を外す)/ jk(入れる)
  確率は <作業フォルダ>/wf_jbase.json.gz / wf_jk.json.gz。比べるのは rule_report.py と同じやり方(jockey_report)"""
import gzip, json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, ".."))
WORK = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
os.makedirs(WORK, exist_ok=True)
os.environ.setdefault("KEIBA_CACHE", os.path.join(WORK, "dataset_jk.pkl"))
os.environ["KEIBA_JOCKEY_CTX"] = "1"          # 新しい材料も作る(base はそれを外して学ぶ)
import train_model as T   # noqa: E402
import keiba_model as M   # noqa: E402
NEW = ("jockey_venue_", "jockey_course_", "jockey_dist_")
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
          "verbose": -1, "seed": 1, "num_threads": 4}


def build():
    races = T.load_races()
    norm = T.l3f_norms(races)
    sp = T.speed_table(races, T.VALID_FROM)
    rows, jt_all = T.build(races, norm, sp)
    pickle.dump((norm, rows, jt_all, sp), open(T.CACHE, "wb"))
    print("作った", len(rows), "頭", flush=True)


def wf():
    import lightgbm as lgb
    _, rows, _, _ = pickle.load(open(T.CACHE, "rb"))
    allf = sorted({k for r in rows for k in r["f"]})
    y = np.array([r["y"] for r in rows]); dates = np.array([r["date"] for r in rows])
    for name, feats in (("jbase", [k for k in allf if not k.startswith(NEW)]), ("jk", allf)):
        out = os.path.join(WORK, "wf_%s.json.gz" % name)
        if os.path.exists(out):
            continue
        X = np.empty((len(rows), len(feats)), dtype=np.float32)
        for i, r in enumerate(rows):
            X[i] = [M._num(r["f"].get(k)) for k in feats]
        cat = [feats.index(k) for k in ("venue", "style") if k in feats]
        tr, va = dates < T.VALID_FROM, (dates >= T.VALID_FROM) & (dates < T.TEST_FROM)
        b = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], categorical_feature=cat), 3000,
                      valid_sets=[lgb.Dataset(X[va], y[va], categorical_feature=cat)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        it = b.best_iteration
        imp = sorted(zip(feats, b.feature_importance("gain")), key=lambda x: -x[1])
        print(name, "材料", len(feats), "反復", it, flush=True)
        probs = {}
        for mth in sorted({d[:6] for d in dates if d >= T.WF_FROM + "01"}):
            t0 = time.time()
            m_tr, m_pr = dates < mth + "01", np.array([d[:6] == mth for d in dates])
            pm = lgb.train(PARAMS, lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), it).predict(X[m_pr])
            byr = {}
            for r, v in zip((r for r, k in zip(rows, m_pr) if k), pm):
                byr.setdefault(r["rid"], {})[str(r["u"])] = float(v)
            for rid, d in byr.items():
                tot, want = sum(d.values()), min(3, len(d))
                probs[rid] = {u: round(min(0.99, v * want / tot), 4) for u, v in d.items()}
            print(" ", name, mth, len(byr), "レース", round(time.time() - t0), "秒", flush=True)
        with gzip.open(out, "wt", encoding="utf-8") as f:
            json.dump({"iter": it, "imp": [(k, round(float(v))) for k, v in imp], "probs": probs}, f)
        print("保存", out, flush=True)


if __name__ == "__main__":
    {"build": build, "wf": wf}[sys.argv[1]]()

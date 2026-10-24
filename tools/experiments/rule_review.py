"""加点・減点・軸不可・買う基準の見直し(中央)。
  1. 材料を作る: 今の材料に、加点・減点を1つずつの材料(tag:〜 = その点)として足す
       KEIBA_WORK=<作業フォルダ> python tools/experiments/rule_review.py build
  2. 月ごとに学ぶ(2024年1月〜。その月より前だけで学ぶ)。今の材料だけ(base)と、1つずつ足したもの(tag)の2通り
       KEIBA_WORK=<作業フォルダ> python tools/experiments/rule_review.py wf
  確率は <作業フォルダ>/wf_base.json.gz / wf_tag.json.gz。成績は rule_report.py で出す"""
import gzip, json, os, pickle, re, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, ".."))
WORK = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
os.makedirs(WORK, exist_ok=True)
os.environ.setdefault("KEIBA_CACHE", os.path.join(WORK, "dataset_tag.pkl"))
import train_model as T   # noqa: E402
import keiba_model as M   # noqa: E402


def tag_key(t):
    """札の文字から数字・かっこを外した名前(同じ種類の札をまとめる)"""
    t = re.sub(r"[((].*?[))]", "", t)
    t = re.sub(r"[+\-−]?\d+(\.\d+)?(kg|m|日)?", "", t)
    t = t.replace("右回り", "回り").replace("左回り", "回り").replace("つ重複", "重複")
    return t.strip(" :")


def patch():
    orig = M.race_rows

    def race_rows(race, today, pred, l3f_norm, jt=None, sp=None):
        rows = orig(race, today, pred, l3f_norm, jt, sp)
        pe = {x["umaban"]: x for x in (pred or {}).get("horses", [])}
        for u, f in rows:
            e = pe.get(u, {})
            for m in e.get("minus", []) + e.get("plus", []):
                k = "tag:" + tag_key(m["t"])
                f[k] = f.get(k, 0) + m["p"]
            for a in e.get("axis_ng", []):
                f["axis:" + a] = 1
        return rows
    M.race_rows = race_rows


def build():
    patch()
    races = T.load_races()
    print("レース", len(races), races[0][0], "〜", races[-1][0], flush=True)
    t0 = time.time()
    norm = T.l3f_norms(races)
    sp = T.speed_table(races, T.VALID_FROM)
    rows, jt_all = T.build(races, norm, sp)
    pickle.dump((norm, rows, jt_all, sp), open(T.CACHE, "wb"))
    print("作った", len(rows), "頭", round(time.time() - t0), "秒", flush=True)


PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
          "verbose": -1, "seed": 1, "num_threads": 4}


def wf():
    import lightgbm as lgb
    _, rows, _, _ = pickle.load(open(T.CACHE, "rb"))
    allf = sorted({k for r in rows for k in r["f"]})
    y = np.array([r["y"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    want = os.environ.get("KEIBA_SETS", "base,tag").split(",")
    for name, feats in (("base", [k for k in allf if not k.startswith(("tag:", "axis:"))]), ("tag", allf)):
        if name not in want:
            continue
        out = os.path.join(WORK, "wf_%s%s.json.gz" % (name, os.environ.get("KEIBA_SUFFIX", "")))
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
            bm = lgb.train(PARAMS, lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), it)
            pm = bm.predict(X[m_pr])
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

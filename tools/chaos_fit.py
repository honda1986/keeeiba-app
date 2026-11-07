"""中央の荒れ度(1番人気が3着に来ない確率)の式(keiba/chaos_model.json)を作る。アプリがオッズから計算して出す(参考。買い方には使わない)
  python tools/chaos_fit.py
  対象のレース(結果と確定オッズのあるもの。2022年〜)で、1番人気が3着に来なかったかを
  log(1番人気のオッズ)とオッズのばらけ方(keiba_final.chaos_x)のロジスティック回帰(L2=1)で当てる。
  高知の最終レースの荒れ度は tools/kochi_final_fit.py(final_model.json の chaos)。毎週の学び直しで作り直す"""
import glob, json, os, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import keiba as K
import keiba_final as F

BANDS = ((0, .2), (.2, .3), (.3, .4), (.4, .55), (.55, 1.01))
X, Y, YR = [], [], []
for f in sorted(glob.glob(K.OUT + "/days/20*.json")):
    for r in (json.load(open(f, encoding="utf-8")).get("races") or {}).values():
        res = r.get("result") or {}
        if not K.target(r["id"]) or len(res.get("order") or []) < 3:
            continue
        odds = {int(u): v for u, v in (res.get("odds") or {}).items() if v}
        x = F.chaos_x(odds)
        if not x:
            continue
        fav = min((u for u in odds if odds[u] > 1), key=lambda u: odds[u])
        X.append(x); Y.append(int(fav not in res["order"][:3])); YR.append(os.path.basename(f)[:4])
X, Y, YR = np.array(X), np.array(Y, float), np.array(YR)
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd
w = F.fit_logit(Z, Y)
# 年ごとに、ほかの年で学んだ式で読んだ荒れ度の段ごとの、実際に1番人気が3着外だった率(見込みと比べる)
cv = np.zeros(len(Y))
for yy in sorted(set(YR)):
    w2 = F.fit_logit(Z[YR != yy], Y[YR != yy])
    cv[YR == yy] = 1 / (1 + np.exp(-(w2[0] + Z[YR == yy] @ w2[1:])))
bands = []
for lo, hi in BANDS:
    m = (cv >= lo) & (cv < hi)
    bands.append({"lo": lo, "hi": min(hi, 1.0), "races": int(m.sum()),
                  "actual": round(float(Y[m].mean()), 3) if m.any() else None,
                  "expect": round(float(cv[m].mean()), 3) if m.any() else None,
                  "by_year": {str(yy): round(float(Y[m & (YR == yy)].mean()), 3) for yy in sorted(set(YR)) if (m & (YR == yy)).sum() >= 20}})
model = {"x": ["log(1番人気のオッズ)", "オッズのばらけ方"], "mu": mu.round(5).tolist(), "sd": sd.round(5).tolist(),
         "w": w.round(5).tolist(), "fitted": datetime.now(K.JST).strftime("%Y-%m-%d"), "races": int(len(Y)),
         "years": [min(YR.tolist()), max(YR.tolist())], "fav_out": round(float(Y.mean()), 3), "bands": bands}
json.dump({"chaos": model}, open(os.path.join(K.OUT, "chaos_model.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("保存", os.path.join(K.OUT, "chaos_model.json"), "{}R 係数 {}".format(len(Y), w.round(2)))
for b in bands:
    print("  荒れ度 {:3.0%}〜{:3.0%} {:5d}R 実際 {:.0%} / 見込み {:.0%} {}".format(b["lo"], b["hi"], b["races"], b["actual"], b["expect"], b["by_year"]))

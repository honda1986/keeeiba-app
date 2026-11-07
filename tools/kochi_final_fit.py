"""高知の最終レース用の確率の式(keiba/kochi/final_model.json)を作る。keiba_final.py が使う
  KEIBA_OUT=keiba/kochi KEIBA_CENTERS=nar KEIBA_NAR_VENUES=高知 python tools/kochi_final_fit.py
  2024年からの最終レース(月ごとの確率で作り直した「再現」の予想)で、3着内に来たかを当てるロジスティック回帰(L2)。
  年ごとに分け、ほかの年で学んだ式の当たり方も出す"""
import glob, json, os, sys
from datetime import datetime
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import keiba as K
import keiba_final as F


fit = lambda X, y, l2=2.0: F.fit_logit(X, y, l2)


X, y, yr, rid = [], [], [], []
for f in sorted(glob.glob(K.OUT + "/days/202[456789]*.json")):
    d = os.path.basename(f)[:8]
    day = K.load_day(d)
    rs = [r for r in day["races"].values() if K.target(r["id"]) and r.get("rno")]
    if not rs:
        continue
    r = max(rs, key=lambda r: r["rno"])
    p, res = r.get("pred") or {}, r.get("result") or {}
    if not p.get("model") or len(res.get("order") or []) < 3:
        continue
    hs = [dict(h, p3=h.get("p3_model", h.get("p3"))) for h in p["horses"]]      # 今のモデルの確率で
    odds = {int(u): v for u, v in (res.get("odds") or {}).items() if v}
    fe = F.features(r, hs, odds)
    top3 = set(res["order"][:3])
    for u, fx in fe.items():
        X.append([fx[k] for k in F.FEATS]); y.append(int(u in top3)); yr.append(d[:4]); rid.append(r["id"])
X, y, yr = np.array(X, float), np.array(y, float), np.array(yr)
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd
ll = {}
for yy in sorted(set(yr)):
    w = fit(Xs[yr != yy], y[yr != yy]); p = 1 / (1 + np.exp(-(w[0] + Xs[yr == yy] @ w[1:])))
    ll[yy] = float(np.mean(y[yr == yy] * np.log(p) + (1 - y[yr == yy]) * np.log(1 - p)))
w = fit(Xs, y)
# 荒れ度: 1番人気が3着に来ないかを、1番人気のオッズとオッズのばらけ方から(2023年からの最終レース。オッズは確定)
CX, CY, CYR = [], [], []
for f in sorted(glob.glob(K.OUT + "/days/20*.json")):
    day = json.load(open(f))
    rs = [r for r in day["races"].values() if K.target(r["id"]) and r.get("rno")]
    if not rs:
        continue
    r = max(rs, key=lambda r: r["rno"]); res = r.get("result") or {}
    odds = {int(u): v for u, v in (res.get("odds") or {}).items() if v}
    x = F.chaos_x(odds)
    if not x or len(res.get("order") or []) < 3:
        continue
    fav = min((u for u in odds if odds[u] > 1), key=lambda u: odds[u])
    CX.append(x); CY.append(int(fav not in res["order"][:3])); CYR.append(os.path.basename(f)[:4])
CX, CY, CYR = np.array(CX), np.array(CY, float), np.array(CYR)
cmu, csd = CX.mean(0), CX.std(0) + 1e-9
cw = fit((CX - cmu) / csd, CY, 1.0)
cby = {}
for yy in sorted(set(CYR)):
    w2 = fit(((CX - cmu) / csd)[CYR != yy], CY[CYR != yy], 1.0)
    pz = 1 / (1 + np.exp(-(w2[0] + ((CX - cmu) / csd)[CYR == yy] @ w2[1:])))
    cby[yy] = {"races": int((CYR == yy).sum()), "actual": round(float(CY[CYR == yy].mean()), 3),
               "high_actual": round(float(CY[CYR == yy][pz >= .55].mean()), 3) if (pz >= .55).any() else None,
               "low_actual": round(float(CY[CYR == yy][pz < .4].mean()), 3) if (pz < .4).any() else None}
model = {"feats": F.FEATS, "mu": mu.round(5).tolist(), "sd": sd.round(5).tolist(), "w": w.round(5).tolist(),
         "fitted": datetime.now().strftime("%Y-%m-%d"), "races": len(set(rid)), "horses": int(len(y)), "ll_by_year": ll,
         "chaos": {"x": ["log(1番人気のオッズ)", "オッズのばらけ方"], "mu": cmu.round(5).tolist(), "sd": csd.round(5).tolist(),
                   "w": cw.round(5).tolist(), "races": int(len(CY)), "by_year": cby}}
json.dump(model, open(os.path.join(K.OUT, "final_model.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("保存", os.path.join(K.OUT, "final_model.json"), "最終レース", len(set(rid)), "頭", len(y))
print("係数", " / ".join("{} {:+.2f}".format(k, v) for k, v in zip(F.FEATS, w[1:])))
print("年ごと(ほかの年で学んだ式)の対数尤度", ll)
print("荒れ度: 係数", cw.round(2), "年ごと(ほかの年で学んだ式で、高=55%以上 / 低=40%未満 と読んだレースの、1番人気が3着外の率)", cby)

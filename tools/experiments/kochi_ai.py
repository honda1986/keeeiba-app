"""高知の最終レース: たくさんの特徴を機械学習(LightGBM)で組み合わせて、人気が見落としている馬を探す(keiba/kochi。2024年〜)
  KEIBA_OUT=keiba/kochi KEIBA_CENTERS=nar KEIBA_NAR_VENUES=高知 python tools/experiments/kochi_ai.py
  学ぶのは高知の全レース(最終レースかどうかも特徴に入れる)。年ごとに分け、ほかの2年で学んだモデルで、その年を当てる。
  騎手・厩舎の成績は、そのレースの日より前の分だけ(先の結果を使わない)"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
import lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import keiba as K
import keiba_final as F
YS = ("2024", "2025", "2026")
lg = lambda v: math.log(min(max(v, 1e-4), 1 - 1e-4) / (1 - min(max(v, 1e-4), 1 - 1e-4)))
rel = lambda x, n: (x - 1) / max(n - 1, 1) if x and n else np.nan
BABA = {"良": 0, "稍": 1, "重": 2, "不": 3}

jock = defaultdict(lambda: [0, 0]); trn = defaultdict(lambda: [0, 0]); pair = defaultdict(lambda: [0, 0])
rows = []
for f in sorted(glob.glob(K.OUT + "/days/20*.json")):
    d = os.path.basename(f)[:8]
    day = K.load_day(d)
    rs = [r for r in day["races"].values() if K.target(r["id"]) and r.get("rno")]
    if not rs:
        continue
    last = max(r["rno"] for r in rs)
    done = []
    for r in rs:
        res = r.get("result") or {}
        if len(res.get("order") or []) < 3:
            continue
        top3 = set(res["order"][:3]); pay = res.get("payout") or {}
        od = {int(u): v for u, v in (res.get("odds") or {}).items() if v and v > 1}
        nk = {int(u): v for u, v in (res.get("ninki") or {}).items()}
        card = [h for h in r.get("horses") or [] if not h.get("scratched") and h["umaban"] in od]
        done.append((r, top3))
        p = r.get("pred") or {}
        if d < "20240101" or not p.get("model") or len(card) < 5:
            continue
        ph = {h["umaban"]: h for h in p["horses"]}
        us = [h["umaban"] for h in card]
        inv = sum(1 / od[u] for u in us); q = [1 / od[u] / inv for u in us]; mt3 = F.pl_top3(q)
        n = len(us)
        for i, h in enumerate(card):
            u = h["umaban"]; ps = (h.get("past") or [])[:5]; e = ph.get(u, {})
            rr = [rel(x.get("rank"), x.get("n")) for x in ps]; rn = [rel(x.get("ninki"), x.get("n")) for x in ps]
            pf = [rel((x.get("pass") or [None])[0], x.get("n")) for x in ps]; pl = [rel((x.get("pass") or [None])[-1], x.get("n")) for x in ps]
            j, t = h.get("jockey") or "", h.get("trainer") or ""
            jr, tr, pr = jock[j], trn[t], pair[(j, t)]
            l0 = ps[0] if ps else {}
            rows.append(dict(
                y=d[:4], d=d, rid=r["id"], u=u, top=int(u in top3), final=int(r["rno"] == last), n=n, nk=nk.get(u, 99), odds=od[u],
                plc=(pay.get("複勝") or {}).get(str(u), 0) if "複勝" in pay else None,
                lmt3=lg(mt3[i]), lq=math.log(q[i]), nk_rel=rel(nk.get(u), n), lp3=lg(e.get("p3", .2)),
                r1=rr[0] if rr else np.nan, r_mean=np.nanmean(rr) if rr else np.nan, r_best=np.nanmin(rr) if rr else np.nan,
                r_trend=(rr[0] - np.nanmean(rr[1:])) if len(rr) >= 3 else np.nan,
                nk1=rn[0] if rn else np.nan, beat_nk=sum(1 for a, b in zip(rr, rn) if a == a and b == b and a < b) / max(len(rr), 1) if rr else np.nan,
                mg1=min(max(l0.get("margin") or 0, 0), 5) if l0 else np.nan,
                mg_mean=np.nanmean([min(max(x.get("margin") or 0, 0), 5) for x in ps]) if ps else np.nan,
                pos1=pf[0] if pf else np.nan, gain1=(pf[0] - pl[0]) if pf and pf[0] == pf[0] and pl[0] == pl[0] else np.nan,
                gain_mean=np.nanmean([a - b for a, b in zip(pf, pl) if a == a and b == b]) if pf else np.nan,
                cls_diff=(l0.get("cls") or 0) - (r.get("cls") or 0) if l0 else np.nan,
                dist_diff=((r.get("dist") or 0) - (l0.get("dist") or 0)) / 100 if l0 else np.nan,
                other_venue=int(bool(l0) and l0.get("venue") != "高知"), kochi_runs=sum(1 for x in ps if x.get("venue") == "高知"),
                rest=min(e.get("rest_days") or 999, 999), nruns=e.get("nruns") or 0,
                weight=h.get("weight") or np.nan, wdiff=h.get("wdiff") if h.get("wdiff") is not None else np.nan,
                waku=rel(u, n), cls=r.get("cls") or 0, dist=r.get("dist") or 0, baba=BABA.get(r.get("baba"), 0),
                j_rate=(jr[1] + 3) / (jr[0] + 10), j_n=math.log1p(jr[0]), t_rate=(tr[1] + 3) / (tr[0] + 10), t_n=math.log1p(tr[0]),
                jt_rate=(pr[1] + 3) / (pr[0] + 10), jt_n=math.log1p(pr[0])))
    # この日の結果で騎手・厩舎の成績を足す(次の日から使う)
    for r, top3 in done:
        for h in r.get("horses") or []:
            if h.get("scratched"):
                continue
            k = int(h["umaban"] in top3)
            for dct, key in ((jock, h.get("jockey") or ""), (trn, h.get("trainer") or ""), (pair, (h.get("jockey") or "", h.get("trainer") or ""))):
                dct[key][0] += 1; dct[key][1] += k

FE = ["lmt3", "lq", "nk_rel", "lp3", "r1", "r_mean", "r_best", "r_trend", "nk1", "beat_nk", "mg1", "mg_mean", "pos1", "gain1", "gain_mean",
      "cls_diff", "dist_diff", "other_venue", "kochi_runs", "rest", "nruns", "weight", "wdiff", "waku", "n", "cls", "dist", "baba",
      "j_rate", "j_n", "t_rate", "t_n", "jt_rate", "jt_n", "final"]
X = np.array([[x[k] for k in FE] for x in rows], float); y = np.array([x["top"] for x in rows]); yr = np.array([x["y"] for x in rows])
fin = np.array([x["final"] for x in rows]) == 1; rid = np.array([x["rid"] for x in rows])
print("高知 2024〜: {}頭(最終レース {}頭)".format(len(rows), int(fin.sum())))


def auc(s, t):
    o = np.argsort(s, kind="mergesort"); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1); pos = t.sum()
    return (rk[t == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(t) - pos))


def norm3(p):
    out = p.copy()
    for r in set(rid):
        m = rid == r; out[m] = np.minimum(.97, p[m] * 3 / p[m].sum())
    return out


P = dict(objective="binary", learning_rate=.03, num_leaves=15, min_data_in_leaf=60, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=7)
pred = np.zeros(len(rows)); mkt = np.zeros(len(rows)); imp = np.zeros(len(FE))
for t in YS:
    trm, tem = yr != t, yr == t
    # 木の数は学ぶ年の中で決める(後ろ2割で)
    idx = np.where(trm)[0]; cut = idx[int(len(idx) * .8)]
    m = lgb.train(P, lgb.Dataset(X[idx[idx < cut]], y[idx[idx < cut]]), 3000, valid_sets=[lgb.Dataset(X[idx[idx >= cut]], y[idx[idx >= cut]])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    m = lgb.train(P, lgb.Dataset(X[trm], y[trm]), max(100, int(m.best_iteration * 1.1)))
    pred[tem] = m.predict(X[tem]); imp += m.feature_importance("gain")
    mm = lgb.train(dict(P, num_leaves=4), lgb.Dataset(X[trm][:, :1], y[trm]), 300)        # 市場だけ(比べる)
    mkt[tem] = mm.predict(X[tem][:, :1])
pred = norm3(pred); mkt = norm3(mkt)
ll = lambda p, m: float(np.mean(y[m] * np.log(np.clip(p[m], 1e-4, 1)) + (1 - y[m]) * np.log(np.clip(1 - p[m], 1e-4, 1))))
print("\n■ 当たり方(年ごとに、ほかの2年で学んだモデル)        最終レース: AUC / 対数尤度   ‖ 全レース: AUC")
for lab, p in (("市場(単勝オッズ)だけ", mkt), ("機械学習(全部の特徴)", pred)):
    print("  {:22s} {:.4f} / {:.4f} [{}] ‖ {:.4f}".format(lab, auc(p[fin], y[fin]), ll(p, fin),
          " ".join("{:.3f}".format(auc(p[fin & (yr == t)], y[fin & (yr == t)])) for t in YS), auc(p, y)))
print("  (くらべる: 最終レース用の式 AUC 0.699 / 今のモデル 0.645)")

print("\n■ 効いている特徴(上位15)")
for k, v in sorted(zip(FE, imp), key=lambda z: -z[1])[:15]:
    print("  {:12s} {:.0f}".format(k, v))

print("\n■ 人気が見落としている馬: 機械学習の確率が市場の見込みの k 倍以上(最終レース・年ごと)")
mt3 = np.array([1 / (1 + math.exp(-x["lmt3"])) for x in rows])
for nmin, k in ((1, 1.3), (1, 1.5), (4, 1.3), (4, 1.5), (6, 1.3), (6, 1.5), (6, 2.0)):
    sel = fin & (np.array([x["nk"] for x in rows]) >= nmin) & (pred >= k * mt3)
    cells = []
    for t in YS:
        s = sel & (yr == t)
        pl = [x["plc"] for x, m in zip(rows, s) if m and x["plc"] is not None]
        cells.append("{} {:3d}頭 3着内 {:3.0%}(市場の見込み {:3.0%}) 複勝 {:3.0f}%".format(t, int(s.sum()), y[s].mean() if s.any() else 0, mt3[s].mean() if s.any() else 0,
                                                                                np.mean(pl) if pl else 0))
    pl = [x["plc"] for x, m in zip(rows, sel) if m and x["plc"] is not None]
    print("  {}番人気以下 {:.1f}倍以上: {:3d}頭 3着内 {:3.0%} / 見込み {:3.0%} / 複勝 {:3.0f}%  [{}]".format(
        nmin, k, int(sel.sum()), y[sel].mean() if sel.any() else 0, mt3[sel].mean() if sel.any() else 0, np.mean(pl) if pl else 0, " | ".join(cells)))
import pickle
pickle.dump(dict(rows=rows, pred=pred, mkt=mkt, FE=FE), open(os.environ.get("KEIBA_TMP", "/tmp") + "/kochi_ai.pkl", "wb"))

"""割安ワイドの選び方を良くできるか(中央 2022〜2026年。確定オッズ)。式は4年で作り、残りの1年で確かめる(5回)
  OUTDIR=/path/to/dir python tools/experiments/gap_model.py
  今の選び方 … 割安度 = 0.75×組の強さ ÷ (0.775÷ワイドの中央のオッズ)。1レースでいちばん高い組が 1.3 以上なら買う
  足す工夫:
   - ワイドの払戻の見込み: 当たったときの払戻は「下限〜上限」の下限寄りになる(下限 + α×幅。α は学ぶ年で決める)
   - 単勝オッズから出した「2頭とも3着内」の確率(Harville。2・3着は 0.7 乗でならす)
   - 馬ごとのずれ(その馬を含む組の割安度の平均)と、その組だけのずれ(割安度 − 2頭の馬ごとのずれ)
   - 頭数・レース番号
  当たる確率を、ロジスティック回帰と LightGBM で出し、期待値 = 確率 × 払戻の見込み で選ぶ。
  確かめる年の、期待値の帯ごとの回収率と、1レースで期待値がいちばん高い組だけを買ったときの回収率を、今の選び方とくらべる"""
import glob, json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
R = os.path.join(HERE, "..", "..")
OUT = os.environ.get("OUTDIR", "/tmp/gap_model"); os.makedirs(OUT, exist_ok=True)
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()


def harville_pairs(q, g=0.7):
    """単勝の確率 q(合計1)-> 2頭とも3着内の確率の行列。2・3着は q の g 乗でならす(ev_fit.py と同じ考え方)"""
    n = len(q); q2 = q ** g; q2 = q2 / q2.sum()
    a = q[:, None, None]; b = q2[None, :, None]; c = q2[None, None, :]
    den2 = 1 - q2[:, None, None]; den3 = 1 - q2[:, None, None] - q2[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = a * (b / den2) * (c / den3)
    i, j, k = np.indices((n, n, n))
    p[(i == j) | (j == k) | (i == k) | ~np.isfinite(p)] = 0
    s = p + p.transpose(0, 2, 1) + p.transpose(1, 0, 2) + p.transpose(1, 2, 0) + p.transpose(2, 0, 1) + p.transpose(2, 1, 0)     # 3頭の組(順不同)。各組が6回
    return s.sum(axis=2) / 2          # [a, b] = Σ_x P({a,b,x})(x を a,b と入れ替えた分が2重)


f = os.path.join(OUT, "pairs.pkl")
if os.path.exists(f):
    D = pickle.load(open(f, "rb"))
else:
    cols = defaultdict(list)
    nr = 0
    for fn in sorted(glob.glob(R + "/keiba/days/20*.json")):
        d = os.path.basename(fn)[:8]
        for rid, r in (json.load(open(fn)).get("races") or {}).items():
            tp, wf, res = r.get("trio_pair_final"), r.get("wide_final"), r.get("result") or {}
            if r.get("center") != "jra" or not tp or not wf or len(res.get("order") or []) < 3:
                continue
            od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
            out = {h["umaban"] for h in r.get("horses") or [] if h.get("scratched")}
            us = sorted(u for u in od if u not in out)
            if len(us) < 5:
                continue
            q = np.array([1 / od[u] for u in us]); q = q / q.sum()
            H = harville_pairs(q); ix = {u: i for i, u in enumerate(us)}
            top3 = set(res["order"][:3]); pay = (res.get("payout") or {}).get("ワイド") or {}
            pr = []
            for k, v in wf.items():
                a, b = (int(x) for x in k.split("-"))
                lo, hi = v[0], v[1]
                if not lo or lo <= 1.0 or k not in tp or a not in ix or b not in ix:
                    continue
                mid = (lo + hi) / 2
                pr.append((a, b, lo, hi, np.log(0.75 * tp[k] * mid / 0.775), 0.75 * tp[k], H[ix[a], ix[b]], q[ix[a]], q[ix[b]], pay.get(k, 0), int(a in top3 and b in top3)))
            if len(pr) < 6:
                continue
            hs = defaultdict(list)       # 馬ごとのずれ(その馬を含む、ワイド50倍未満の組の割安度の log の平均)
            for a, b, lo, hi, lr, *_ in pr:
                if (lo + hi) / 2 < 50:
                    hs[a].append(lr); hs[b].append(lr)
            hm = {u: float(np.mean(v)) for u, v in hs.items()}
            nr += 1
            for a, b, lo, hi, lr, pt, hv, qa, qb, p, hit in pr:
                if not (1.3 <= lo < 30):
                    continue
                ia, ib = hm.get(a, 0.0), hm.get(b, 0.0)
                for key, val in (("y", d[:4]), ("race", nr), ("lo", lo), ("hi", hi), ("lr", lr), ("pt", pt), ("hv", hv), ("qmax", max(qa, qb)), ("qmin", min(qa, qb)),
                                 ("hsum", ia + ib), ("hres", lr - (ia + ib) / 2), ("n", len(us)), ("rno", r.get("rno") or 0), ("pay", p), ("hit", hit)):
                    cols[key].append(val)
    D = {k: np.array(v) for k, v in cols.items()}
    pickle.dump(D, open(f, "wb"))
N = len(D["hit"])
print("中央 {}R・ワイドの組 {}点(下限 1.3〜30倍)・当たり {}  ({:.0f}秒)".format(len(set(D["race"])), N, int(D["hit"].sum()), time.time() - t0), flush=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
mid = (D["lo"] + D["hi"]) / 2
FE = {"3連複の見込み": lg(D["pt"]), "ワイドの見込み(下限)": lg(np.clip(0.775 / D["lo"], 0, .98)), "ワイドの幅": np.log(D["hi"] / D["lo"]),
      "単勝からの見込み": lg(D["hv"]), "馬ごとのずれの合計": D["hsum"], "その組だけのずれ": D["hres"], "頭数": np.log(D["n"]), "レース番号": D["rno"] / 12.0,
      "人気の方の単勝": np.log(D["qmax"]), "人気薄の方の単勝": np.log(D["qmin"])}
names = list(FE); X = np.column_stack([FE[k] for k in names])
y = D["hit"].astype(float); Y = D["y"]; pay = D["pay"].astype(float)

import keiba_final as F
import lightgbm as lgb
P = {"logit": np.zeros(N), "lgb": np.zeros(N)}; W = np.zeros(N); coefs = []
for t in YS:
    tr, te = Y != t, Y == t
    # 払戻の見込み: 当たった組の (払戻/100 − 下限) ÷ (上限 − 下限) の平均 α
    h = tr & (y > 0) & (D["hi"] > D["lo"])
    alpha = float(np.clip(np.mean((pay[h] / 100 - D["lo"][h]) / (D["hi"][h] - D["lo"][h])), 0, 1))
    W[te] = D["lo"][te] + alpha * (D["hi"][te] - D["lo"][te])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    idx = np.where(tr)[0]; sub = idx[::3]          # ロジスティック回帰は 1/3 の行で(十分多い)
    w = np.array(F.fit_logit(((X[sub] - mu) / sd).tolist(), y[sub].tolist(), 0.01))
    P["logit"][te] = 1 / (1 + np.exp(-(np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ w)))
    coefs.append((t, alpha, w[0] - (w[1:] * mu / sd).sum(), w[1:] / sd))
    va = tr & (Y == max(yy for yy in YS if yy != t)); tr2 = tr & ~va
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500, "feature_fraction": 0.8,
              "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 1}
    b = lgb.train(params, lgb.Dataset(X[tr2], y[tr2]), 1500, valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(80, verbose=False)])
    P["lgb"][te] = b.predict(X[te], num_iteration=b.best_iteration)
    ll = lambda p: float(np.mean(y[te] * np.log(np.clip(p, 1e-6, 1)) + (1 - y[te]) * np.log(np.clip(1 - p, 1e-6, 1))))
    print("{}年を確かめる: 払戻の見込み α {:.2f} / 対数尤度 3連複だけ {:.5f}・回帰 {:.5f}・LightGBM {:.5f}(回数 {})  ({:.0f}秒)".format(
        t, alpha, ll(D["pt"][te]), ll(P["logit"][te]), ll(P["lgb"][te]), b.best_iteration, time.time() - t0), flush=True)
print("\n回帰の係数(2026年を確かめるときのもの): 切片 {:.3f}".format(coefs[-1][2]))
for k, c in zip(names, coefs[-1][3]):
    print("  {:14s} {:+.3f}".format(k, c))

EV = {"今の割安度×0.775": 0.775 * np.exp(D["lr"]), "3連複の見込み×払戻の見込み": D["pt"] * W, "回帰": P["logit"] * W, "LightGBM": P["lgb"] * W}
np.save(os.path.join(OUT, "ev_logit.npy"), EV["回帰"]); np.save(os.path.join(OUT, "ev_lgb.npy"), EV["LightGBM"])


def line(name, m):
    print("  {:30s} {:7d}点 {:5.1f}% [{}] 的中 {:4.1%}".format(name, int(m.sum()), pay[m].mean() if m.any() else 0, " ".join(
        "{:4.0f}".format(pay[m & (Y == t)].mean()) if (m & (Y == t)).any() else "   -" for t in YS), y[m].mean() if m.any() else 0))


for nm, ev in EV.items():
    print("\n■ {}: 期待値の帯ごとの回収率(確かめる年) [{}]".format(nm, " ".join(YS)))
    for a, b in ((0, .8), (.8, .9), (.9, 1.0), (1.0, 1.05), (1.05, 1.1), (1.1, 1.2), (1.2, 9)):
        line("期待値 {:.2f}〜{:.2f}".format(a, b), (ev >= a) & (ev < b))
print("\n■ 1レースで期待値がいちばん高い組だけ買う(ワイドの中央 1.5〜15倍の中で。確かめる年)")
band = (mid >= 1.5) & (mid < 15)
order = np.argsort(D["race"], kind="stable")
for nm, ev in EV.items():
    best = np.zeros(N, dtype=bool)
    e = np.where(band, ev, -1)
    # レースごとの最大
    last = {}
    for i in range(N):
        r_ = D["race"][i]
        if e[i] > 0 and (r_ not in last or e[i] > e[last[r_]]):
            last[r_] = i
    best[list(last.values())] = True
    print("  " + nm)
    for th in (0.95, 1.0, 1.05, 1.1, 1.15, 1.2):
        line("    期待値 {:.2f} 以上".format(th), best & (ev >= th))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

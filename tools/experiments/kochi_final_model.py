"""高知の最終レース(ファイナルレース)だけを別に解析して、最終レース用の確率を作る(keiba/kochi。2024年〜)
  KEIBA_OUT=keiba/kochi python tools/experiments/kochi_final_model.py
  1) 荒れ方: 最終レースとほかのレースで、3着内に来た馬の人気・◎の成績・3連複の配当を比べる
  2) 最終レースで3着内に来る馬の特徴(今のモデルの確率・人気・前走など)を、ロジスティック回帰で比べる
     年ごとに分け(2024・2025・2026)、ほかの2年で学んだ式で、その年の当たり方(対数尤度・AUC)を出す"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("KEIBA_OUT", "keiba/kochi")
import keiba as K
YS = ("2024", "2025", "2026")


def pl_top3(w):
    S = w.sum(); p1 = w / S; A = p1[:, None] * w[None, :] / (S - w)[:, None]; np.fill_diagonal(A, 0)
    B = S - w[:, None] - w[None, :]; C = np.where(B > 1e-12, A / np.maximum(B, 1e-12), 0); np.fill_diagonal(C, 0)
    return p1 + A.sum(0) + w * (C.sum() - C.sum(1) - C.sum(0))


rows, races = [], []
for f in sorted(glob.glob(K.OUT + "/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    day = K.load_day(d)
    rs = [r for r in day["races"].values() if r.get("venue") == "高知" and r.get("rno")]
    if not rs:
        continue
    last = max(r["rno"] for r in rs)
    for r in rs:
        p, res = r.get("pred") or {}, r.get("result") or {}
        od = {int(u): v for u, v in (res.get("odds") or {}).items() if v and v > 1}
        if not p.get("model") or len(res.get("order") or []) < 3 or len(od) < 5:
            continue
        hs = {h["umaban"]: h for h in p["horses"] if h.get("p3") is not None and h["umaban"] in od}
        card = {h["umaban"]: h for h in r.get("horses") or []}
        us = sorted(hs)
        inv = sum(1 / od[u] for u in us); q = np.array([1 / od[u] / inv for u in us])
        mt3 = pl_top3(q)                                    # 単勝オッズから出した3着内の確率(市場)
        top3 = set(res["order"][:3]); nk = {int(u): v for u, v in (res.get("ninki") or {}).items()}
        final = r["rno"] == last
        tri = (res.get("payout") or {}).get("3連複") or {}
        races.append(dict(y=d[:4], final=final, n=len(us), top_nk=sorted(nk.get(u, 99) for u in top3), tri=max(tri.values()) if tri else 0,
                          hon=p.get("honmei"), hon_in=p.get("honmei") in top3, fav_in=any(nk.get(u) == 1 for u in top3)))
        for i, u in enumerate(us):
            h, c = hs[u], card.get(u, {})
            ps = c.get("past") or []
            l0 = ps[0] if ps else {}
            def rel(x, n):
                return (x - 1) / max(n - 1, 1) if x and n else 0.5
            rows.append(dict(y=d[:4], final=final, rid=r["id"], u=u, top=int(u in top3), p3=h["p3"], q=q[i], mt3=mt3[i], n=len(us),
                             nk=nk.get(u, 99), mark=h.get("mark") or "",
                             last_rank=rel(l0.get("rank"), l0.get("n")), last_nk=rel(l0.get("ninki"), l0.get("n")),
                             last_margin=min(max(l0.get("margin") or 0, 0), 5) if l0 else 2.0,
                             cls_diff=(l0.get("cls") or 0) - (r.get("cls") or 0) if l0 else 0,
                             best3=min([rel(x.get("rank"), x.get("n")) for x in ps[:3]] or [0.5]),
                             pos=rel((l0.get("pass") or [0])[0], l0.get("n")) if l0.get("pass") else 0.5,
                             rest=min((h.get("rest_days") or 60), 365) / 30, nruns=min(h.get("nruns") or 0, 10),
                             wdiff=abs(c.get("wdiff") or 0), dist_diff=abs((r.get("dist") or 0) - (l0.get("dist") or 0)) / 200 if l0 else 0,
                             ability=(h.get("ability") or 0) / 10, score=(h.get("score") or 0) / 10, waku=rel(u, len(us))))
fin_r = [r for r in races if r["final"]]; oth_r = [r for r in races if not r["final"]]
print("高知 2024〜: 最終レース {}R / ほか {}R".format(len(fin_r), len(oth_r)))

print("\n■ 1. 荒れ方(最終レース / ほかのレース)")
for lab, xs in (("最終レース", fin_r), ("ほかのレース", oth_r)):
    print("  {:8s} 1番人気が3着内 {:4.0%} / ◎が3着内 {:4.0%} / 3着内に7番人気以下がいる {:4.0%} / 3着内の人気の平均 {:.1f} / 3連複の配当の中央 {:,.0f}円 / 頭数の平均 {:.1f}".format(
        lab, sum(r["fav_in"] for r in xs) / len(xs), sum(r["hon_in"] for r in xs) / len(xs),
        sum(1 for r in xs if max(r["top_nk"]) >= 7) / len(xs), np.mean([np.mean(r["top_nk"]) for r in xs]),
        np.median([r["tri"] for r in xs]), np.mean([r["n"] for r in xs])))
fr = [x for x in rows if x["final"]]; orr = [x for x in rows if not x["final"]]
print("  人気ごとの3着内率   " + " / ".join("{}番人気 最終 {:3.0%} ほか {:3.0%}".format(k, np.mean([x["top"] for x in fr if x["nk"] == k]), np.mean([x["top"] for x in orr if x["nk"] == k])) for k in (1, 2, 3, 5)))
print("  見込みとの差        最終: モデル {:+.0%} 市場 {:+.0%} / ほか: モデル {:+.0%} 市場 {:+.0%}(◎○▲の3着内 実際 − 見込み)".format(
    np.mean([x["top"] - x["p3"] for x in fr if x["mark"] in "◎○▲" and x["mark"]]), np.mean([x["top"] - x["mt3"] for x in fr if x["mark"] in "◎○▲" and x["mark"]]),
    np.mean([x["top"] - x["p3"] for x in orr if x["mark"] in "◎○▲" and x["mark"]]), np.mean([x["top"] - x["mt3"] for x in orr if x["mark"] in "◎○▲" and x["mark"]])))

lg = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))


def auc(s, y):
    o = np.argsort(s, kind="mergesort"); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1); pos = y.sum()
    return (rk[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def fit(X, y, l2=1.0):
    Z = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Z.shape[1]); R = l2 * np.eye(len(w)); R[0, 0] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-Z @ w)); H = (Z * (p * (1 - p))[:, None]).T @ Z + R + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, Z.T @ (y - p) - R @ w)
    return w


def norm3(p, rid):
    """レースで合計3にそろえる"""
    out = p.copy()
    for r in set(rid):
        m = rid == r; out[m] = np.minimum(.97, p[m] * 3 / p[m].sum())
    return out


FEATS = {
    "今のモデルの確率": ["lp3"],
    "今のモデル(合わせ直し)": ["lp3"],
    "市場(単勝オッズ)": ["lmt3"],
    "モデル+市場": ["lp3", "lmt3"],
    "モデル+市場+前走など": ["lp3", "lmt3", "last_rank", "last_nk", "last_margin", "best3", "pos", "cls_diff", "rest", "nruns", "wdiff", "dist_diff", "waku"],
    "市場+前走など(モデル無し)": ["lmt3", "last_rank", "last_nk", "last_margin", "best3", "pos", "cls_diff", "rest", "nruns", "wdiff", "dist_diff", "waku"],
}


def X_of(xs, fs):
    cols = []
    for f in fs:
        v = np.array([lg(x["p3"]) if f == "lp3" else lg(x["mt3"]) if f == "lmt3" else x[f] for x in xs], float)
        cols.append(v)
    return np.column_stack(cols)


print("\n■ 2. 最終レースの当たり方(年ごとに、ほかの2年の最終レースで学んだ式。3年分の合計)")
y = np.array([x["top"] for x in fr]); yrs = np.array([x["y"] for x in fr]); rid = np.array([x["rid"] for x in fr])
best = {}
for name, fs in FEATS.items():
    if name == "今のモデルの確率":
        p = np.array([x["p3"] for x in fr])
    else:
        X = X_of(fr, fs); p = np.zeros(len(fr))
        mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
        for yy in YS:
            w = fit(Xs[yrs != yy], y[yrs != yy], 2.0); p[yrs == yy] = 1 / (1 + np.exp(-(w[0] + Xs[yrs == yy] @ w[1:])))
        p = norm3(p, rid)
    ll = np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    lly = " ".join("{:.4f}".format(np.mean(y[yrs == yy] * np.log(np.clip(p[yrs == yy], 1e-4, 1)) + (1 - y[yrs == yy]) * np.log(np.clip(1 - p[yrs == yy], 1e-4, 1)))) for yy in YS)
    print("  {:26s} 対数尤度 {:.4f} [{}] AUC {:.4f}".format(name, ll, lly, auc(p, y)))
    best[name] = p
# 全部の最終レースで学んだ式(係数を見る)
X = X_of(fr, FEATS["モデル+市場+前走など"]); mu, sd = X.mean(0), X.std(0) + 1e-9
w = fit((X - mu) / sd, y, 2.0)
print("  係数(標準化。大きいほど3着内に来やすい): " + " / ".join("{} {:+.2f}".format(f, v) for f, v in zip(FEATS["モデル+市場+前走など"], w[1:])))
import pickle
pickle.dump(dict(rows=rows, best=best, fr_idx=[i for i, x in enumerate(rows) if x["final"]]), open(os.environ.get("KEIBA_TMP", "/tmp") + "/kochi_final_model.pkl", "wb"))

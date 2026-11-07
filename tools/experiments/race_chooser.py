"""レースごとに買い目を選ぶ(中央、2024年1月〜。予想はアプリと同じ)
  python tools/experiments/race_chooser.py
  券種ごとに「当たる確率」(分類)と「当たったときの払戻」(log の回帰)を LightGBM で学び、期待値 = 確率 × 払戻 がいちばん大きい
  買い目を選ぶ(期待値が k 未満なら見送り)。特徴は ◎○▲△ の確率・差・単勝オッズ・人気、頭数・クラス・芝ダ・距離・脚質・▲の延長など。
  2024・2025年で学び、2026年で確かめる(単勝オッズは確定。買うときのオッズとは少しずれる)"""
import glob, json, math, os, sys
import numpy as np
import lightgbm as lgb
sys.argv = sys.argv[:1]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trio_third.py")
exec(open(__file__).read().split("# ---------------------------------------------------------------- 1.")[0])
YS = ("2024", "2025", "2026")
key = lambda typ, c: "-".join(str(x) for x in (c if typ in ("馬単", "3連単") else sorted(c)))
pay = lambda r, typ, c: (r["pay"].get(typ) or {}).get(key(typ, c), 0)
BETS = {
    "単勝◎": lambda r: [("単勝", (r["h"],))],
    "ワイド◎-○": lambda r: [("ワイド", (r["h"], r["t"]))],
    "ワイド◎-▲": lambda r: [("ワイド", (r["h"], r["a"]))],
    "馬連◎-○": lambda r: [("馬連", (r["h"], r["t"]))],
    "馬単◎→○": lambda r: [("馬単", (r["h"], r["t"]))],
    "3連複◎○▲": lambda r: [("3連複", (r["h"], r["t"], r["a"]))],
    "3連複◎○-▲△": lambda r: [("3連複", (r["h"], r["t"], x)) for x in (r["a"], r["b"])],
    "3連複◎-○▲△": lambda r: [("3連複", (r["h"], x, y)) for x, y in ((r["t"], r["a"]), (r["t"], r["b"]), (r["a"], r["b"]))],
    "3連単◎→○→▲": lambda r: [("3連単", (r["h"], r["t"], r["a"]))],
}
ST = {"逃げ": 0, "先行": 1, "差し": 2, "追込": 3}
for r in races:
    hs, od = r["hs"], r["od"]
    inv = sum(1 / v for v in od.values() if v > 1)
    q = lambda u: (1 / od[u] / inv) if od.get(u) else 0
    r["x"] = [r["p1"], hs[r["t"]]["p3"], hs[r["a"]]["p3"], hs[r["b"]]["p3"], r["p1"] - hs[r["t"]]["p3"], hs[r["t"]]["p3"] - hs[r["a"]]["p3"],
              q(r["h"]), q(r["t"]), q(r["a"]), q(r["b"]), r["nk"].get(r["h"], 9), r["nk"].get(r["t"], 9), r["nk"].get(r["a"], 9),
              r["n"], r["cls"] or 0, 1 if r["surf"] == "芝" else 0, r["dist"],
              ST.get(hs[r["h"]].get("style"), 2), ST.get(hs[r["t"]].get("style"), 2), ST.get(hs[r["a"]].get("style"), 2),
              1 if "距離延長" in (hs[r["a"]].get("axis_ng") or []) else 0, 1 if hs[r["a"]].get("cut") else 0,
              sum(sorted([r["nk"].get(u, 9) for u in (r["h"], r["t"], r["a"])]))]
    r["ret"] = {b: (sum(pay(r, t, c) for t, c in f(r)), 100 * len(f(r))) for b, f in BETS.items()}
tr = [r for r in races if r["y"] != "2026"]; te = [r for r in races if r["y"] == "2026"]
Xtr = np.array([r["x"] for r in tr]); Xte = np.array([r["x"] for r in te])
P = dict(objective="binary", learning_rate=.03, num_leaves=15, min_data_in_leaf=80, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=1)
PR = dict(P, objective="regression")
ev_te, ev_tr = {}, {}
print("券種ごとのモデル(2024・2025年で学ぶ)")
for b in BETS:
    y = np.array([1 if r["ret"][b][0] else 0 for r in tr])
    # 学ぶ期間の中を時系列で8:2に分けて木の数を決める
    cut = int(len(tr) * .8)
    m = lgb.train(P, lgb.Dataset(Xtr[:cut], y[:cut]), 2000, valid_sets=[lgb.Dataset(Xtr[cut:], y[cut:])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    clf = lgb.train(P, lgb.Dataset(Xtr, y), max(50, int(m.best_iteration * 1.1)))
    hit = y == 1
    yp = np.log(np.array([r["ret"][b][0] / r["ret"][b][1] for r in tr])[hit])     # 当たったときの払戻(賭け金あたり)
    m2 = lgb.train(PR, lgb.Dataset(Xtr[hit][:int(hit.sum() * .8)], yp[:int(hit.sum() * .8)]), 2000,
                   valid_sets=[lgb.Dataset(Xtr[hit][int(hit.sum() * .8):], yp[int(hit.sum() * .8):])], callbacks=[lgb.early_stopping(100, verbose=False)])
    reg = lgb.train(PR, lgb.Dataset(Xtr[hit], yp), max(50, int(m2.best_iteration * 1.1)))
    resid = np.var(yp - reg.predict(Xtr[hit]))
    f = lambda X: clf.predict(X) * np.exp(reg.predict(X) + resid / 2)      # 期待値(賭け金あたり)
    ev_te[b] = f(Xte); ev_tr[b] = f(Xtr)
    print("  {:12s} 当たる率 {:.1%} / 2026年の期待値の平均 {:.2f}".format(b, y.mean(), ev_te[b].mean()))


def simulate(rows, evs, k, allowed=None):
    st = rt = n = 0; by = {}; chosen = {}
    for i, r in enumerate(rows):
        bs = [b for b in BETS if allowed is None or b in allowed]
        b = max(bs, key=lambda b: evs[b][i])
        if evs[b][i] < k:
            continue
        g, s = r["ret"][b]; rt += g; st += s; n += 1; chosen[b] = chosen.get(b, 0) + 1
        y = r["y"]; by.setdefault(y, [0, 0]); by[y][0] += g; by[y][1] += s
    return n, (rt / st * 100 if st else 0), rt - st, by, chosen


print("\n■ 期待値がいちばん大きい買い目を買う(期待値 k 以上のレースだけ)。学んだ期間(参考)/ 2026年")
for k in (0.9, 1.0, 1.1, 1.2, 1.3, 1.5):
    n1, r1, p1_, _, _ = simulate(tr, ev_tr, k)
    n2, r2, p2_, _, ch = simulate(te, ev_te, k)
    print("  期待値{:.1f}以上: 2024-25 {:4d}R {:4.0f}%(学んだ期間) / 2026 {:4d}R {:4.0f}% 収支{:+,}円  選んだ券種 {}".format(
        k, n1, r1, n2, r2, int(p2_), dict(sorted(ch.items(), key=lambda x: -x[1])[:4])))
# 今の買い方(2026年)
cur = [r for r in te if r["p1"] >= .8 and r["p1"] - r["hs"][r["t"]]["p3"] >= .15]
g = sum(pay(r, "ワイド", (r["h"], r["t"])) if "距離延長" in (r["hs"][r["a"]].get("axis_ng") or []) else pay(r, "3連複", (r["h"], r["t"], r["a"])) for r in cur)
print("  (今の買い方 2026年: {}R {:.0f}% 収支{:+,}円)".format(len(cur), g / len(cur), g - 100 * len(cur)))
# 今の買うレースだけで、券種をレースごとに選ぶ
idx = [i for i, r in enumerate(te) if r["p1"] >= .8 and r["p1"] - r["hs"][r["t"]]["p3"] >= .15]
sub = [te[i] for i in idx]; evs = {b: ev_te[b][idx] for b in BETS}
n, roi, pl, by, ch = simulate(sub, evs, 0)
print("  今の買うレースで、券種だけレースごとに選ぶ 2026: {}R {:.0f}% 収支{:+,}円 選んだ券種 {}".format(n, roi, int(pl), ch))

"""前日のデータから「推定オッズ」を出し、市場のずれを補正して、期待値の高い馬の単勝を買うと儲かるか(中央 2022〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/est_odds.py
  (X のポスト「前日の期待値 1.1以上を単勝で買い続けたら 2019〜2026年 109%」を、手元のデータでなぞる)
  1. 推定オッズ … 前日に分かる材料(過去走・騎手・厩舎・評価・枠など。オッズや人気は使わない。過去走の人気は使う)から、
                  その馬の単勝の売れ方(確定オッズから出した割合)を当てる LightGBM。その年を除いて学ぶ
  2. 1着の確率 … 同じ材料で「1着か」を当てる LightGBM(その年を除いて学ぶ)
  3. 補正 … レースの中で、確率 = softmax(a × log(推定の売れ方) + b × log(1着の確率))。a・b もその年を除いて決める
             (= 推定オッズの見込みに対して、実際の勝ち方が違うところを直す)
  4. 期待値 … 補正した確率 × 推定オッズ(推定の売れ方から、控除 20% を引いたオッズ)。前日に決まる。
             くらべるため、補正した確率 × 確定オッズ(発走の直前に買う形。手元では確定オッズしか無いので甘めに出る)も出す
  払戻は確定オッズ(買った時刻によらず確定オッズで払われる)。ポストに合わせて 100倍で頭打ちにした数字も出す"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import lightgbm as lgb
YS = ("2022", "2023", "2024", "2025", "2026")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
t0 = time.time()
races = T.load_races()
info = {rc["id"]: rc["result"] for d, rc in races}
del races
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
rows = [r for r in rows if r["f"].get("center") == 0 and r["rid"] in info]
# レースごとにまとめる(確定オッズのある馬だけ。5頭以上)
by = defaultdict(list)
for r in rows:
    by[r["rid"]].append(r)
R, keep = [], []
for rid in sorted(by, key=lambda x: (by[x][0]["date"], x)):
    res = info[rid]; order = res.get("order") or []
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    rs = [r for r in by[rid] if r["u"] in od]
    if len(order) < 3 or len(rs) < 5 or order[0] not in {r["u"] for r in rs}:
        continue
    R.append((rid, rs[0]["date"], len(keep), len(rs), od, order))
    keep += rs
rows = keep
feats = sorted({k for r in rows for k in r["f"]})
X = np.array([[M._num(r["f"].get(k)) for k in feats] for r in rows], dtype=np.float32)
sizes = np.array([x[3] for x in R]); st = np.array([x[2] for x in R])
race_of = np.repeat(np.arange(len(R)), sizes)
yr = np.array([x[1][:4] for x in R])[race_of]
odds = np.array([R[g][4][rows[i]["u"]] for i, g in enumerate(race_of)])
win = np.array([int(rows[i]["u"] == R[g][5][0]) for i, g in enumerate(race_of)])
inv = 1 / odds
share = inv / np.repeat(np.add.reduceat(inv, st), sizes)              # 確定オッズから出した売れ方(合計 1)
tgt = np.log(share * np.repeat(sizes, sizes))                         # 頭数でそろえた売れ方(log)
cat = [feats.index(k) for k in ("venue", "style") if k in feats]
print("中央 {}R・{}頭・材料 {}個 ({:.0f}分)".format(len(R), len(rows), len(feats), (time.time() - t0) / 60), flush=True)


def softmax_race(v):
    v = v - np.repeat(np.maximum.reduceat(v, st), sizes)
    e = np.exp(v)
    return e / np.repeat(np.add.reduceat(e, st), sizes)


P = dict(reg={"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 80, "feature_fraction": 0.7,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1},
         bin={"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.7,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1})
est = np.zeros(len(rows)); pw = np.zeros(len(rows))
for Y in YS:
    te = yr == Y; tr = ~te
    # 学ぶ年の最後の 1割(日付の順)で回数を決める
    trd = np.where(tr)[0]; cut = trd[int(len(trd) * 0.9)]
    a = tr & (np.arange(len(rows)) < cut); b = tr & (np.arange(len(rows)) >= cut)
    for key, yy, out in (("reg", tgt, est), ("bin", win, pw)):
        ps = []
        for sd in SEEDS:
            prm = dict(P[key], seed=sd)
            m = lgb.train(prm, lgb.Dataset(X[a], yy[a], categorical_feature=cat), 3000,
                          valid_sets=[lgb.Dataset(X[b], yy[b], categorical_feature=cat)], callbacks=[lgb.early_stopping(100, verbose=False)])
            m2 = lgb.train(prm, lgb.Dataset(X[tr], yy[tr], categorical_feature=cat), int(m.best_iteration * 1.1))
            ps.append(m2.predict(X[te]))
        out[te] = np.mean(ps, axis=0)
    print("  {}年を除いて学んだ ({:.0f}分)".format(Y, (time.time() - t0) / 60), flush=True)
est_share = softmax_race(est)                                          # 推定の売れ方(合計 1)
pwn = pw / np.repeat(np.add.reduceat(pw, st), sizes)                   # 1着の確率(合計 1)
lg = lambda v: np.log(np.clip(v, 1e-6, 1))


def cl_fit(F, y, idx):
    """条件つきロジット(レースの中で1頭が勝つ)。idx = 使うレースの番号"""
    m = np.isin(race_of, idx); Fm, ym = F[m], y[m]
    sz = sizes[idx]; s0 = np.concatenate([[0], np.cumsum(sz)[:-1]])
    b = np.array([1.0] + [0.0] * (F.shape[1] - 1))
    for _ in range(50):
        s = Fm @ b; s = s - np.repeat(np.maximum.reduceat(s, s0), sz)
        e = np.exp(s); p = e / np.repeat(np.add.reduceat(e, s0), sz)
        Fp = Fm * p[:, None]; Mx = np.add.reduceat(Fp, s0, axis=0)
        step = np.linalg.solve(Fp.T @ Fm - Mx.T @ Mx + 1e-8 * np.eye(len(b)), Fm.T @ (ym - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


ryear = np.array([x[1][:4] for x in R])
F_est = np.column_stack([lg(est_share), lg(pwn)])
F_fin = np.column_stack([lg(share), lg(pwn)])
pc = np.zeros(len(rows)); pf = np.zeros(len(rows)); coefs = {}
for Y in YS:
    tr_idx = np.where(ryear != Y)[0]
    b1 = cl_fit(F_est, win, tr_idx); b2 = cl_fit(F_fin, win, tr_idx); coefs[Y] = (b1, b2)
    m = yr == Y
    pc[m] = softmax_race(F_est @ b1)[m]; pf[m] = softmax_race(F_fin @ b2)[m]
print("補正の係数(推定の売れ方・1着の確率): " + " / ".join("{} {:.2f}・{:.2f}".format(Y, *coefs[Y][0]) for Y in YS))
print("補正の係数(確定の売れ方・1着の確率): " + " / ".join("{} {:.2f}・{:.2f}".format(Y, *coefs[Y][1]) for Y in YS))
est_odds = 0.8 / est_share
ev_est = pc * est_odds; ev_fin = pf * odds
ll = lambda p: float(np.mean(np.log(np.clip(p[win == 1], 1e-9, 1))))
print("\n■ 当たり方(勝ち馬の log 確率の平均。大きいほど良い): 確定オッズ {:.4f} / 推定オッズ {:.4f} / 1着の確率 {:.4f} / 補正(推定+1着) {:.4f} / 補正(確定+1着) {:.4f}".format(
    ll(share), ll(est_share), ll(pwn), ll(pc), ll(pf)))
print("  推定と確定の売れ方の相関 {:.3f}(log)/ 1番人気が当たる {:.1%}".format(np.corrcoef(lg(est_share), lg(share))[0, 1],
      np.mean([np.argmax(est_share[s:s + n]) == np.argmax(share[s:s + n]) for s, n in zip(st, sizes)])))
# 市場のずれ: 推定オッズの帯ごとの 実際の勝ち ÷ 推定の見込み、確定オッズの見込み
print("\n■ 推定オッズの帯ごと: 実際の1着 ÷ 推定の見込み / 実際 ÷ 確定オッズの見込み / 単勝の回収率")
for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 1e9)):
    m = (est_odds >= lo) & (est_odds < hi)
    print("  推定 {:>4.0f}〜{:<4.0f}倍 {:6d}頭 {:.2f} / {:.2f} / {:5.1f}%".format(lo, hi if hi < 1e9 else 999, m.sum(), win[m].sum() / (0.8 / est_odds[m]).sum(),
          win[m].sum() / share[m].sum(), 100 * np.mean(win[m] * odds[m])))


def show(name, m):
    n = int(m.sum())
    if n < 30:
        print("  {:46s} {:5d}頭(少ない)".format(name, n)); return
    pay = win[m] * odds[m]; cap = win[m] * np.minimum(odds[m], 100)
    ys = " ".join("{:3.0f}".format(100 * np.mean(cap[yr[m] == Y])) if (yr[m] == Y).any() else "  -" for Y in YS)
    sd = np.std(cap) / np.sqrt(n) * 196
    print("  {:46s} {:5d}頭 的中 {:5.1%} 平均 {:5.1f}倍 回収率 {:6.1f}%(100倍で頭打ち {:6.1f}% ±{:.0f}) [{}] 確定÷推定オッズ {:.2f}".format(
        name, n, win[m].mean(), np.median(odds[m]), 100 * pay.mean(), 100 * cap.mean(), sd, ys, np.median(odds[m] / est_odds[m])))


print("\n■ 前日に決まる期待値(補正した確率 × 推定オッズ)で単勝を買う  [ ] は 2022〜2026年(100倍で頭打ち)")
for th in (1.0, 1.1, 1.2, 1.3, 1.5):
    show("期待値 {:.1f}以上".format(th), ev_est >= th)
for th in (1.1, 1.3):
    for lo, hi in ((1, 10), (10, 30), (30, 1e9)):
        show("期待値 {:.1f}以上・推定 {:.0f}〜{:.0f}倍".format(th, lo, hi if hi < 1e9 else 999), (ev_est >= th) & (est_odds >= lo) & (est_odds < hi))
print("\n■ くらべる: 補正した確率 × 確定オッズ(発走の直前に買う形。甘めに出る)")
for th in (1.0, 1.1, 1.2, 1.3):
    show("期待値 {:.1f}以上".format(th), ev_fin >= th)
show("(物差し)全部の馬", np.ones(len(rows), bool))
show("(物差し)1番人気", np.concatenate([np.arange(n) == np.argmax(share[s:s + n]) for s, n in zip(st, sizes)]))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

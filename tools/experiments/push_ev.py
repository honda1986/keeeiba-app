"""後押しをもっとうまく使う: 後押しの式の確率(オッズ + 評価値)で期待値を出して買ったらどうなるか(中央 2022〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/push_ev.py
  後押しの式(tools/train_push.py)は「オッズの土台 + 評価値」で、今は評価値の分(後押し)しか使っていない。
  土台まで入れると、馬ごとの1着の確率・組ごとの2頭とも3着内の確率になる。その年を除いて作った式で:
   単勝 … 期待値 = 1着の確率 × 確定の単勝オッズ
   ワイド … 期待値 = 2頭とも3着内の確率 × ワイドの確定オッズの中ほど(下限 + 0.35 ×(上限 − 下限)。keiba_logic.PRE と同じ)
  くらべるため、評価値を入れない式(土台だけ = オッズのゆがみだけ)の期待値でも同じように買う。
  注意: 土台は確定オッズ(単勝・3連複・ワイド)から作るので、発走前に買うより甘く出る"""
import os, pickle, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import train_push as TP
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
D = TP.build(rows, races)
del rows, races
h, P = D["H"], D["P"]; Z = D["Z"]; sizes = D["sizes"]


def fit_win_full(B, Z_, y, sizes_, l2=TP.L2):
    """train_push.fit_win と同じで、土台の重みも返す"""
    F = np.column_stack([B, Z_]) if Z_ is not None else B; kb = B.shape[1]; k = F.shape[1]
    st = np.concatenate([[0], np.cumsum(sizes_)[:-1]])
    R = np.eye(k) * l2; R[:kb, :kb] = 0
    b = np.zeros(k); b[0] = 1.0
    for _ in range(50):
        s = F @ b; s -= np.repeat(np.maximum.reduceat(s, st), sizes_)
        e = np.exp(s); p = e / np.repeat(np.add.reduceat(e, st), sizes_)
        Fp = F * p[:, None]; M = np.add.reduceat(Fp, st, axis=0)
        step = np.linalg.solve(Fp.T @ F - M.T @ M + R + 1e-8 * np.eye(k), F.T @ (y - p) - R @ b)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def fit_pair_full(B, ZP, y, l2=TP.L2):
    F = np.column_stack([B, ZP]) if ZP is not None else B; kb = B.shape[1]; k = F.shape[1]
    R = np.eye(k) * l2; R[:kb, :kb] = 0
    b = np.zeros(k)
    for _ in range(40):
        p = 1 / (1 + np.exp(-(F @ b)))
        step = np.linalg.solve((F * (p * (1 - p))[:, None]).T @ F + R + 1e-8 * np.eye(k), F.T @ (y - p) - R @ b)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def softmax_race(s, sz):
    st = np.concatenate([[0], np.cumsum(sz)[:-1]])
    s = s - np.repeat(np.maximum.reduceat(s, st), sz); e = np.exp(s)
    return e / np.repeat(np.add.reduceat(e, st), sz)


ZP = Z[P["ia"]] + Z[P["ib"]]
pw = {k: np.zeros(len(h["u"])) for k in ("push", "base")}
pp = {k: np.zeros(len(P["y"])) for k in ("push", "base")}
for yy in YS:
    te = D["hyear"] == yy; tp_ = D["pyear"] == yy; rs = D["year"] == yy
    for k, Zk, ZPk in (("push", Z, ZP), ("base", None, None)):
        bw = fit_win_full(D["BW"][~te], None if Zk is None else Zk[~te], h["win"][~te], sizes[~rs])
        Fw = D["BW"][te] if Zk is None else np.column_stack([D["BW"][te], Zk[te]])
        pw[k][te] = softmax_race(Fw @ bw, sizes[rs])
        bp = fit_pair_full(D["BP"][~tp_], None if ZPk is None else ZPk[~tp_], P["y"][~tp_])
        Fp = D["BP"][tp_] if ZPk is None else np.column_stack([D["BP"][tp_], ZPk[tp_]])
        pp[k][tp_] = 1 / (1 + np.exp(-(Fp @ bp)))
    print("  {}年を除いて作った ({:.0f}分)".format(yy, (time.time() - t0) / 60), flush=True)
del ZP

odds = h["odds"]; win = h["win"]; hy = D["hyear"]; py = D["pyear"]
# ワイドの中ほどのオッズ: P["mid"] は (下限 + 上限)/2。下限・上限は wide_final から取り直す
lo_hi = []
k0 = 0; pk = 0
for g, rid in enumerate(D["rid"]):
    wf = D["info"][rid].get("wide_final") or {}; tp = D["info"][rid].get("trio_pair_final") or {}
    for key, v in wf.items():
        s = tp.get(key)
        if not s or not v or v[0] <= 1:
            continue
        a, b = (int(t) for t in key.split("-"))
        us = set(h["u"][k0:k0 + sizes[g]].astype(int))
        if a not in us or b not in us:
            continue
        lo_hi.append(v)
    k0 += sizes[g]
lo_hi = np.array(lo_hi, float)
assert len(lo_hi) == len(P["y"]), (len(lo_hi), len(P["y"]))
wmid = lo_hi[:, 0] + 0.35 * (lo_hi[:, 1] - lo_hi[:, 0])
print("中央 {}頭・ワイドの組 {}点 ({:.0f}分)".format(len(odds), len(wmid), (time.time() - t0) / 60), flush=True)


def show(name, m, ret, yrs):
    n = int(m.sum())
    if n < 50:
        print("  {:40s} {:6d}点(少ない)".format(name, n)); return
    r = ret[m]; y = yrs[m]; top = np.sort(r)[-3:].sum()
    print("  {:40s} {:6d}点 的中 {:5.1%} 回収率 {:6.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
        name, n, np.mean(r > 0), 100 * r.mean(), 100 * (r.sum() - top) / n,
        " ".join("{:3.0f}".format(100 * r[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))


wret = win * odds
print("\n■ 単勝: 期待値 = 1着の確率 × 確定オッズ  [ ] は 2022〜2026年(確率はその年を除いて作った式)")
for k, nm in (("push", "後押しの式(オッズ+評価値)"), ("base", "土台だけ(オッズのゆがみ)")):
    ev = pw[k] * odds
    for th in (1.0, 1.1, 1.2, 1.3):
        for lo_o, hi_o in ((1, 10), (10, 30)):
            show("{} 期待値{:.1f}以上・{:.0f}〜{:.0f}倍".format(nm, th, lo_o, hi_o), (ev >= th) & (odds >= lo_o) & (odds < hi_o), wret, hy)
show("(物差し)単勝 全部 30倍未満", odds < 30, wret, hy)
pret = P["pay"]
print("\n■ ワイド: 期待値 = 2頭とも3着内の確率 × オッズの中ほど")
for k, nm in (("push", "後押しの式"), ("base", "土台だけ")):
    ev = pp[k] * wmid
    for th in (1.0, 1.1, 1.2, 1.3):
        show("{} 期待値{:.1f}以上".format(nm, th), ev >= th, pret, py)
        show("{} 期待値{:.1f}以上・確率30%以上".format(nm, th), (ev >= th) & (pp[k] >= 0.30), pret, py)
show("(物差し)ワイド 全部", np.ones(len(pret), bool), pret, py)
# 後押しの分だけで、土台の期待値が同じくらいの中で差が出るか(土台の期待値 0.9〜1.1 の帯で、後押しの式との差で分ける)
print("\n■ 土台の期待値が 0.85〜1.05 の組・馬を、後押しで上がった分(後押しの式 ÷ 土台)で分ける")
evb, evp = pp["base"] * wmid, pp["push"] * wmid
mb = (evb >= 0.85) & (evb < 1.05)
for lo_r, hi_r in ((0, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 9)):
    r = evp / np.maximum(evb, 1e-9)
    show("ワイド 後押しで ×{:.1f}〜{}".format(lo_r, "{:.1f}".format(hi_r) if hi_r < 9 else ""), mb & (r >= lo_r) & (r < hi_r), pret, py)
evb, evp = pw["base"] * odds, pw["push"] * odds
mb = (evb >= 0.85) & (evb < 1.05) & (odds < 30)
for lo_r, hi_r in ((0, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2), (1.2, 9)):
    r = evp / np.maximum(evb, 1e-9)
    show("単勝 後押しで ×{:.1f}〜{}".format(lo_r, "{:.1f}".format(hi_r) if hi_r < 9 else ""), mb & (r >= lo_r) & (r < hi_r), wret, hy)

# 境目を動かしても続くか(1つのマスだけ良いのではないか)。後押しの式のワイド: 期待値 × 確率 の表(回収率と、100%を超えた年の数)
print("\n■ ワイド(後押しの式): 期待値の境目 × 確率の境目  回収率%(点数・100%を超えた年の数/5)")
evp = pp["push"] * wmid
print("  {:>10s}".format("") + "".join("  確率{:>3.0f}%以上      ".format(100 * pt_) for pt_ in (0.2, 0.25, 0.3, 0.35, 0.4)))
for th in (1.0, 1.05, 1.1, 1.15, 1.2):
    cells = []
    for pt_ in (0.2, 0.25, 0.3, 0.35, 0.4):
        m = (evp >= th) & (pp["push"] >= pt_)
        r = pret[m]; y = py[m]
        good = sum(1 for yy in YS if (y == yy).any() and r[y == yy].mean() >= 1)
        cells.append("{:5.1f}%({:4d}・{}/5)".format(100 * r.mean(), int(m.sum()), good) if m.sum() >= 30 else "{:>17s}".format("-"))
    print("  期待値{:.2f}以上 ".format(th) + " ".join(cells))
# 今の「期待値ワイド」(確率 = 0.75 × 3連複から見た組の強さ。keiba_logic.ev_wide_picks。上位6組 + ◎-○ は略して上位6組)と同じ物差しで
print("\n■ くらべる: 今の期待値ワイドの確率(0.75 × 3連複の組の強さ)と、後押しの式の確率(どちらも確定オッズで)")
rank_in_race = np.zeros(len(pret), int)
st_p = np.concatenate([[0], np.cumsum(D["psz"])[:-1]])
for s0, n in zip(st_p, D["psz"]):
    if n:
        o = np.argsort(-P["pt"][s0:s0 + n]); rk = np.empty(n, int); rk[o] = np.arange(n); rank_in_race[s0:s0 + n] = rk
cur = P["pt"] * wmid
show("今の形: 期待値0.95以上・確率の上位6組", (cur >= 0.95) & (rank_in_race < 6), pret, py)
show("今の形: 期待値1.0以上・確率30%以上", (cur >= 1.0) & (P["pt"] >= 0.3), pret, py)
show("今の形: 期待値1.1以上・確率30%以上", (cur >= 1.1) & (P["pt"] >= 0.3), pret, py)
show("後押しの式: 期待値1.0以上・確率30%以上", (evp >= 1.0) & (pp["push"] >= 0.3), pret, py)
show("後押しの式: 期待値1.1以上・確率30%以上", (evp >= 1.1) & (pp["push"] >= 0.3), pret, py)
# 1日あたりの点数
days = np.repeat(np.array(D["date"]), D["psz"])
m = (evp >= 1.1) & (pp["push"] >= 0.3)
# 今の期待値ワイドと後押しの式を組み合わせる
curpick = (cur >= 0.95) & (rank_in_race < 6)
print("\n■ 今の期待値ワイドと後押しの式を組み合わせる")
show("今の形のうち、後押しの式でも期待値1.0以上", curpick & (evp >= 1.0), pret, py)
show("今の形のうち、後押しの式で期待値1.0未満", curpick & (evp < 1.0), pret, py)
show("後押しの式 期待値1.1以上・確率20%以上", (evp >= 1.1) & (pp["push"] >= 0.2), pret, py)
show("後押しの式 期待値1.1以上・確率20%以上 のうち今の形にない", (evp >= 1.1) & (pp["push"] >= 0.2) & ~curpick, pret, py)
show("どちらか(今の形 + 後押しの式 期待値1.1以上・確率20%以上)", curpick | ((evp >= 1.1) & (pp["push"] >= 0.2)), pret, py)
show("両方", curpick & (evp >= 1.1) & (pp["push"] >= 0.2), pret, py)
m2 = (evp >= 1.1) & (pp["push"] >= 0.2)
print("  後押しの式 期待値1.1以上・確率20%以上: 開催日 {}日に {}点(1日 {:.1f}点)".format(len(set(np.repeat(np.array(D["date"]), D["psz"]))), int(m2.sum()),
      m2.sum() / len(set(np.repeat(np.array(D["date"]), D["psz"])))))
print("  後押しの式 期待値1.1以上・確率30%以上: 開催日 {}日に {}点(1日 {:.1f}点・0点の日 {:.0%})".format(
    len(set(days)), int(m.sum()), m.sum() / len(set(days)), 1 - len(set(days[m])) / len(set(days))))
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

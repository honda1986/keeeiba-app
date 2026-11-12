"""割安ワイドは「良く見える基準を選んだだけ」ではないか(中央 2022〜2026年。確定オッズ)
  python tools/experiments/gap_select_check.py
  割安ワイドの基準(1レースで割安度がいちばん高い組・ワイド 1.5〜15倍・割安度 1.30 以上)は、5年分の結果を見ながら決めた。
  1) 基準を少しずつ変えた90通り(ワイドの下限・上限・割安度の下限)の回収率を全部出す(今の基準だけが良いのか、まわりも同じか)
  2) 確かめる年を除いた年でいちばん良い基準を選び、確かめる年に当てはめる(選んだ分の上振れを除いた成績)。
     その年より前の年だけで選ぶ形でも
  3) 今の基準の成績が、たまたまでどこまで動くか(レースを選び直す)。年の前半・後半
  4) 選ばない買い方: 理屈どおり「0.775 × 割安度 が 1 以上(= 割安度 1.29 以上)の組を全部買う」"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2022", "2023", "2024", "2025", "2026")
mid = lambda v: (v[0] + v[1]) / 2 if isinstance(v, (list, tuple)) else v
races = []      # (日付, [(割安度, ワイドのオッズ, 払戻)] 割安度の大きい順)
for f in sorted(glob.glob(R + "/keiba/days/20*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        tp, wf, res = r.get("trio_pair_final"), r.get("wide_final"), r.get("result") or {}
        if r.get("center") != "jra" or not tp or not wf or len(res.get("order") or []) < 3:
            continue
        pay = (res.get("payout") or {}).get("ワイド") or {}
        out = {h["umaban"] for h in r.get("horses") or [] if h.get("scratched")}
        cs = []
        for k, v in wf.items():
            w = mid(v)
            if not w or w < 1.5 or w >= 30 or k not in tp or out & {int(x) for x in k.split("-")}:
                continue
            cs.append((0.75 * tp[k] * w / 0.775, w, pay.get(k, 0)))
        cs.sort(reverse=True)
        races.append((d, cs))
print("中央 {}R".format(len(races)))


def run(lo, hi, th):
    """1レースで、ワイド lo〜hi 倍の中で割安度がいちばん高い組が th 以上なら買う -> [(日付, 払戻)]"""
    out = []
    for d, cs in races:
        for ratio, w, p in cs:
            if lo <= w < hi:
                if ratio >= th:
                    out.append((d, p))
                break
    return out


def roi(rows, y=None):
    v = [p for d, p in rows if y is None or d[:4] == y]
    return (float(np.mean(v)) if v else 0.0), len(v)


GRID = [(lo, hi, th) for lo in (1.5, 2.0, 3.0) for hi in (8, 10, 15, 20, 30) for th in (1.15, 1.20, 1.25, 1.30, 1.35, 1.40)]
RES = {g: run(*g) for g in GRID}
NOW = (1.5, 15, 1.30)

print("\n■ 1) 基準を変えた {} 通りの回収率(5年・確定オッズ)。行 = ワイドの範囲、列 = 割安度の下限。( ) はレース数".format(len(GRID)))
ths = (1.15, 1.20, 1.25, 1.30, 1.35, 1.40)
print("  {:14s}".format("") + "".join("{:>13.2f}".format(t) for t in ths))
for lo in (1.5, 2.0, 3.0):
    for hi in (8, 10, 15, 20, 30):
        print("  {:>4}〜{:<4}倍  ".format(lo, hi) + "".join("{:>6.0f}%({:>4})".format(*roi(RES[(lo, hi, t)])) for t in ths))
allr = [roi(RES[g])[0] for g in GRID]
print("  {} 通りの回収率: いちばん低い {:.0f}%・中央 {:.0f}%・いちばん高い {:.0f}%。100% を超えるのは {} 通り。今の基準は {:.0f}%".format(
    len(GRID), min(allr), np.median(allr), max(allr), sum(a > 100 for a in allr), roi(RES[NOW])[0]))

print("\n■ 2) 選ぶ年と確かめる年を分ける(選ぶ年で 1,000R 以上の基準の中から、回収率がいちばん高いものを選ぶ)")
for name, train_of in (("ほかの4年で選ぶ", lambda y: [t for t in YS if t != y]), ("その年より前の年だけで選ぶ", lambda y: [t for t in YS if t < y])):
    tot = []
    print("  " + name)
    for y in YS:
        tr = train_of(y)
        if not tr:
            continue
        def tr_roi(g):
            v = [p for d, p in RES[g] if d[:4] in tr]
            return (np.mean(v) if len(v) >= 1000 * len(tr) / 4 else -1)
        best = max(GRID, key=tr_roi)
        r_te, n_te = roi(RES[best], y)
        tot += [p for d, p in RES[best] if d[:4] == y]
        print("    {}年: 選んだ基準 ワイド{}〜{}倍・割安度{:.2f}以上(選ぶ年で {:.0f}%) → この年 {:.0f}%({}R) / 今の基準なら {:.0f}%".format(
            y, best[0], best[1], best[2], tr_roi(best), r_te, n_te, roi(RES[NOW], y)[0]))
    print("    確かめた年の合計: {:.0f}%({}R)".format(np.mean(tot), len(tot)))

print("\n■ 3) 今の基準(ワイド1.5〜15倍・割安度1.30以上)の成績のたまたまの幅")
v = np.array([p for d, p in RES[NOW]], dtype=float)
rng = np.random.default_rng(0)
bs = np.array([rng.choice(v, len(v)).mean() for _ in range(5000)])
print("  5年 {}R・{:.1f}%・的中 {:.1%}  95%の幅 {:.0f}〜{:.0f}%・100%を超えている見込み {:.0%}".format(len(v), v.mean(), (v > 0).mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 100).mean()))
top = np.sort(v)[::-1]
print("  払戻の大きい10レースを除くと {:.1f}%(いちばん大きい払戻 {:.0f}円)".format((v.sum() - top[:10].sum()) / (len(v) - 10), top[0]))
hy = defaultdict(list)
for d, p in RES[NOW]:
    hy[d[:4] + ("前" if d[4:6] <= "06" else "後")].append(p)
print("  半年ごと: " + " ".join("{} {:.0f}%".format(k, np.mean(x)) for k, x in sorted(hy.items())))
h1 = [np.mean(x) for k, x in sorted(hy.items())]
print("  100% を超えた半年は {} 回のうち {} 回".format(len(h1), sum(a > 100 for a in h1)))

print("\n■ 4) 選ばない買い方: 割安度 1.29 以上(0.775×割安度 ≧ 1)の組を、ワイド1.5〜30倍の全部で買う")
rows = [(d, p) for d, cs in races for ratio, w, p in cs if ratio >= 1.29]
print("  5年 {}点・{:.1f}%  年ごと [{}]".format(len(rows), roi(rows)[0], " ".join("{:.0f}".format(roi(rows, y)[0]) for y in YS)))
v2 = np.array([p for d, p in rows], dtype=float)
bs2 = np.array([rng.choice(v2, len(v2)).mean() for _ in range(3000)])
print("  95%の幅 {:.0f}〜{:.0f}%・100%を超えている見込み {:.0%}".format(np.percentile(bs2, 2.5), np.percentile(bs2, 97.5), (bs2 > 100).mean()))
for a, b in ((1.0, 1.1), (1.1, 1.2), (1.2, 1.29), (1.29, 1.4), (1.4, 9)):
    rr = [(d, p) for d, cs in races for ratio, w, p in cs if a <= ratio < b]
    print("  割安度 {:.2f}〜{:.2f}: {:>7}点 {:5.1f}%(理屈では {:.0f}%前後) 年ごと [{}]".format(a, b, len(rr), roi(rr)[0], 77.5 * (a + min(b, 1.6)) / 2,
          " ".join("{:.0f}".format(roi(rr, y)[0]) for y in YS)))

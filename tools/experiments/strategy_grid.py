"""今までの結果を合わせた買い方を、少ない条件の組で選ぶ(中央、2024年1月〜。予想はアプリと同じ)
  python tools/experiments/strategy_grid.py
  買う: ◎の確率 ≥ T1 かつ ◎と○の確率の差 ≥ G。買い方: 3連複◎○▲、▲が距離延長ならワイド◎-○。
  T1・G は2024・2025年で選び(2年で300レース以上)、2026年で確かめる"""
import os, sys, random
sys.argv = sys.argv[:1]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trio_third.py")
exec(open(__file__).read().split("# ---------------------------------------------------------------- 1.")[0])
YS = ("2024", "2025", "2026")
wide = lambda r: r["pay"]["ワイド"].get("-".join(str(x) for x in sorted((r["h"], r["t"]))), 0)
ext = lambda r: "距離延長" in (r["hs"][r["a"]].get("axis_ng") or [])
bet = lambda r: wide(r) if ext(r) else trio(r)
g12 = lambda r: r["hs"][r["h"]]["p3"] - r["hs"][r["t"]]["p3"]


def stat(xs):
    n = len(xs)
    return n, (sum(map(bet, xs)) / n if n else 0), (sum(1 for r in xs if bet(r)) / n if n else 0), sum(bet(r) - 100 for r in xs)


rows = []
for t1 in (.70, .75, .80, .85, .90):
    for g in (-1, .05, .10, .15, .20, .25, .30):
        pick = [r for r in races if r["p1"] >= t1 and g12(r) >= g]
        sel = [r for r in pick if r["y"] in ("2024", "2025")]
        te = [r for r in pick if r["y"] == "2026"]
        if len(sel) < 300:
            continue
        rows.append((stat(sel)[1], t1, g, stat(sel), stat(te), [stat([r for r in pick if r["y"] == y])[1] for y in YS]))
rows.sort(key=lambda x: -x[0])
print("◎≥T1・差≥G  2024-25(選ぶ) R 回収率 的中 / 2026(確かめ) R 回収率 的中 収支 / 年ごと")
for roi, t1, g, s, t, ys in rows[:15]:
    print("  ◎{:.0%}以上 差{:>4s}  {:4d}R {:4.0f}% {:3.0%} / {:3d}R {:4.0f}% {:3.0%} {:+6,}円 / [{}]".format(
        t1, "なし" if g < 0 else "{:.0%}".format(g), s[0], s[1], s[2], t[0], t[1], t[2], int(t[3]), " ".join("%.0f" % v for v in ys)))
print("  …(今: ◎80%以上・差なし)", next("{:4d}R {:4.0f}% / 2026 {:3d}R {:4.0f}%".format(s[0], s[1], t[0], t[1]) for _, t1, g, s, t, _ in rows if t1 == .8 and g < 0))

"""1番人気と◎のワイド1点(◎が1番人気なら単勝◎の1点)を買ったら(中央、2024年1月〜。予想はアプリと同じ)
  python tools/experiments/fav_honmei.py
  1番人気は確定オッズ(買う時のオッズとほぼ同じとみなす)。◎の確率ごと・年ごとに、今の買い方とも比べる"""
import os, sys
sys.argv = sys.argv[:1]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trio_third.py")
exec(open(__file__).read().split("# ---------------------------------------------------------------- 1.")[0])
YS = ("2024", "2025", "2026")
fav = lambda r: min(r["od"], key=lambda u: r["od"][u])
win = lambda r: (r["pay"].get("単勝") or {}).get(str(r["h"]), 0)
wide2 = lambda r, a, b: (r["pay"].get("ワイド") or {}).get("-".join(str(x) for x in sorted((a, b))), 0)


def plan(r):
    """(券種, 払戻)"""
    f = fav(r)
    return ("単勝", win(r)) if f == r["h"] else ("ワイド", wide2(r, r["h"], f))


ext = lambda r: "距離延長" in (r["hs"][r["a"]].get("axis_ng") or [])
now = lambda r: wide2(r, r["h"], r["t"]) if ext(r) else trio(r)
g12 = lambda r: r["hs"][r["h"]]["p3"] - r["hs"][r["t"]]["p3"]


def line(lab, xs, f):
    n = len(xs)
    if not n:
        return
    ys = " ".join("{:4.0f}".format(sum(f(r) for r in xs if r["y"] == y) / max(1, sum(1 for r in xs if r["y"] == y))) for y in YS)
    print("  {:30s} {:5d}R 回収率 {:4.0f}% [{}] 的中 {:3.0%} 収支 {:+,}円".format(
        lab, n, sum(map(f, xs)) / n, ys, sum(1 for r in xs if f(r)) / n, int(sum(f(r) - 100 for r in xs))))


for lab, cond in (("全レース", lambda r: True), ("◎50〜60%", lambda r: .5 <= r["p1"] < .6), ("◎60〜70%", lambda r: .6 <= r["p1"] < .7),
                  ("◎70〜80%", lambda r: .7 <= r["p1"] < .8), ("◎80%以上", lambda r: r["p1"] >= .8),
                  ("今の買うレース(◎80%・差15%)", lambda r: r["p1"] >= .8 and g12(r) >= .15)):
    xs = [r for r in races if cond(r)]
    print("\n■ {}  ({}レース。◎が1番人気 {:.0%})".format(lab, len(xs), sum(1 for r in xs if fav(r) == r["h"]) / max(1, len(xs))))
    line("この買い方(単勝◎ か ワイド◎-1番人気)", xs, lambda r: plan(r)[1])
    line("  うち◎が1番人気 → 単勝◎", [r for r in xs if fav(r) == r["h"]], win)
    line("  うち◎が1番人気でない → ワイド", [r for r in xs if fav(r) != r["h"]], lambda r: plan(r)[1])
    line("比べる: ワイド◎-○", xs, lambda r: wide2(r, r["h"], r["t"]))
    line("比べる: 単勝◎", xs, win)
    line("比べる: 今の買い方(3連複◎○▲など)", xs, now)
# ◎が1番人気でないとき: ◎の人気ごと
xs = [r for r in races if fav(r) != r["h"]]
print("\n■ ◎が1番人気でないレースのワイド◎-1番人気(◎の人気ごと)")
for k in (2, 3, 4, 5):
    s = [r for r in xs if r["nk"][r["h"]] == k] if k < 5 else [r for r in xs if r["nk"][r["h"]] >= 5]
    line("◎が{}番人気{}".format(k, "以下" if k == 5 else ""), s, lambda r: plan(r)[1])

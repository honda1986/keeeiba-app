"""3連複のオッズで絞ると良くなるか(中央、2024年1月〜。◎70%以上のレースの3連複◎-○-x の確定オッズ trio_final を使う)
  python tools/experiments/trio_odds.py
  1) 今の買い方(◎80%以上・◎と○の差15%以上、3連複◎○▲、▲が延長ならワイド)を、3連複◎○▲のオッズの帯で分ける
  2) 「オッズが X 倍未満なら見送り」の X を2024・2025年で選び、2026年で確かめる(◎の下限・差の下限も一緒に)
  3) 3連複が当たる確率をモデルの確率から出し(2024・2025年で決める)、期待値(確率×オッズ)で絞る
  オッズは確定オッズ(買うときは発走10分前ごろのオッズ。少しずれる)"""
import glob, json, math, os, random
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res, tf = r.get("pred") or {}, r.get("result") or {}, r.get("trio_final")
        if r.get("center") != "jra" or not p.get("model") or not tf or len(res.get("order") or []) < 3:
            continue
        hs = {h["umaban"]: h for h in p["horses"] if h.get("p3") is not None}
        h, t, aite = p.get("honmei"), p.get("taikou"), p.get("aite") or []
        if not h or not t or not aite:
            continue
        a = aite[0]
        key = "-".join(str(x) for x in sorted((h, t, a)))
        pay = res.get("payout") or {}
        races.append(dict(d=d, y=d[:4], h=h, t=t, a=a, p1=hs[h]["p3"], p2=hs[t]["p3"], pa=hs[a]["p3"], odds=tf.get(key),
                          ext="距離延長" in (hs[a].get("axis_ng") or []),
                          trio=(pay.get("3連複") or {}).get(key, 0),
                          wide=(pay.get("ワイド") or {}).get("-".join(str(x) for x in sorted((h, t))), 0)))
races = [r for r in races if r["odds"]]
print("3連複のオッズがあるレース", len(races), {y: sum(1 for r in races if r["y"] == y) for y in YS})
bet = lambda r: r["wide"] if r["ext"] else r["trio"]


def stat(xs, g=bet):
    n = len(xs)
    return n, (sum(map(g, xs)) / n if n else 0), (sum(1 for r in xs if g(r)) / n if n else 0)


def yr(xs, g=bet):
    return " ".join("{:4.0f}".format(stat([r for r in xs if r["y"] == y], g)[1]) if any(r["y"] == y for r in xs) else "   -" for y in YS)


buy = [r for r in races if r["p1"] >= .8 and r["p1"] - r["p2"] >= .15]
n, roi, hit = stat(buy)
print("\n■ 1. 今の買い方 {}R {:.0f}% [{}] 的中 {:.0%}。3連複◎○▲のオッズで分ける(▲が延長のレースもオッズの帯に入れる)".format(n, roi, yr(buy), hit))
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e9)):
    s = [r for r in buy if lo <= r["odds"] < hi]
    if s:
        n, roi, hit = stat(s)
        print("  {:>4g}〜{:<4s}倍 {:4d}R 回収率 {:4.0f}% [{}] 的中 {:3.0%}".format(lo, "{:g}".format(hi) if hi < 1e9 else "", n, roi, yr(s), hit))

print("\n■ 2. ◎の下限 × ◎と○の差の下限 × 3連複のオッズの下限(この倍率未満は見送り)を2024・2025年で選ぶ(2年で200R以上)")
grid = []
for t1 in (.7, .75, .8, .85):
    for g in (-1, .1, .15, .2):
        for x in (0, 3, 4, 5, 6, 8, 10):
            pick = [r for r in races if r["p1"] >= t1 and r["p1"] - r["p2"] >= g and r["odds"] >= x]
            sel = [r for r in pick if r["y"] != "2026"]; te = [r for r in pick if r["y"] == "2026"]
            if len(sel) >= 200:
                grid.append((stat(sel)[1], t1, g, x, stat(sel), stat(te), sum(bet(r) - 100 for r in te)))
grid.sort(key=lambda z: -z[0])
for roi, t1, g, x, s, te, pl in grid[:12]:
    print("  ◎{:.0%}以上 差{:>4s} {:>2g}倍以上  2024-25 {:4d}R {:4.0f}% 的中{:3.0%} / 2026 {:3d}R {:4.0f}% 的中{:3.0%} 収支{:+,}円".format(
        t1, "なし" if g < 0 else "{:.0%}".format(g), x, s[0], s[1], s[2], te[0], te[1], te[2], int(pl)))
now = next(z for z in grid if z[1] == .8 and z[2] == .15 and z[3] == 0)
print("  (今: ◎80%以上 差15% オッズで絞らない  2024-25 {}R {:.0f}% / 2026 {}R {:.0f}%)".format(now[4][0], now[4][1], now[5][0], now[5][1]))

print("\n■ 3. 3連複◎○▲が当たる確率(モデルの◎○▲の確率から。2024・2025年で決める)× オッズ = 期待値 で絞る")
X = lambda r: [1, math.log(r["p1"]), math.log(r["p2"]), math.log(r["pa"])]
tr = [r for r in races if r["y"] != "2026"]
Z = np.array([X(r) for r in tr]); yv = np.array([1.0 if r["trio"] else 0.0 for r in tr]); w = np.zeros(4)
for _ in range(50):
    pz = 1 / (1 + np.exp(-Z @ w)); H = (Z * (pz * (1 - pz))[:, None]).T @ Z + 1e-6 * np.eye(4); w += np.linalg.solve(H, Z.T @ (yv - pz))
print("  係数", " ".join("%.2f" % v for v in w))
for r in races:
    r["pt"] = 1 / (1 + math.exp(-np.dot(X(r), w))); r["ev"] = r["pt"] * r["odds"]
for lo, hi in ((0, .6), (.6, .8), (.8, 1.0), (1.0, 1.3), (1.3, 1.8), (1.8, 99)):
    s = [r for r in races if lo <= r["ev"] < hi]
    if s:
        n, roi, hit = stat(s, lambda r: r["trio"])
        print("  期待値 {:.1f}〜{:<4s} {:4d}R 3連複◎○▲ {:4.0f}% [{}] 的中 {:3.0%} 当たる確率の平均 {:.0%}".format(
            lo, "{:.1f}".format(hi) if hi < 99 else "", n, roi, yr(s, lambda r: r["trio"]), hit, sum(r["pt"] for r in s) / n))
for k in (1.0, 1.2, 1.5):
    s = [r for r in races if r["ev"] >= k]
    s2 = [r for r in buy if r["ev"] >= k]
    print("  期待値 {:.1f}以上: ◎70%以上の全部 {}R {:.0f}% [{}] / 今の買うレースのうち {}R {:.0f}% [{}]".format(
        k, len(s), stat(s, lambda r: r["trio"])[1], yr(s, lambda r: r["trio"]), len(s2), stat(s2)[1], yr(s2)))

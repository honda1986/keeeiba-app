"""1レース1000円(100円×10)を、今の買うレース(中央。◎80%以上・◎と○の差15%以上)の中でどう分けると収支が上がるか(2024〜2026年)
  python tools/experiments/stake_split.py
  点数を広げる(3連複の流し・◎1頭軸)・保険を足す(ワイド・馬連)・3連単を混ぜる・オッズで配る(均等払戻)・レースで金額を変える、を
  今の買い方(3連複◎-○-▲の1点。▲が距離延長ならワイド◎-○)と同じ1000円でくらべる。▲が延長のレースは、どの案でもワイド◎-○のまま
  (「延長も3連複」の案だけ別)。払戻は確定。均等払戻は確定の3連複オッズ(trio_final)で配るので、実際より少し良く出る"""
import glob, json, os
from collections import defaultdict
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in sorted((json.load(open(f)).get("races") or {}).items(), key=lambda x: x[1].get("start") or ""):
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or p.get("verdict") not in ("A", "B") or not p.get("bets") or len(res.get("order") or []) < 3:
            continue
        hs = sorted(p["horses"], key=lambda h: -(h.get("p3") or 0))
        h, t = p["honmei"], p["taikou"]
        rest = [x["umaban"] for x in hs if x["umaban"] not in (h, t)]
        a = (p.get("aite") or []) + [u for u in rest if u not in (p.get("aite") or [])]
        races.append(dict(d=d, y=d[:4], h=h, t=t, a=a[:3], pay=res["payout"], trio=r.get("trio_final") or {},
                          wide=p["bets"][0]["type"] == "ワイド", A=p["verdict"] == "A", gap=p.get("gap12") or 0))
print("買うレース {}R(うち▲が延長でワイド {}R)".format(len(races), sum(r["wide"] for r in races)))


def ret(r, typ, us, unit):
    """us の馬券を unit 単位(100円)買ったときの払戻(円)"""
    k = "-".join(map(str, us)) if typ in ("3連単", "馬単") else srt(us)
    return (r["pay"].get(typ) or {}).get(k, 0) * unit


def plan(r, tickets):
    """tickets = [(券種, 馬, 単位)]。▲延長のレースはワイド◎-○に10単位"""
    if r["wide"]:
        return 1000, ret(r, "ワイド", [r["h"], r["t"]], 10)
    return sum(u for *_, u in tickets) * 100, sum(ret(r, *tk) for tk in tickets)


def even(r, xs):
    """3連複◎-○-x を、どれが当たっても払戻がそろうように10単位を配る(確定オッズ。取れなければ同じだけ)"""
    od = [r["trio"].get(srt([r["h"], r["t"], x])) for x in xs]
    if not all(od):
        w = [1 / len(xs)] * len(xs)
    else:
        w = [1 / o for o in od]; w = [v / sum(w) for v in w]
    u = [max(1, round(10 * v)) for v in w]
    while sum(u) > 10: u[u.index(max(u))] -= 1
    while sum(u) < 10: u[u.index(min(u))] += 1
    return [("3連複", [r["h"], r["t"], x], n) for x, n in zip(xs, u)]


H, T = lambda r: r["h"], lambda r: r["t"]
S = {
    "今: 1点に1000円": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 10)]),
    "3連複 ◎○▲700 + ◎○△300": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 7), ("3連複", [r["h"], r["t"], r["a"][1]], 3)]),
    "3連複 ◎○-▲△ 500ずつ": lambda r: plan(r, [("3連複", [r["h"], r["t"], x], 5) for x in r["a"][:2]]),
    "3連複 ◎○-▲△☆ 400/300/300": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 4)] + [("3連複", [r["h"], r["t"], x], 3) for x in r["a"][1:3]]),
    "3連複 ◎1頭軸-○▲△ 500/250/250": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 5), ("3連複", [r["h"], r["t"], r["a"][1]], 3), ("3連複", [r["h"], r["a"][0], r["a"][1]], 2)]),
    "3連複◎○▲800 + ワイド◎-○200": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 8), ("ワイド", [r["h"], r["t"]], 2)]),
    "3連複◎○▲600 + ワイド◎-○400": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 6), ("ワイド", [r["h"], r["t"]], 4)]),
    "3連複◎○▲700 + 馬連◎-○300": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 7), ("馬連", [r["h"], r["t"]], 3)]),
    "3連複◎○▲600 + 3連単◎○→◎○→▲ 200ずつ": lambda r: plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 6), ("3連単", [r["h"], r["t"], r["a"][0]], 2), ("3連単", [r["t"], r["h"], r["a"][0]], 2)]),
    "3連単 ◎○→◎○→▲ 500ずつ": lambda r: plan(r, [("3連単", [r["h"], r["t"], r["a"][0]], 5), ("3連単", [r["t"], r["h"], r["a"][0]], 5)]),
    "均等払戻 3連複◎○-▲△": lambda r: plan(r, even(r, r["a"][:2])),
    "均等払戻 3連複◎○-▲△☆": lambda r: plan(r, even(r, r["a"][:3])),
    "延長も3連複◎○▲(ワイドにしない)": lambda r: (1000, ret(r, "3連複", [r["h"], r["t"], r["a"][0]], 10)),
}


def stats(rows):
    """rows = [(年, 賭け, 払戻)] -> 回収率・年ごと・収支・的中・長い連敗・深い落ち込み"""
    bet = sum(b for _, b, _ in rows); back = sum(x for *_, x in rows)
    yr = {y: (sum(x for yy, _, x in rows if yy == y) / max(1, sum(b for yy, b, _ in rows if yy == y))) for y in YS}
    run = mx = 0; cum = peak = dd = 0
    for _, b, x in rows:
        run = 0 if x > 0 else run + 1; mx = max(mx, run)
        cum += x - b; peak = max(peak, cum); dd = min(dd, cum - peak)
    hit = sum(x > 0 for *_, x in rows) / len(rows)
    return back / bet, yr, back - bet, hit, mx, dd


print("\n■ 1レース1000円の分け方(2024〜2026年)  回収率 [2024 2025 2026] / 収支 / 当たり / 長い連敗 / いちばん深い落ち込み")
for name, fn in S.items():
    rows = [(r["y"],) + fn(r) for r in races]
    roi, yr, pl, hit, mx, dd = stats(rows)
    print("  {:34s} {:4.0f}% [{}]  {:+9,.0f}円  {:3.0%}  {:2d}連敗  {:+8,.0f}円".format(
        name, roi * 100, " ".join("{:4.0f}".format(yr[y] * 100) for y in YS), pl, hit, mx, dd))

print("\n■ レースで金額を変える(平均が1000円になるように。買い方は今の1点)")
base = [plan(r, [("3連複", [r["h"], r["t"], r["a"][0]], 10)]) for r in races]
for label, key in (("A(◎85%以上) / B", lambda r: r["A"]), ("◎と○の差 25%以上 / 未満", lambda r: r["gap"] >= .25),
                   ("3連複◎○▲のオッズ 8倍以上 / 未満", lambda r: (r["trio"].get(srt([r["h"], r["t"], r["a"][0]])) or 0) >= 8)):
    g = [bool(key(r)) for r in races]
    for flag in (True, False):
        sub = [(r["y"],) + b for r, b, gg in zip(races, base, g) if gg == flag]
        roi, yr, *_ = stats(sub)
        print("  {:30s} {} {:4d}R {:4.0f}% [{}]".format(label, "○" if flag else "×", len(sub), roi * 100, " ".join("{:4.0f}".format(yr[y] * 100) for y in YS)))
    for w in (1.5, 2.0):
        n1 = sum(g); n0 = len(g) - n1; k = len(g) / (w * n1 + n0)      # 平均1000円にそろえる
        rows = [(r["y"], b[0] * (w if gg else 1) * k, b[1] * (w if gg else 1) * k) for r, b, gg in zip(races, base, g)]
        roi, yr, pl, hit, mx, dd = stats(rows)
        print("    ○を{}倍に厚く {:4.0f}% [{}]  {:+9,.0f}円  落ち込み {:+8,.0f}円".format(w, roi * 100, " ".join("{:4.0f}".format(yr[y] * 100) for y in YS), pl, dd))

"""ケリー基準で賭け金を決めたら(中央の今の買い方: 3連複◎-○-▲の1点。▲が延長ならワイド◎-○。2024〜2026年)
  python tools/experiments/kelly.py
  手持ち10万円から、レースの順(日付・発走時刻)に賭けていく。賭け金は100円単位(切り捨て。100円未満なら買わない)。
  - 定額: 100円 / 1000円
  - 定率: 手持ちの1% / 2%
  - 戦略全体のケリー: 1点の払戻の倍率 X(外れ 0)の分け方から、E[log(1 + f(X-1))] がいちばん大きい f。2024・2025年で決めて2026年で確かめる
  - レースごとのケリー: f = (p×オッズ - 1) / (オッズ - 1)。p はモデルの3着内の確率から Plackett-Luce で出した的中の確率、
    オッズは確定の3連複オッズ(trio_final)・ワイドは下限(wide_final)。確定オッズを使うので実際より少し良く出る
  最後に、2024〜2026年の払戻を混ぜて並べ替え、600レースを5,000回試して、終わりの手持ちと「半分以下に減った」率を出す"""
import glob, json, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pl import fit, combo_p
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))
rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or p.get("verdict") not in ("A", "B") or not p.get("bets") or len(res.get("order") or []) < 3:
            continue
        b = p["bets"][0]; c = b["combos"][0]
        X = (res["payout"].get(b["type"]) or {}).get(srt(c), 0) / 100
        od = res.get("odds") or {}
        hs = [h for h in p["horses"] if h.get("p3") is not None and (od.get(str(h["umaban"])) or 0) > 1]
        us = [h["umaban"] for h in hs]; ix = {u: i for i, u in enumerate(us)}
        pk = o = None
        if all(u in ix for u in c) and len(hs) >= 5:
            w = fit([h["p3"] for h in hs])
            pk = combo_p(w, [ix[u] for u in c], len(c))
            o = (r.get("trio_final") or {}).get(srt(c)) if b["type"] == "3連複" else ((r.get("wide_final") or {}).get(srt(c)) or [None])[0]
        rows.append(dict(d=d, st=r.get("start") or "", y=d[:4], X=X, p=pk, o=o, typ=b["type"]))
rows.sort(key=lambda x: (x["d"], x["st"]))
print("買うレース {}R(的中の確率とオッズがそろう {}R)".format(len(rows), sum(1 for x in rows if x["p"] and x["o"])))

# モデルの的中の確率は合っているか
pp = np.array([x["p"] for x in rows if x["p"] and x["o"]]); hit = np.array([x["X"] > 0 for x in rows if x["p"] and x["o"]])
ev = np.array([x["p"] * x["o"] for x in rows if x["p"] and x["o"]]); Xs = np.array([x["X"] for x in rows if x["p"] and x["o"]])
print("\n■ モデルの的中の確率と実際(買うレース)")
for lo, hi in ((0, .15), (.15, .25), (.25, .35), (.35, 1)):
    m = (pp >= lo) & (pp < hi)
    if m.any():
        print("  確率 {:3.0%}〜{:3.0%} {:4d}R 実際 {:4.0%} / 見込み {:4.0%}".format(lo, hi, int(m.sum()), hit[m].mean(), pp[m].mean()))
print("■ モデルの期待値(確率×オッズ)ごとの回収率")
for lo, hi in ((0, .8), (.8, 1.0), (1.0, 1.3), (1.3, 99)):
    m = (ev >= lo) & (ev < hi)
    if m.any():
        print("  期待値 {:.1f}〜{:.1f} {:4d}R 回収率 {:4.0f}%".format(lo, min(hi, 9.9), int(m.sum()), Xs[m].mean() * 100))


def kelly_f(X, grid=np.linspace(0, .3, 301)):
    """E[log(1 + f(X-1))] がいちばん大きい f(払戻の倍率 X の並び)"""
    X = np.asarray(X)
    return max(grid, key=lambda f: np.mean(np.log(np.maximum(1 + f * (X - 1), 1e-9))))


def run(seq, stake, bank0=100000):
    """stake(手持ち, レース) -> 賭け金。終わりの手持ち・いちばん深い落ち込み(率)・いちばん少ない手持ち"""
    bank = peak = low = bank0; dd = 0.0; n = 0
    for x in seq:
        s = int(min(stake(bank, x), bank) // 100 * 100)
        if s <= 0:
            continue
        n += 1
        bank += s * (x["X"] - 1)
        peak = max(peak, bank); low = min(low, bank); dd = max(dd, 1 - bank / peak)
    return bank, dd, low, n


fk = kelly_f([x["X"] for x in rows if x["y"] < "2026"])
print("\n■ 戦略全体のケリー: 2024・2025年の払戻から f = {:.1%}(1/2 {:.1%}・1/4 {:.1%})。3年分では {:.1%}".format(fk, fk / 2, fk / 4, kelly_f([x["X"] for x in rows])))


def per_race(mult):
    def st(bank, x):
        if not x["p"] or not x["o"] or x["o"] <= 1:
            return 0
        f = (x["p"] * x["o"] - 1) / (x["o"] - 1)
        return bank * max(0.0, f) * mult
    return st


PLANS = [("定額 100円", lambda b, x: 100), ("定額 1000円", lambda b, x: 1000),
         ("定率 1%", lambda b, x: b * .01), ("定率 2%", lambda b, x: b * .02),
         ("ケリー(全体) フル {:.1%}".format(fk), lambda b, x: b * fk), ("ケリー(全体) 1/2", lambda b, x: b * fk / 2),
         ("ケリー(全体) 1/4", lambda b, x: b * fk / 4),
         ("ケリー(レースごと) フル", per_race(1)), ("ケリー(レースごと) 1/2", per_race(.5)), ("ケリー(レースごと) 1/4", per_race(.25))]
for title, seq in (("2026年(確かめ。2026-01 に10万円から)", [x for x in rows if x["y"] == "2026"]), ("2024〜2026年(2024-01 に10万円から)", rows)):
    print("\n■ {}  {}R".format(title, len(seq)))
    for name, st in PLANS:
        bank, dd, low, n = run(seq, st)
        print("  {:26s} 終わり {:>12,.0f}円  いちばん深い落ち込み {:4.0%}  いちばん少ない {:>10,.0f}円  買った {}R".format(name, bank, dd, low, n))

# 並べ替えて何度も試す(払戻の分け方は2024〜2026年のまま、順番だけ変える)
print("\n■ 600レースを5,000回(払戻は2024〜2026年から引き直し)。終わりの手持ち(10万円から)の中央・下から10%、半分以下に減った回の率")
rng = np.random.default_rng(1)
Xall = np.array([x["X"] for x in rows])
for name, f in (("定率 1%", .01), ("定率 2%", .02), ("ケリー 1/4", fk / 4), ("ケリー 1/2", fk / 2), ("ケリー フル", fk), ("ケリーの2倍", fk * 2)):
    ends, halves = [], 0
    for _ in range(5000):
        bank = peak = 100000.0; half = False
        for X in rng.choice(Xall, 600):
            s = bank * f // 100 * 100
            bank += s * (X - 1)
            if bank <= 50000:
                half = True
        ends.append(bank); halves += half
    e = np.array(ends)
    print("  {:12s} f={:5.1%}  中央 {:>13,.0f}円  下から10% {:>11,.0f}円  半分以下に減った {:4.0%}".format(name, f, np.median(e), np.quantile(e, .1), halves / 5000))

# これからの回収率が下がったら、ケリーの割合はどうなるか(払戻を同じ割合で縮めて計算)
print("\n■ これからの回収率が下がったときのケリーの割合(2024〜2026年の払戻を縮めて計算)")
for target in (1.29, 1.20, 1.15, 1.10, 1.05):
    Xt = Xall * target / Xall.mean()
    f = kelly_f(Xt)
    print("  回収率 {:.0%} → フル {:4.1%}・1/2 {:4.1%}・1/4 {:4.1%}(10万円なら1レース {:,.0f}円 / {:,.0f}円 / {:,.0f}円)".format(
        target, f, f / 2, f / 4, 100000 * f // 100 * 100, 100000 * f / 2 // 100 * 100, 100000 * f / 4 // 100 * 100))

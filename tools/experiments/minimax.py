"""ミニマックス(いちばん悪い場合がいちばんましになるものを選ぶ)で予想・買い目を決めたら(中央。2024〜2026年)
  python tools/experiments/minimax.py
  1. 着順の最悪(ワルドのマキシミン): どの買い目も「全部外れ」が最悪。全部の組を買っても控除の分だけ必ず減るので、答えは「買わない」
  2. 見立ての最悪: 各馬の3着内の確率を「モデル」と「単勝オッズから出した確率(Harville)」の小さい方にして ◎○▲ と買うレースを決める
  3. 期待値の最悪: ◎○が決まったレース(◎70%以上。3連複◎-○-x の確定オッズ trio_final があるもの)で、3連複◎-○-x(x は全馬)と
     ワイド◎-○の期待値を、モデルの確率とオッズの確率の両方で出し、小さい方(ワイドはオッズの下限)がいちばん大きい買い目を買う
  基準(下限)は 2024・2025年で選び、2026年で確かめる。払戻は確定。確定オッズで期待値を出すので、実際より少し良く出る"""
import glob, json, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, R)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keiba_final as F
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))


from pl import fit, combo_p


races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or not p.get("honmei") or len(res.get("order") or []) < 3:
            continue
        od = res.get("odds") or {}
        hs = [h for h in p["horses"] if h.get("p3") is not None and (od.get(str(h["umaban"])) or 0) > 1]
        if len(hs) < 6:
            continue
        us = [h["umaban"] for h in hs]
        pm = np.array([h["p3"] for h in hs]); pm = np.minimum(pm * 3 / pm.sum(), .97)
        wk = np.array([1 / od[str(u)] for u in us]); wk /= wk.sum()
        pk = F.pl_top3(wk)
        races.append(dict(y=d[:4], us=us, pm=pm, pk=pk, wk=wk, ng=[set(h.get("axis_ng") or ()) for h in hs], n=len(hs),
                          pay=res["payout"], top3=res["order"][:3], cur=p, tf=r.get("trio_final") or {}, wf=r.get("wide_final") or {}))
print("中央 {}R(2024〜2026年)".format(len(races)))

# ---- 1. 着順の最悪
print("\n■ 1. 着順の最悪(マキシミン): 1点でも買えば最悪は「全部外れ」(−賭け金)。3連複を全部の組で買うと")
allbox = [r["pay"].get("3連複", {}).get(srt(r["top3"]), 0) / (100 * r["n"] * (r["n"] - 1) * (r["n"] - 2) / 6) for r in races]
print("  1レースの回収率 平均 {:.0%}・いちばん悪いレース {:.0%}・いちばん良いレース {:.0%}  → 最悪をいちばんましにするのは「買わない」".format(
    np.mean(allbox), min(allbox), max(allbox)))


def bet_ret(r, h, t, a):
    """今の買い方: 3連複◎○▲、▲が延長ならワイド◎-○"""
    if "距離延長" in r["ng"][r["us"].index(a)]:
        return r["pay"].get("ワイド", {}).get(srt([h, t]), 0)
    return r["pay"].get("3連複", {}).get(srt([h, t, a]), 0)


def marks(r, p):
    """軸にできる馬の p の上位2頭を◎○、残りの1番目を▲"""
    order = np.argsort(-p)
    ok = [i for i in order if not r["ng"][i]]
    if len(ok) < 2:
        return None
    h, t = ok[0], ok[1]
    a = next(i for i in order if i not in (h, t))
    return h, t, a


def summary(name, rows):
    def roi(m):
        return np.mean([x for _, x in m]) if m else 0
    print("  {:44s} {:4d}R {:4.0f}% [{}]  的中 {:3.0%}".format(name, len(rows), roi(rows),
          " ".join("{:4.0f}".format(roi([x for x in rows if x[0] == y])) if sum(x[0] == y for x in rows) >= 10 else "   -" for y in YS),
          np.mean([x > 0 for _, x in rows]) if rows else 0))


# ---- 2. 見立ての最悪
print("\n■ 2. 見立ての最悪: 3着内の確率 = min(モデル, オッズ)。◎○▲と買うレースをそれで決める  [2024 2025 2026]")
cur = [(r["y"], sum(r["pay"].get(b["type"], {}).get(srt(c), 0) for b in r["cur"]["bets"] for c in b["combos"]))
       for r in races if r["cur"].get("verdict") in ("A", "B") and r["cur"].get("bets")]
summary("今(モデルの確率。◎80%以上・◎と○の差15%以上)", cur)
cands = {}
for r in races:
    for kind, p in (("model", r["pm"]), ("minimax", np.minimum(r["pm"], r["pk"]))):
        mk = marks(r, p)
        if not mk:
            continue
        h, t, a = mk
        cands.setdefault(kind, []).append((r["y"], p[h], p[h] - p[t], bet_ret(r, r["us"][h], r["us"][t], r["us"][a])))
best = None
for t1 in (.5, .55, .6, .65, .7, .75, .8):
    for g in (0, .05, .1, .15, .2):
        m = [x for x in cands["minimax"] if x[1] >= t1 and x[2] >= g]
        tr = [x[3] for x in m if x[0] < "2026"]
        if len(tr) >= 150 and (best is None or np.mean(tr) > best[0]):
            best = (np.mean(tr), t1, g)
_, t1, g = best
summary("ミニマックスの◎○▲(◎{:.0%}以上・差{:.0%}以上。2024・2025で選んだ)".format(t1, g),
        [(x[0], x[3]) for x in cands["minimax"] if x[1] >= t1 and x[2] >= g])
summary("ミニマックスの◎○▲(今と同じ ◎80%以上・差15%以上)", [(x[0], x[3]) for x in cands["minimax"] if x[1] >= .8 and x[2] >= .15])
same = sum(1 for r in races if marks(r, r["pm"]) and marks(r, np.minimum(r["pm"], r["pk"])) and marks(r, r["pm"])[:3] == marks(r, np.minimum(r["pm"], r["pk"]))[:3])
print("  ◎○▲がモデルと同じになるレース {:.0%}".format(same / len(races)))

# ---- 3. 期待値の最悪
print("\n■ 3. 期待値の最悪: 3連複◎-○-x とワイド◎-○の中で、min(モデルの期待値, オッズの期待値) がいちばん大きい1点を、それが下限以上なら買う")
rows = []
for r in races:
    if not r["tf"] or r["cur"].get("taikou") is None:
        continue
    us = r["us"]; ix = {u: i for i, u in enumerate(us)}
    h, t = r["cur"]["honmei"], r["cur"]["taikou"]
    if h not in ix or t not in ix:
        continue
    wm = fit(r["pm"]); opts = []
    for x in us:
        if x in (h, t):
            continue
        o = r["tf"].get(srt([h, t, x]))
        if not o:
            continue
        c = [ix[h], ix[t], ix[x]]
        opts.append((min(combo_p(wm, c, 3), combo_p(r["wk"], c, 3)) * o, "3連複", srt([h, t, x])))
    wo = r["wf"].get(srt([h, t]))
    if wo:
        c = [ix[h], ix[t]]
        opts.append((min(combo_p(wm, c, 2), combo_p(r["wk"], c, 2)) * wo[0], "ワイド", srt([h, t])))
    if not opts:
        continue
    ev, typ, k = max(opts)
    rows.append((r["y"], ev, r["pay"].get(typ, {}).get(k, 0), typ, r["cur"].get("verdict") in ("A", "B")))
print("  対象 {}R。いちばん良い1点の最悪の期待値: 平均 {:.2f}・1.0以上 {}R".format(len(rows), np.mean([x[1] for x in rows]), sum(x[1] >= 1 for x in rows)))
for th in (0.8, 0.9, 1.0, 1.1):
    m = [(y, ret) for y, ev, ret, typ, b in rows if ev >= th]
    summary("最悪の期待値 {:.1f}以上(3連複 {} / ワイド {})".format(th, sum(1 for x in rows if x[1] >= th and x[3] == "3連複"),
            sum(1 for x in rows if x[1] >= th and x[3] == "ワイド")), m)
    m = [(y, ret) for y, ev, ret, typ, b in rows if ev >= th and b]
    summary("  └ 今の買うレースだけ", m)

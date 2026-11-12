"""「ワイドは2倍以上・3連複は3倍以上だけ買う」を、買い方選びに使っていない年も入れて確かめる(中央 2022〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/min_odds_fresh.py
  PRED は fresh_years.py を YEARS=2022,2023,2024,2025,2026 で回して作った確率(5年とも、その年を除いて学んだモデル)。
  その確率で今の買い方(◎80%以上・差15%以上、3連複◎-○-▲、▲が延長ならワイド◎-○)の買い目を作り、確定オッズで分ける。
  3連複◎-○-▲ のオッズは、取ってあれば(trio_final)それ、当たっていれば払戻、どちらも無ければ
  3つの組の強さ(trio_pair_final)から見積もる(見積もりの式は、オッズが分かっている組で決める)"""
import gc, os, pickle, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
YS = ("2022", "2023", "2024", "2025", "2026")
srt = lambda c: "-".join(str(x) for x in sorted(c))
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
pred = np.load(os.environ["PRED"])
info = {rc["id"]: {"id": rc["id"], "center": rc.get("center"), "result": rc["result"], "trio_wide_u": T.trio_wide_us(rc),
                   "tf": rc.get("trio_final") or {}, "tp": rc.get("trio_pair_final") or {}, "wf": rc.get("wide_final") or {}} for d, rc in races}
del races; gc.collect()

# 3連複のオッズの見積もり: log(1/オッズ) を、3つの組の強さの log で(オッズが分かっている組で決める)
X, Y = [], []
for x in info.values():
    for k, o in x["tf"].items():
        a, b, c = k.split("-")
        s = [x["tp"].get(a + "-" + b), x["tp"].get(a + "-" + c), x["tp"].get(b + "-" + c)]
        if all(s) and o > 1:
            X.append([1.0] + sorted(np.log(s))); Y.append(np.log(1 / o))
X, Y = np.array(X), np.array(Y)
half = np.arange(len(X)) % 2 == 0
coef = np.linalg.lstsq(X[half], Y[half], rcond=None)[0]
est = X[~half] @ coef
print("3連複のオッズの見積もり: {}組で決めて {}組で確かめる。相関 {:.3f}・3倍以上かどうかが合う割合 {:.1%}".format(
    half.sum(), (~half).sum(), np.corrcoef(est, Y[~half])[0, 1], np.mean((np.exp(-est) >= 3) == (np.exp(-Y[~half]) >= 3))))
coef = np.linalg.lstsq(X, Y, rcond=None)[0]

by = defaultdict(list)
for i, r in enumerate(rows):
    if not np.isnan(pred[i]):
        by[r["rid"]].append(i)
pw = {}
for rid, ix in by.items():
    tot = pred[ix].sum(); want = min(3, len(ix))
    for i in ix:
        pw[(rid, rows[i]["u"])] = min(0.99, pred[i] * want / tot)
jg = T.group(T.of_center(rows, "jra"))
pick = T.pick_sel(T.fixed_selects()["jra"], pw)
bets = []      # (年, 券種, オッズ(ワイドは下限), 払戻, オッズの出どころ)
for rid, rs in jg.items():
    if (rid, rs[0]["u"]) not in pw:
        continue
    s_ = pick(rid, rs, None)
    if not s_ or not s_[2]:
        continue
    x = info[rid]; y = rs[0]["date"][:4]; pay = x["result"].get("payout") or {}
    h, t, a = s_[0], s_[1], s_[2][0]
    if T.trio_wide(x, a):
        k = srt((h, t)); w = x["wf"].get(k)
        if w:
            bets.append((y, "ワイド", w[0], pay.get("ワイド", {}).get(k, 0), "確定"))
    else:
        k = srt((h, t, a)); p = pay.get("3連複", {}).get(k, 0)
        if k in x["tf"]:
            bets.append((y, "3連複", x["tf"][k], p, "確定"))
        elif p:
            bets.append((y, "3連複", p / 100, p, "払戻"))
        else:
            u = [str(v) for v in sorted((h, t, a))]
            s = [x["tp"].get(u[0] + "-" + u[1]), x["tp"].get(u[0] + "-" + u[2]), x["tp"].get(u[1] + "-" + u[2])]
            if all(s):
                bets.append((y, "3連複", float(np.exp(-(np.array([1.0] + sorted(np.log(s))) @ coef))), p, "見積もり"))
src = defaultdict(int)
for b in bets:
    src[b[1] + " " + b[4]] += 1
print("買い目 {}点: {}".format(len(bets), dict(src)))


def line(name, sel, ys=YS):
    v = [b[3] for b in sel]
    by_ = defaultdict(list)
    for b in sel:
        by_[b[0]].append(b[3])
    print("  {:34s} {:4d}R {:5.1f}% [{}] 的中 {:4.1%} 収支 {:+7,.0f}円".format(name, len(v), np.mean(v) if v else 0, " ".join(
        "{:4.0f}".format(np.mean(by_[y])) if by_[y] else "   -" for y in ys), np.mean([x > 0 for x in v]) if v else 0, sum(v) - 100 * len(v)))


ok = lambda b, wmin=2, tmin=3: b[2] >= (wmin if b[1] == "ワイド" else tmin)
for title, ys in (("2022・2023年(買い方選びに使っていない年)", ("2022", "2023")), ("2024〜2026年(買い方を選んだ年)", ("2024", "2025", "2026")), ("5年", YS)):
    S = [b for b in bets if b[0] in ys]
    print("\n■ {}  回収率 [{}]".format(title, " ".join(ys)))
    line("今の買い方(全部)", S, ys)
    line("ワイド2倍以上・3連複3倍以上だけ", [b for b in S if ok(b)], ys)
    line("  買わなくなるレース", [b for b in S if not ok(b)], ys)
    for t in ("ワイド", "3連複"):
        line("  {} 基準以上".format(t), [b for b in S if b[1] == t and ok(b)], ys)
        line("  {} 基準未満".format(t), [b for b in S if b[1] == t and not ok(b)], ys)
print("\n■ 5年で下限を変える(回収率・レース数)。行 = ワイドの下限、列 = 3連複の下限")
tms = (0, 2, 3, 4, 5, 7)
print("  {:11s}".format("") + "".join("{:>14}".format("3連複{}倍〜".format(t) if t else "3連複 全部") for t in tms))
for w in (0, 1.5, 2, 2.5, 3):
    print("  {:11s}".format("ワイド{}倍〜".format(w) if w else "ワイド 全部") + " ".join(
        "{:>7.0f}%({:>3})".format(np.mean([b[3] for b in bets if ok(b, w, t)]), sum(ok(b, w, t) for b in bets)) for t in tms))
rng = np.random.default_rng(0)
for nm, sel in (("5年 全部", bets), ("5年 基準以上だけ", [b for b in bets if ok(b)])):
    v = np.array([b[3] for b in sel], dtype=float)
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(5000)])
    print("  {} {:.0f}%({}R) 95%の幅 {:.0f}〜{:.0f}%・100%を超えている見込み {:.0%}".format(nm, v.mean(), len(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 100).mean()))

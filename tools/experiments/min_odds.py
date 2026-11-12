"""オッズが低いレースは買わない、を今の買い方に足すとどうなるか(中央 2024〜2026年。アプリの予想・確定オッズ)
  python tools/experiments/min_odds.py
  今の買い方は 3連複◎-○-▲ の1点(▲が距離延長のレースはワイド◎-○)。「ワイドは2倍以上、3連複は3倍以上のレースだけ買う」と、
  下限を変えた場合を、年ごとに出す。ワイドのオッズは下限〜上限で出るので、下限が基準以上のものを買う(中央の値でも出す)。
  3連複のオッズが取れていないレース(外れで、◎○が今の予想と違っていたもの)は、どの下限でも数えない"""
import glob, json, os
from collections import defaultdict
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
rows = []      # (年, 券種, オッズ(ワイドは下限), ワイドの中央, 払戻)
unknown = 0
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or p.get("verdict") not in ("A", "B") or not p.get("bets") or len(res.get("order") or []) < 3:
            continue
        b = p["bets"][0]; key = "-".join(str(x) for x in sorted(b["combos"][0]))
        pay = ((res.get("payout") or {}).get(b["type"]) or {}).get(key, 0)
        if b["type"] == "3連複":
            o = (r.get("trio_final") or {}).get(key) or (pay / 100 if pay else None)
            if o is None:
                unknown += 1
                continue
            rows.append((d[:4], "3連複", o, o, pay))
        else:
            w = (r.get("wide_final") or {}).get(key)
            if not w:
                unknown += 1
                continue
            rows.append((d[:4], "ワイド", w[0], (w[0] + w[1]) / 2, pay))
print("買ったレース {}R(3連複 {}・ワイド {})。オッズが分からず数えないもの {}R".format(
    len(rows), sum(r[1] == "3連複" for r in rows), sum(r[1] == "ワイド" for r in rows), unknown))


def line(name, sel):
    v = [r[4] for r in sel]
    if not v:
        print("  {:38s}    0R".format(name)); return
    by = defaultdict(list)
    for r in sel:
        by[r[0]].append(r[4])
    print("  {:38s} {:4d}R {:5.1f}% [{}] 的中 {:4.1%} 収支 {:+7,.0f}円".format(name, len(v), np.mean(v), " ".join(
        "{:4.0f}".format(np.mean(by[y])) if by[y] else "   -" for y in YS), np.mean([x > 0 for x in v]), sum(v) - 100 * len(v)))


ok = lambda r, wmin, tmin, mid=False: (r[3 if mid else 2] >= wmin) if r[1] == "ワイド" else (r[2] >= tmin)
print("\n■ 今の買い方に「ワイドは2倍以上・3連複は3倍以上」を足す  回収率 [2024 2025 2026]")
line("今の買い方(全部)", rows)
line("ワイド2倍以上・3連複3倍以上だけ買う", [r for r in rows if ok(r, 2, 3)])
line("  買わなくなるレース", [r for r in rows if not ok(r, 2, 3)])
line("(ワイドは下限でなく中央の値が2倍以上)", [r for r in rows if ok(r, 2, 3, True)])
print("\n■ 券種ごと")
for t, ths in (("ワイド", (0, 1.5, 2, 2.5, 3, 4)), ("3連複", (0, 2, 3, 4, 5, 7, 10, 15))):
    sub = [r for r in rows if r[1] == t]
    for th in ths:
        line("{} {}倍以上".format(t, th) if th else "{} 全部".format(t), [r for r in sub if r[2] >= th])
    for a, b in zip(ths[:-1], ths[1:]):
        line("  {} {}〜{}倍".format(t, a, b), [r for r in sub if a <= r[2] < b])
    line("  {} {}倍以上".format(t, ths[-1]), [r for r in sub if r[2] >= ths[-1]])
print("\n■ 下限を変えた表(行 = ワイドの下限、列 = 3連複の下限)。回収率(レース数)")
tms = (0, 2, 3, 4, 5, 7)
print("  {:10s}".format("") + "".join("{:>14}".format("3連複{}倍〜".format(t) if t else "3連複 全部") for t in tms))
for w in (0, 1.5, 2, 2.5, 3):
    out = []
    for t in tms:
        sel = [r[4] for r in rows if ok(r, w, t)]
        out.append("{:>7.0f}%({:>3})".format(np.mean(sel), len(sel)))
    print("  {:10s}".format("ワイド{}倍〜".format(w) if w else "ワイド 全部") + " ".join(out))

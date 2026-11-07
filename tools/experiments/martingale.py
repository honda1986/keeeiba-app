"""◎の単勝をマーチンゲールで買ったら(中央、2024年1月〜。今の買うレースだけ、発走順。オッズ・払戻は確定)
  python tools/experiments/martingale.py [始めの額]
  倍賭け型 … 外れたら次は倍、当たったら始めの額に戻す
  取り戻し型 … 当たればそれまでの負け + 始めの額が戻る額を賭ける(オッズは確定オッズを買うときのオッズとみなす。100円単位で切り上げ)"""
import glob, json, math, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
BASE = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rs = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or p.get("verdict") not in ("A", "B") or not res.get("order"):
            continue
        h = p["honmei"]
        od = (res.get("odds") or {}).get(str(h))
        pay = ((res.get("payout") or {}).get("単勝") or {}).get(str(h), 0)
        if od:
            rs.append((os.path.basename(f)[:8], r.get("start") or "99:99", rid, od, pay))
rs.sort()
print("買うレース {}R({}〜{})。◎の単勝の的中 {:.0%}、平均オッズ {:.2f}、1点100円ずつなら回収率 {:.0f}%".format(
    len(rs), rs[0][0], rs[-1][0], sum(1 for x in rs if x[4]) / len(rs), sum(x[3] for x in rs) / len(rs), sum(x[4] for x in rs) / len(rs)))


def run(kind):
    cum = loss = 0; stake = BASE; peak = 0; dd = 0; mx = 0; streak = mxs = 0; total = 0; by = {}
    for d, st, rid, od, pay in rs:
        if kind == "取り戻し型":
            stake = max(BASE, math.ceil((loss + BASE) / max(od - 1, 0.01) / 100) * 100)
        mx = max(mx, stake); total += stake
        ret = int(pay * stake / 100)
        cum += ret - stake
        by[d[:4]] = by.get(d[:4], 0) + ret - stake
        if ret:
            streak = 0; loss = 0
            if kind == "倍賭け型":
                stake = BASE
        else:
            streak += 1; mxs = max(mxs, streak); loss += stake
            if kind == "倍賭け型":
                stake *= 2
        peak = max(peak, cum); dd = min(dd, cum - peak)
    return cum, total, mx, mxs, dd, by


for kind in ("倍賭け型", "取り戻し型", "毎回同じ額"):
    if kind == "毎回同じ額":
        cum = sum(int(x[4] * BASE / 100) - BASE for x in rs); total = BASE * len(rs)
        by = {}
        for x in rs:
            by[x[0][:4]] = by.get(x[0][:4], 0) + int(x[4] * BASE / 100) - BASE
        c = peak = dd = 0
        for x in rs:
            c += int(x[4] * BASE / 100) - BASE; peak = max(peak, c); dd = min(dd, c - peak)
        print("\n■ {}({:,}円ずつ): 収支 {:+,}円 / 投資 {:,}円 / 回収率 {:.0f}% / いちばん深い落ち込み {:,}円 / 年ごと {}".format(
            kind, BASE, cum, total, (cum + total) / total * 100, dd, " ".join("{} {:+,}".format(y, v) for y, v in sorted(by.items()))))
        continue
    cum, total, mx, mxs, dd, by = run(kind)
    print("\n■ {}: 収支 {:+,}円 / 投資 {:,}円 / 回収率 {:.0f}% / 1回の最高額 {:,}円 / いちばん長い連敗 {} / いちばん深い落ち込み {:,}円 / 年ごと {}".format(
        kind, cum, total, (cum + total) / total * 100, mx, mxs, dd, " ".join("{} {:+,}".format(y, v) for y, v in sorted(by.items()))))

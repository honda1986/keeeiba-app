"""rule_review.py で作った月ごとの確率(base / tag)を比べる。当たり方(AUC・対数損失)と、ワイド◎-○の回収率を年ごとに
  KEIBA_WORK=<作業フォルダ> python tools/experiments/rule_report.py"""
import glob, gzip, json, math, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
WORK = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
sets = {"今のモデル(repo)": json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]}
imps = {}
for name in ("base", "tag"):
    p = os.path.join(WORK, "wf_%s.json.gz" % name)
    if os.path.exists(p):
        j = json.load(gzip.open(p))
        sets[name], imps[name] = j["probs"], j["imp"]
races = {}
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = json.load(open(f))
    for rid, r in (d.get("races") or {}).items():
        res = r.get("result") or {}
        if r.get("center") != "jra" or len(res.get("order") or []) < 3 or not (res.get("payout") or {}).get("ワイド"):
            continue
        ng = {str(h["umaban"]) for h in ((r.get("pred") or {}).get("horses") or []) if h.get("axis_ng")}
        races[rid] = (os.path.basename(f)[:4], set(map(str, res["order"][:3])), ng, res["payout"]["ワイド"])
common = set(races)
for pr in sets.values():
    common &= set(pr)
YEARS = ("2024", "2025", "2026")


def auc(pairs):
    pairs.sort(key=lambda x: x[0])
    pos = sum(y for _, y in pairs); neg = len(pairs) - pos
    rank = s = 0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        s += avg * sum(y for _, y in pairs[i:j])
        i = j
    return (s - pos * (pos + 1) / 2) / (pos * neg)


print("同じ {} レースで比べる".format(len(common)))
print("\n当たり方  AUC(年ごと) / 対数損失(計。小さいほど良い)")
for name, pr in sets.items():
    a, ll, n = [], 0.0, 0
    for y in YEARS:
        pairs = []
        for rid in common:
            yy, top3, _, _ = races[rid]
            if yy != y:
                continue
            for u, p in pr[rid].items():
                t = int(u in top3)
                pairs.append((p, t))
                p = min(max(p, 1e-4), 1 - 1e-4)
                ll -= t * math.log(p) + (1 - t) * math.log(1 - p)
                n += 1
        a.append(auc(pairs))
    print("  {:14s} {}  / {:.4f}".format(name, " ".join("%.4f" % x for x in a), ll / n))

CONDS = (("全レース", lambda a, b: True), ("今の基準 ◎60+ ○40-50・○≧3番手", None), ("◎80%以上", lambda a, b: a >= .8),
         ("◎85%以上", lambda a, b: a >= .85), ("◎75%以上", lambda a, b: a >= .75))
print("\nワイド◎-○の回収率  2024 2025 2026 | 計")
for label, cond in CONDS:
    print("■", label)
    for name, pr in sets.items():
        out = []
        for y in YEARS:
            n = ret = hit = 0
            for rid in common:
                yy, _, ng, pay = races[rid]
                if yy != y:
                    continue
                hs = sorted(pr[rid].items(), key=lambda x: -x[1])
                ax = [x for x in hs if x[0] not in ng]
                if len(hs) < 5 or len(ax) < 2:
                    continue
                (h, p1), (t, p2) = ax[0], ax[1]
                p3 = max(p for u, p in hs if u not in (h, t))
                ok = (p1 >= .6 and .4 <= p2 < .5 and p2 >= p3) if cond is None else cond(p1, p2)
                if not ok:
                    continue
                v = pay.get("-".join(sorted((h, t), key=int)), 0)
                n += 1; ret += v; hit += v > 0
            out.append((n, ret, hit))
        tn, tr, th = (sum(x[i] for x in out) for i in range(3))
        print("  {:14s} {} | {:5d}R 的中{:4.0%} 計{:4.0f}%".format(name, " ".join("%4.0f%%" % (r / max(n, 1)) for n, r, _ in out),
                                                              tn, th / max(tn, 1), tr / max(tn, 1)))
if "tag" in imps:
    print("\n1つずつの材料のうち、効いている順(gain。全{}個の材料の中の順位)".format(len(imps["tag"])))
    for i, (k, v) in enumerate(imps["tag"]):
        if k.startswith(("tag:", "axis:")):
            print("  {:3d}位 {:24s} {}".format(i + 1, k, v))

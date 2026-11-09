"""「負けた相手のその後」(前走のレースの強さ)は、今のモデルが見落としている情報か(中央。2024〜2026年の予想で確かめる)
  python tools/experiments/race_level.py
  前走のレースの強さ = 前走で一緒に走った馬が、その後(今回のレースより前)の次の1走で3着内に来た率(頭数で平均に寄せる)。
  前走で先着された相手だけの強さも出す。今のモデルの3着内の確率(再現の予想)と実際をくらべ、強さで分けて
  実際 / 見込み がそろってずれるなら、モデルにまだ入っていない情報。2022年からの中央の結果(日ごとのファイル)だけで作る"""
import bisect, glob, json, os
from collections import defaultdict
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2024", "2025", "2026")
runs = defaultdict(list)          # horse_id -> [(日付, rid, 着順, 頭数)]
field = {}                        # rid -> [(horse_id, 着順)]
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[2-6]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        res = r.get("result") or {}
        if r.get("center") != "jra" or len(res.get("order") or []) < 3:
            continue
        hid = {h["umaban"]: h.get("horse_id") for h in r.get("horses") or [] if h.get("horse_id")}
        rank = {u: i + 1 for i, u in enumerate(res["order"])}
        n = len(res["order"])
        field[rid] = [(hid[u], rank[u]) for u in res["order"] if u in hid]
        for u in res["order"]:
            if u in hid:
                runs[hid[u]].append((d, rid, rank[u], n))
        p = r.get("pred") or {}
        if d >= "20240101" and p.get("model"):
            races.append((d, rid, r, hid, rank))
for v in runs.values():
    v.sort()
dates = {h: [x[0] for x in v] for h, v in runs.items()}
BASE = np.mean([x[2] <= 3 for v in runs.values() for x in v])


def level(rid, rdate, me, rank_me, T, ahead_only=False):
    """rid(日付 rdate)で一緒に走った馬の、その後 T より前の次の1走の3着内率(平均に寄せる)と数"""
    hits = cnt = 0
    for o, rk in field.get(rid, []):
        if o == me or (ahead_only and rk >= rank_me):
            continue
        ds = dates.get(o, [])
        i = bisect.bisect_right(ds, rdate)
        if i < len(ds) and ds[i] < T:
            cnt += 1; hits += runs[o][i][2] <= 3
    k = 5.0
    return (hits + k * BASE) / (cnt + k), cnt


rows = []
for d, rid, r, hid, rank in races:
    for h in r["pred"]["horses"]:
        u = h["umaban"]; me = hid.get(u)
        if not me or h.get("p3") is None or u not in rank:
            continue
        ds = dates.get(me, [])
        i = bisect.bisect_left(ds, d) - 1          # 前走(今回より前の最後の中央のレース)
        if i < 0:
            continue
        pd_, prid, prk, pn = runs[me][i]
        lv, c = level(prid, pd_, me, prk, d)
        la, ca = level(prid, pd_, me, prk, d, ahead_only=True)
        rows.append(dict(y=d[:4], p3=h["p3"], top3=rank[u] <= 3, lv=lv, c=c, la=la, ca=ca, prk=prk, pn=pn))
print("中央 2024〜2026年の予想 {:,}頭(前走が2022年からの中央にある馬)。その後の次走で3着内の平均 {:.1%}".format(len(rows), BASE))
lv = np.array([x["lv"] for x in rows]); q = np.quantile(lv, [1 / 3, 2 / 3])


def show(title, key, labels, sub=lambda x: True):
    print("\n■ " + title + "  実際 / 見込み(モデル)  [2024 2025 2026]")
    for lab, fn in labels:
        m = [x for x in rows if sub(x) and fn(x)]
        if len(m) < 300:
            continue
        ae = lambda ms: sum(x["top3"] for x in ms) / sum(x["p3"] for x in ms)
        print("  {:26s} {:6,d}頭 3着内 {:4.1%} / 見込み {:4.1%}  実/見 {:4.2f}  [{}]".format(lab, len(m), np.mean([x["top3"] for x in m]),
              np.mean([x["p3"] for x in m]), ae(m), " ".join("{:4.2f}".format(ae([x for x in m if x["y"] == y])) for y in YS)))


T3 = (("前走の相手が弱い(下1/3)", lambda x: x["lv"] < q[0]), ("ふつう", lambda x: q[0] <= x["lv"] < q[1]), ("前走の相手が強い(上1/3)", lambda x: x["lv"] >= q[1]))
show("前走のレースの強さ(全員の次走)", "lv", T3)
for lab, sub in (("前走1〜3着", lambda x: x["prk"] <= 3), ("前走4〜6着", lambda x: 4 <= x["prk"] <= 6), ("前走7着以下", lambda x: x["prk"] >= 7)):
    show("前走の強さ × " + lab, "lv", T3, sub)
qa = np.quantile([x["la"] for x in rows if x["prk"] >= 2], [1 / 3, 2 / 3])
show("前走で先着された相手の強さ(前走2着以下)", "la", (("先着した相手が弱い", lambda x: x["la"] < qa[0]), ("ふつう", lambda x: qa[0] <= x["la"] < qa[1]),
     ("先着した相手が強い", lambda x: x["la"] >= qa[1])), lambda x: x["prk"] >= 2)

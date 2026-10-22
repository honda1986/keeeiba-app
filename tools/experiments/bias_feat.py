"""当日の馬場の偏り(同じ日・同じ場・同じ芝ダの、前のレースまでの結果)を材料にする。
   bias_in    … 3着内の馬の内枠寄り(0.5 − 枠の位置)の平均。+ ほど内が来ている
   bias_front … 3着内に占める「ふだん逃げ・先行の馬」の割合 − 出走馬に占める割合。+ ほど前が残っている
   fit_in / fit_front … その馬が偏りに合うか(内枠 × bias_in、逃げ先行 × bias_front)
   prev_* … 前の開催日(3日以内)の同じ場・芝ダの全レースの偏り"""
import os, pickle, sys
from collections import defaultdict
from datetime import date
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train_model as T
SP = os.environ.get("KEIBA_WORK", "/tmp/keiba_work")
_, rows, _, _ = pickle.load(open(T.CACHE, "rb"))
F = {(r["rid"], r["u"]): r["f"] for r in rows}
groups = defaultdict(list)
for d, rc in T.load_races():
    groups[(d, rc["venue"], rc["surface"])].append(rc)


def summary(rc):
    """1レース分: 3着内の (内寄り, 逃げ先行) と出走馬の逃げ先行の割合"""
    hs = [h for h in rc["horses"] if not h.get("scratched")]
    if len(hs) < 5:
        return None
    maxw = max(h.get("waku") or 1 for h in hs)
    top3 = set(rc["result"]["order"][:3])
    def front(h):
        st = (F.get((rc["id"], h["umaban"])) or {}).get("style")
        return None if st is None else (1.0 if st <= 1 else 0.0)
    fr = [front(h) for h in hs]
    fr_known = [x for x in fr if x is not None]
    t3 = [h for h in hs if h["umaban"] in top3]
    pos = lambda h: 0.5 - ((h.get("waku") or 1) - 1) / max(maxw - 1, 1)
    # 3着内の内寄り − 出走馬全体の内寄り(多頭数は外の枠ほど馬が多いので、そのままだと外寄りに出る)
    inn = sum(pos(h) for h in t3) / max(len(t3), 1) - sum(pos(h) for h in hs) / len(hs)
    t3f = [front(h) for h in t3]
    t3f = [x for x in t3f if x is not None]
    if not fr_known or not t3f:
        return inn, None
    return inn, sum(t3f) / len(t3f) - sum(fr_known) / len(fr_known)


def agg(sums):
    ins = [s[0] for s in sums if s]
    frs = [s[1] for s in sums if s and s[1] is not None]
    return (sum(ins) / len(ins) if ins else None, sum(frs) / len(frs) if frs else None, len(ins))


day_sum = {}
for key, rcs in groups.items():
    rcs.sort(key=lambda r: r.get("rno") or 0)
    day_sum[key] = [summary(r) for r in rcs]
by_vs = defaultdict(list)
for (d, v, s) in groups:
    by_vs[(v, s)].append(d)
for k in by_vs:
    by_vs[k].sort()

out = {}
for (d, v, s), rcs in groups.items():
    # 前の開催日(3日以内)
    ds = by_vs[(v, s)]
    i = ds.index(d)
    prev = None
    if i > 0:
        pd = ds[i - 1]
        if (date(int(d[:4]), int(d[4:6]), int(d[6:])) - date(int(pd[:4]), int(pd[4:6]), int(pd[6:]))).days <= 3:
            prev = agg(day_sum[(pd, v, s)])
    for j, rc in enumerate(rcs):
        b_in, b_fr, b_n = agg(day_sum[(d, v, s)][:j])        # このレースより前だけ
        hs = [h for h in rc["horses"] if not h.get("scratched")]
        maxw = max(h.get("waku") or 1 for h in hs) if hs else 1
        for h in hs:
            f = F.get((rc["id"], h["umaban"]))
            if f is None:
                continue
            inner = 0.5 - ((h.get("waku") or 1) - 1) / max(maxw - 1, 1)
            st = f.get("style")
            fr = None if st is None else (1.0 if st <= 1 else 0.0) - 0.5
            x = {"bias_n": b_n, "bias_in": b_in, "bias_front": b_fr,
                 "fit_in": inner * b_in if b_in is not None else None,
                 "fit_front": fr * b_fr if (b_fr is not None and fr is not None) else None,
                 "prev_in": prev[0] if prev else None, "prev_front": prev[1] if prev else None,
                 "fit_prev_in": inner * prev[0] if prev and prev[0] is not None else None,
                 "fit_prev_front": fr * prev[1] if (prev and prev[1] is not None and fr is not None) else None}
            out[(rc["id"], h["umaban"])] = x
os.makedirs(SP, exist_ok=True)
pickle.dump(out, open(SP + "/feat.pkl", "wb"))
n = sum(1 for v in out.values() if v["bias_n"])
print("馬", len(out), "当日の前のレースあり", n, "前の日あり", sum(1 for v in out.values() if v["prev_in"] is not None))

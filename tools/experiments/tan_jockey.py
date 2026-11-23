"""◎の単勝を、鞍上(騎手)で分けると回収率は変わるか(中央 2022〜2026年。その年を除いて学んだモデル)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy python tools/experiments/tan_jockey.py
  モデルはもう騎手(それまでの3着内率・乗った数・レースの中の順位)と厩舎を材料にしている。ここでは、その上で
  騎手で分けたら市場とのずれが出るかを見る:
   1. 騎手のそれまでの3着内率(寄せたもの。jockey_top3)の帯  2. レースの中で騎手の成績が何番目か(jockey_top3_rank)
   3. 乗り替わり(同じ馬の前のレース〈手元のデータにあるもの〉と騎手が違う)。減量の印(▲△☆◇★)は落として比べる
   4. リーディング上位(その年より前の年の勝ち数で上位10人)か
  ◎ = レースで確率がいちばん高い馬(確率はレースの中で合計3にそろえたもの)。払戻は確定"""
import os, pickle, re, sys, time
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
YS = ("2022", "2023", "2024", "2025", "2026")
t0 = time.time()
races = T.load_races()
info = {rc["id"]: (d, rc) for d, rc in races}
jname = lambda s: re.sub(r"[▲△☆◇★\s]", "", s or "")
# 同じ馬の前のレースの騎手(日付の順に見ていく)・年ごとの勝ち数
prev_j, last_j = {}, {}
wins = defaultdict(Counter)
for d, rc in sorted(races, key=lambda x: (x[0], x[1]["id"])):
    hs = rc.get("horses") or []
    hs = hs if isinstance(hs, list) else list(hs.values())
    order = (rc.get("result") or {}).get("order") or []
    for h in hs:
        hid, j = h.get("horse_id"), jname(h.get("jockey"))
        if not hid or not j:
            continue
        prev_j[(rc["id"], h["umaban"])] = last_j.get(hid)
        last_j[hid] = j
        if order and order[0] == h["umaban"] and rc.get("center") == "jra":
            wins[d[:4]][j] += 1
top10 = {}
for y in YS:
    c = Counter()
    for yy, cc in wins.items():
        if yy < y:
            c.update(cc)
    top10[y] = {j for j, _ in c.most_common(10)}
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
p3 = np.load(os.environ["PRED"])
by = defaultdict(list)
for i, r in enumerate(rows):
    if not np.isnan(p3[i]) and r["f"].get("center") == 0:
        by[r["rid"]].append(i)
H = []
for rid, ix in by.items():
    if rid not in info:
        continue
    d, rc = info[rid]
    res = rc["result"]; order = res.get("order") or []; pay = res.get("payout") or {}
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    if len(order) < 3 or len(od) < 5:
        continue
    tot = p3[ix].sum(); want = min(3, len(ix))
    ps = sorted(((min(0.99, p3[i] * want / tot), i) for i in ix), reverse=True)
    p, i = ps[0]
    u = rows[i]["u"]
    if u not in od or len(ps) < 2:
        continue
    hs = rc.get("horses") or []
    hs = hs if isinstance(hs, list) else list(hs.values())
    hh = next((x for x in hs if x["umaban"] == u), {})
    j, pj = jname(hh.get("jockey")), prev_j.get((rid, u))
    f = rows[i]["f"]; y = d[:4]
    tan = (pay.get("単勝") or {}).get(str(u))
    H.append(dict(y=y, p=p, gap=p - ps[1][0], odds=od[u], nk=sorted(od, key=lambda x: (od[x], x)).index(u) + 1,
                  tan=(tan if tan is not None else od[u] * 100) if order[0] == u else 0,
                  jt=f.get("jockey_top3"), jr=f.get("jockey_top3_rank"), chg=None if not pj or not j else pj != j, top=j in top10.get(y, set())))
print("中央 ◎ {}頭 ({:.0f}分)。乗り替わりが分かる {:.0%}".format(len(H), (time.time() - t0) / 60, np.mean([x["chg"] is not None for x in H])), flush=True)


def show(name, sel):
    if len(sel) < 30:
        print("  {:30s} {:5d}頭(少ない)".format(name, len(sel))); return
    pay = np.array([x["tan"] for x in sel], float); y = np.array([x["y"] for x in sel])
    top = np.sort(pay)[-3:].sum()
    print("  {:30s} {:5d}頭 1着 {:5.1%} オッズ {:4.1f}倍 回収率 {:5.1f}%(大きい3つを除くと {:5.1f}%) [{}]".format(
        name, len(sel), np.mean(pay > 0), np.median([x["odds"] for x in sel]), pay.mean(), (pay.sum() - top) / len(sel),
        " ".join("{:3.0f}".format(pay[y == yy].mean()) if (y == yy).any() else "  -" for yy in YS)))


def split(title, sel):
    print("\n■ " + title + "  [ ] は 2022〜2026年の回収率(オッズは中央値)")
    show("全部", sel)
    qs = np.quantile([x["jt"] for x in sel if x["jt"] is not None], [1 / 3, 2 / 3])
    show("騎手の3着内率 上1/3(≥{:.0%})".format(qs[1]), [x for x in sel if x["jt"] is not None and x["jt"] >= qs[1]])
    show("騎手の3着内率 中1/3", [x for x in sel if x["jt"] is not None and qs[0] <= x["jt"] < qs[1]])
    show("騎手の3着内率 下1/3(<{:.0%})".format(qs[0]), [x for x in sel if x["jt"] is not None and x["jt"] < qs[0]])
    show("レースでいちばん成績の良い騎手", [x for x in sel if x["jr"] == 1])
    show("レースで2〜3番目", [x for x in sel if x["jr"] in (2, 3)])
    show("レースで4番目より下", [x for x in sel if x["jr"] is not None and x["jr"] >= 4])
    show("リーディング上位10人", [x for x in sel if x["top"]])
    show("それ以外", [x for x in sel if not x["top"]])
    show("乗り替わり", [x for x in sel if x["chg"] is True])
    show("同じ騎手(継続騎乗)", [x for x in sel if x["chg"] is False])


split("単勝 ◎(確率80%以上)を、鞍上で分ける", [x for x in H if x["p"] >= 0.80])
split("単勝 ◎(全部)を、鞍上で分ける", H)
print("\nかかった時間 {:.0f}分".format((time.time() - t0) / 60))

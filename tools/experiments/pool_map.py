"""馬券の種類ごとに、人気の順の組み合わせで買い続けたときの回収率の地図(中央 2022〜2026年の全レース。払戻は確定)
  python tools/experiments/pool_map.py
  「1・2・3番人気の3連複」のように、人気の順だけで決めた組を全レースで買うといくら戻るか。売れすぎの組は控除率より低く、
  売れていない組は高く出る。馬の情報は使わないので、ほかの人が買い目をどう偏らせているかだけが見える。
  年ごと(5年)にそろうかも出す"""
import glob, json, os, sys
from collections import defaultdict
from itertools import combinations, permutations
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YS = ("2022", "2023", "2024", "2025", "2026")


def load():
    out = []
    for f in sorted(glob.glob(R + "/keiba/days/20*.json")):
        d = os.path.basename(f)[:8]
        for rid, r in (json.load(open(f)).get("races") or {}).items():
            res = r.get("result") or {}
            if r.get("center") != "jra" or not res.get("payout") or len(res.get("order") or []) < 3 or r.get("surface") not in ("芝", "ダ"):
                continue
            nk = {int(u): k for u, k in (res.get("ninki") or {}).items() if k}
            od = {int(u): o for u, o in (res.get("odds") or {}).items() if o}
            if len(nk) < 5:
                continue
            out.append(dict(d=d, y=d[:4], rid=rid, n=len(nk), nk=nk, od=od, order=res["order"], pay=res["payout"], r=r))
    return out


def by_rank(rc):
    """人気の順 -> 馬番(同じ人気が2頭いるときは馬番の小さい方を先に)"""
    return [u for u, _ in sorted(rc["nk"].items(), key=lambda x: (x[1], x[0]))]


if __name__ == "__main__":
    races = load()
    print("中央 {}R".format(len(races)))
    srt = lambda c: "-".join(str(z) for z in sorted(c))

    def table(title, typ, k, top, ordered=False, min_n=0):
        print("\n■ {}(人気の順の組を全レースで100円ずつ)  回収率 全体 [{}] 的中".format(title, " ".join(YS)))
        gen = permutations if ordered else combinations
        res = []
        for pat in gen(range(top), k):
            bet = defaultdict(int); ret = defaultdict(int); hit = 0
            for rc in races:
                if rc["n"] < max(top, min_n):
                    continue
                us = by_rank(rc)
                key = "-".join(str(us[i]) for i in pat) if ordered else srt(us[i] for i in pat)
                p = (rc["pay"].get(typ) or {}).get(key, 0)
                bet[rc["y"]] += 100; ret[rc["y"]] += p; hit += p > 0
            tb, tr = sum(bet.values()), sum(ret.values())
            res.append((tr / tb, pat, [ret[y] / bet[y] for y in YS], hit / (tb / 100), tb // 100))
        for roi, pat, ys, h, nn in sorted(res, key=lambda x: -x[0]):
            print("  {:10s} {:4.0f}% [{}] {:5.1%} ({}R)".format(("→" if ordered else "-").join(str(i + 1) for i in pat), roi * 100,
                  " ".join("{:4.0f}".format(v * 100) for v in ys), h, nn))
        return res

    table("ワイド", "ワイド", 2, 7)
    table("馬連", "馬連", 2, 7)
    table("馬単", "馬単", 2, 5, ordered=True)
    t3 = table("3連複", "3連複", 3, 7)
    # まとめ: 1番人気を含む/含まない、人気の合計
    print("\n■ 3連複 まとめ(7番人気までの35組)")
    for name, cond in (("1番人気を含む", lambda p: 0 in p), ("1番人気を含まない", lambda p: 0 not in p),
                       ("1・2番人気を両方含む", lambda p: 0 in p and 1 in p), ("1番人気を含み2番人気を含まない", lambda p: 0 in p and 1 not in p),
                       ("2番人気を含み1番人気を含まない", lambda p: 0 not in p and 1 in p), ("1・2番人気とも含まない", lambda p: 0 not in p and 1 not in p)):
        sel = [x for x in t3 if cond(x[1])]
        print("  {:20s} {}組 回収率 {:4.0f}% [{}]".format(name, len(sel), np.mean([x[0] for x in sel]) * 100,
              " ".join("{:4.0f}".format(np.mean([x[2][i] for x in sel]) * 100) for i in range(5))))
    table("3連単(4番人気まで)", "3連単", 3, 4, ordered=True)

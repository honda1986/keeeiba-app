"""ワイドのオッズと単勝のオッズの食い違いは、ほかの馬券で使えるか(中央 2022〜2026年。確定オッズ・確定の払戻)
  python tools/experiments/cross_pool.py
  ワイドの全部の組の確定オッズ(wide_final)から、各馬の「3着内に来る強さ」を出す(その馬を含む組の 1/オッズ の合計)。
  単勝の人気の順と、ワイドから出した順がずれる馬・レースで、単勝・馬連・3連複・3連単の回収率が変わるかを見る。
  変わるなら、馬券の種類によって値段に入っている情報が違う(片方を見てもう片方を買える)"""
import os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_map as PM
YS = PM.YS
srt = lambda c: "-".join(str(z) for z in sorted(c))


def wide_strength(rc):
    """馬番 -> ワイドから出した3着内の確率(合計3)。組が大きく欠けている馬(取消など)がいるレースは出さない"""
    wf = rc["r"].get("wide_final") or {}
    s = defaultdict(float); c = defaultdict(int)
    for k, v in wf.items():
        a, b = (int(x) for x in k.split("-"))
        o = (v[0] + v[1]) / 2 if isinstance(v, (list, tuple)) else v
        if not o or o <= 1.0:
            continue
        s[a] += 1 / o; s[b] += 1 / o; c[a] += 1; c[b] += 1
    # 大きい頭数では、売れていない組(人気薄どうし)が数組欠けている。強さへの影響はわずかなので、8割の組があれば使う
    us = [u for u in rc["nk"] if c[u] >= 0.8 * (len(rc["nk"]) - 1)]
    if len(us) < len(rc["nk"]) or len(us) < 5:
        return None
    tot = sum(s[u] for u in us)
    return {u: 3 * s[u] / tot for u in us}


def roi_line(name, rows):
    """rows = [(年, 払戻)]"""
    if not rows:
        return
    by = defaultdict(list)
    for y, p in rows:
        by[y].append(p)
    allp = [p for _, p in rows]
    print("  {:44s} {:6d}点 {:4.0f}% [{}] 的中 {:5.1%}".format(name, len(allp), np.mean(allp), " ".join(
        "{:4.0f}".format(np.mean(by[y])) if by[y] else "   -" for y in YS), np.mean([p > 0 for p in allp])))


if __name__ == "__main__":
    races = PM.load()
    data = []
    for rc in races:
        w = wide_strength(rc)
        if w:
            rc["w"] = w
            rc["rw"] = [u for u, _ in sorted(w.items(), key=lambda x: -x[1])]      # ワイドの強さの順
            rc["rn"] = PM.by_rank(rc)                                               # 単勝の人気の順
            data.append(rc)
    print("中央 {}R(ワイドの確定オッズがある)".format(len(data)))

    # 1) 単勝: ワイドの順位が単勝の人気より上の馬・下の馬
    print("\n■ 単勝を買う: ワイドから出した順位と単勝の人気のずれで分ける(単勝オッズの帯ごと)")
    for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 30), (30, 1000)):
        for name, f in (("ワイドの方が2つ以上 上", lambda g: g >= 2), ("1つ上", lambda g: g == 1), ("同じ", lambda g: g == 0),
                        ("1つ下", lambda g: g == -1), ("ワイドの方が2つ以上 下", lambda g: g <= -2)):
            rows = []
            for rc in data:
                win = rc["order"][0]
                for i, u in enumerate(rc["rn"]):
                    o = rc["od"].get(u)
                    if o and lo <= o < hi and f(i - rc["rw"].index(u)):
                        rows.append((rc["y"], o * 100 if u == win else 0))
            roi_line("単勝 {}〜{}倍 {}".format(lo, hi, name), rows)

    # 2) 3着内: ずれた馬は実際どちらの見立てに近いか
    print("\n■ 3着内に来た割合: 単勝の人気ごとに、ワイドの順位が上の馬・同じ馬・下の馬")
    for i in range(8):
        out = []
        for name, f in (("上", lambda g: g >= 1), ("同じ", lambda g: g == 0), ("下", lambda g: g <= -1)):
            v = [rc["rn"][i] in rc["order"][:3] for rc in data if len(rc["rn"]) > i and f(i - rc["rw"].index(rc["rn"][i]))]
            out.append("{} {:4.1%}({})".format(name, np.mean(v) if v else 0, len(v)))
        print("  {}番人気  {}".format(i + 1, "  ".join(out)))

    # 3) 組の馬券: 単勝の人気の上位とワイドの順の上位が違うレース
    print("\n■ 上位の顔ぶれが違うレースで、どちらの上位を買うか")
    for typ, k, ordered in (("馬連", 2, False), ("ワイド", 2, False), ("3連複", 3, False), ("馬単", 2, True), ("3連単", 3, True)):
        a, b, c = [], [], []
        for rc in data:
            tn, tw = rc["rn"][:k], rc["rw"][:k]
            key = (lambda us: "-".join(map(str, us)) if ordered else srt(us))
            pn = (rc["pay"].get(typ) or {}).get(key(tn), 0); pw = (rc["pay"].get(typ) or {}).get(key(tw), 0)
            if (tn != tw) if ordered else (set(tn) != set(tw)):
                a.append((rc["y"], pn)); b.append((rc["y"], pw))
            else:
                c.append((rc["y"], pn))
        roi_line("{} 同じレース(上位{}頭が同じ)".format(typ, k), c)
        roi_line("{} 違うレースで 単勝の人気の上位".format(typ), a)
        roi_line("{} 違うレースで ワイドの順の上位".format(typ), b)

    # 4) 3連複: 1・2番人気は同じで、3頭目だけ違う(今の買い方の▲に近い形)
    print("\n■ 3連複: 単勝の1・2番人気 + 3頭目(3番人気 / ワイドの順で残りの1番目)")
    a, b, c = [], [], []
    for rc in data:
        f1, f2, f3 = rc["rn"][:3]
        w3 = next(u for u in rc["rw"] if u not in (f1, f2))
        pay = rc["pay"].get("3連複") or {}
        if w3 != f3:
            a.append((rc["y"], pay.get(srt((f1, f2, f3)), 0))); b.append((rc["y"], pay.get(srt((f1, f2, w3)), 0)))
        else:
            c.append((rc["y"], pay.get(srt((f1, f2, f3)), 0)))
    roi_line("3頭目が同じレース 1-2-3番人気", c)
    roi_line("3頭目が違うレースで 3番人気", a)
    roi_line("3頭目が違うレースで ワイドの順の3頭目", b)

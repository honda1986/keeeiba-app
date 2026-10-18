#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""買い方を変えたら成績がどうなったかを、保存してある予想と払戻表から計算する(取得はしない)。
   python tools/simulate.py [keiba/days]
規則 30(条件ごとの回収率を見て買い方を絞る)のための道具。
  1 … ◎○がそろって人気すぎるレースを見送る(人気・オッズは、再現では確定人気)
  2 … 買い目をワイドに寄せ、3連複・馬連を減らす"""
import glob
import json
import os
import sys
from collections import defaultdict

UNIT = 100


def key(typ, c):
    return "-".join(str(x) for x in (c if typ == "3連単" else sorted(c)))


def ret_of(bets, pay):
    """bets: [(券種, 組, 金額)] -> (投資, 払戻, 的中したか)"""
    stake = ret = 0
    hit = False
    for typ, c, unit in bets:
        stake += unit
        y = (pay.get(typ) or {}).get(key(typ, c))
        if y:
            ret += y * unit // 100
            hit = True
    return stake, ret, hit


# ---------------------------------------------------------------- 買い方
def bets_now(p):
    """今の買い方(keiba_logic.make_bets と同じもの)"""
    return [(b["type"], c, b["unit"]) for b in p["bets"] for c in b["combos"]]


def aite3(p):
    return [x for x in p.get("aite", []) if x not in (p["honmei"], p["taikou"])]


def bets_wide(p):
    """2a ワイドだけ: ◎-○を厚く、◎-相手・○-相手の上位2頭ずつ"""
    h, t, a = p["honmei"], p["taikou"], aite3(p)[:2]
    return ([("ワイド", [h, t], UNIT * 5)] + [("ワイド", [h, x], UNIT) for x in a]
            + [("ワイド", [t, x], UNIT) for x in a])


def bets_wide_tan3(p):
    """2b ワイド中心+3連単: 3連複・馬連をやめ、ワイド◎-○を厚く"""
    out = [(typ, c, u) for typ, c, u in bets_now(p) if typ in ("3連単",) or (typ == "ワイド" and set(c) != {p["honmei"], p["taikou"]})]
    return [("ワイド", [p["honmei"], p["taikou"]], UNIT * 5)] + out


MIXES = {"今の買い方": bets_now, "2a ワイドだけ": bets_wide, "2b ワイド中心+3連単": bets_wide_tan3}


# ---------------------------------------------------------------- 見送り
def odds_of(rc, u):
    for h in rc.get("horses", []):
        if h["umaban"] == u:
            return h.get("odds"), h.get("ninki")
    return None, None


def skip_none(rc, p):
    return False


def skip_top2(rc, p):
    """1a ◎○が1・2番人気そろい"""
    n1, n2 = odds_of(rc, p["honmei"])[1], odds_of(rc, p["taikou"])[1]
    return bool(n1 and n2 and {n1, n2} == {1, 2})


def skip_prod(limit):
    def f(rc, p):
        o1, o2 = odds_of(rc, p["honmei"])[0], odds_of(rc, p["taikou"])[0]
        return bool(o1 and o2 and o1 * o2 < limit)
    f.__doc__ = "1{} ◎と○の単勝オッズの積 < {}".format("b" if limit == 10 else "c", limit)
    return f


SKIPS = {"見送りなし": skip_none, "1a ◎○が1・2番人気なら見送り": skip_top2,
         "1b オッズの積<10なら見送り": skip_prod(10), "1c オッズの積<20なら見送り": skip_prod(20)}


# ---------------------------------------------------------------- 集計
def load(days_dir):
    races = []
    for f in sorted(glob.glob(os.path.join(days_dir, "*.json"))):
        day = json.load(open(f, encoding="utf-8"))
        for rc in day["races"].values():
            p = rc.get("pred") or {}
            res = rc.get("result") or {}
            if p.get("verdict") in ("A", "B") and res.get("payout") and len(res.get("order", [])) >= 3:
                races.append((day["date"], rc, p))
    return races


def run(races, skip, mix, by=None):
    out = defaultdict(lambda: [0, 0, 0, 0])
    for d, rc, p in races:
        if skip(rc, p):
            continue
        st, rt, hit = ret_of(mix(p), rc["result"]["payout"])
        for k in ("全体",) + ((by(d, rc, p),) if by else ()):
            x = out[k]
            x[0] += 1
            x[1] += hit
            x[2] += st
            x[3] += rt
    return out


def fmt(x):
    n, hit, st, rt = x
    if not n:
        return "-"
    return "{:>4}R 的中{:>3.0f}% 投資{:>8,} 払戻{:>8,} 収支{:>+9,} 回収率{:>4.0f}%".format(
        n, hit / n * 100, st, rt, rt - st, rt / st * 100 if st else 0)


def main():
    days_dir = sys.argv[1] if len(sys.argv) > 1 else "keiba/days"
    races = load(days_dir)
    dates = sorted({d for d, _, _ in races})
    print("買いレース {} / {}〜{}".format(len(races), dates[0] if dates else "", dates[-1] if dates else ""))
    print("\n■ 組み合わせ(見送り × 買い方)")
    for sn, sf in SKIPS.items():
        for mn, mf in MIXES.items():
            print("  {:<22} × {:<14} {}".format(sn, mn, fmt(run(races, sf, mf)["全体"])))
    best = [("見送りなし", "今の買い方")] + [(s, m) for s in SKIPS for m in MIXES if (s, m) != ("見送りなし", "今の買い方")]
    print("\n■ 月別・中央/地方(安定しているかを見る)")
    for sn, mn in best:
        res = run(races, SKIPS[sn], MIXES[mn], by=lambda d, rc, p: d[:6])
        res2 = run(races, SKIPS[sn], MIXES[mn], by=lambda d, rc, p: "中央" if rc.get("center") == "jra" else "地方")
        months = [k for k in sorted(res) if k != "全体"]
        plus = sum(1 for k in months if res[k][3] >= res[k][2])
        print("  {} × {}: 月で黒字 {}/{} | 中央 {} | 地方 {}".format(
            sn, mn, plus, len(months), fmt(res2["中央"]).split("収支")[1] if res2["中央"][0] else "-",
            fmt(res2["地方"]).split("収支")[1] if res2["地方"][0] else "-"))


if __name__ == "__main__":
    main()

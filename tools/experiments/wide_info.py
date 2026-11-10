"""ワイドのオッズに出ている「3着内の強さ」を、今の買い方(3連複◎-○-▲)に重ねる(中央 2024〜2026年。月ごとの確率の予想・確定オッズ)
  python tools/experiments/wide_info.py
  cross_pool.py で、3連複の3頭目は単勝の3番人気より「ワイドのオッズで強い馬」の方が 72% → 86% と良かった。
  ここでは ◎○ はモデルのまま、▲をモデル/ワイド/単勝の人気で選んだとき、モデルとワイドが合うとき・合わないとき、
  ◎や○がワイドでは弱いときを、買うレースと ◎の確率の帯で分けて出す"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_pool as CP
R = os.path.join(HERE, "..", "..")
YS = ("2024", "2025", "2026")
srt = lambda c: "-".join(str(z) for z in sorted(c))


def load():
    out = []
    for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
        d = os.path.basename(f)[:8]
        for rid, r in sorted((json.load(open(f)).get("races") or {}).items(), key=lambda x: x[1].get("start") or ""):
            p, res = r.get("pred") or {}, r.get("result") or {}
            if r.get("center") != "jra" or not p.get("model") or not p.get("honmei") or not p.get("taikou") or len(res.get("order") or []) < 3:
                continue
            nk = {int(u): k for u, k in (res.get("ninki") or {}).items() if k}
            rc = dict(d=d, y=d[:4], rid=rid, nk=nk, od={int(u): o for u, o in (res.get("odds") or {}).items() if o}, order=res["order"],
                      pay=res.get("payout") or {}, r=r)
            w = CP.wide_strength(rc) if len(nk) >= 5 else None
            if not w:
                continue
            hs = sorted(p["horses"], key=lambda h: -(h.get("p3") or 0))
            pm = {h["umaban"]: h.get("p3") or 0 for h in hs}
            h_, t_ = p["honmei"], p["taikou"]
            rest = [x["umaban"] for x in hs if x["umaban"] not in (h_, t_) and x["umaban"] in w]
            if h_ not in w or t_ not in w or len(rest) < 2:
                continue
            rw = [u for u, _ in sorted(w.items(), key=lambda x: -x[1])]
            rn = [u for u, _ in sorted(nk.items(), key=lambda x: (x[1], x[0])) if u in w]
            bet = (p.get("bets") or [{}])[0]
            out.append(dict(rc, h=h_, t=t_, pm=pm, w=w, rw=rw, rn=rn, rest=rest, buy=p.get("verdict") in ("A", "B"),
                            bet_wide=bet.get("type") == "ワイド", p1=pm.get(h_, 0), p2=pm.get(t_, 0),
                            am=rest[0], aw=next(u for u in rw if u not in (h_, t_)), an=next(u for u in rn if u not in (h_, t_))))
    return out


def trio(rc, x):
    return rc["pay"].get("3連複", {}).get(srt((rc["h"], rc["t"], x)), 0)


def now_pay(rc, x=None):
    """今の買い方の払戻(▲が延長ならワイド◎-○)。x を渡すと3連複の3頭目だけ替える(ワイドのレースはそのまま)"""
    if rc["bet_wide"]:
        return rc["pay"].get("ワイド", {}).get(srt((rc["h"], rc["t"])), 0)
    return trio(rc, rc["am"] if x is None else x)


def line(name, rows):
    if not rows:
        print("  {:46s}   0R".format(name)); return
    by = defaultdict(list)
    for y, p in rows:
        by[y].append(p)
    v = [p for _, p in rows]
    print("  {:46s} {:5d}R {:4.0f}% [{}] 的中 {:4.0%}".format(name, len(v), np.mean(v), " ".join(
        "{:4.0f}".format(np.mean(by[y])) if by[y] else "   -" for y in YS), np.mean([p > 0 for p in v])))


if __name__ == "__main__":
    D = load()
    B = [rc for rc in D if rc["buy"]]
    print("予想とワイドの確定オッズがそろうレース {}R / うち買うレース {}R".format(len(D), len(B)))

    for title, S in (("買うレース(◎80%以上・差15%以上)", B), ("◎70%以上の全部", [rc for rc in D if rc["p1"] >= .70]),
                     ("◎60%以上の全部", [rc for rc in D if rc["p1"] >= .60]), ("全レース", D)):
        print("\n■ {}: 3連複◎-○-x の3頭目の選び方".format(title))
        line("モデルの▲(確率で◎○の次)", [(rc["y"], trio(rc, rc["am"])) for rc in S])
        line("ワイドのオッズで◎○の次に強い馬", [(rc["y"], trio(rc, rc["aw"])) for rc in S])
        line("単勝の人気で◎○の次", [(rc["y"], trio(rc, rc["an"])) for rc in S])
        same = [rc for rc in S if rc["am"] == rc["aw"]]; diff = [rc for rc in S if rc["am"] != rc["aw"]]
        line("モデルとワイドが同じ馬", [(rc["y"], trio(rc, rc["am"])) for rc in same])
        line("違うレースで モデルの▲", [(rc["y"], trio(rc, rc["am"])) for rc in diff])
        line("違うレースで ワイドの馬", [(rc["y"], trio(rc, rc["aw"])) for rc in diff])
        line("違うレースで 両方(2点。1点あたり)", [(rc["y"], (trio(rc, rc["am"]) + trio(rc, rc["aw"])) / 2) for rc in diff])
        # モデルの▲がワイドの順で何番目か(◎○を除いて)
        for k, nm in ((0, "1番目"), (1, "2番目"), (2, "3番目"), (3, "4番目以下")):
            sel = [rc for rc in S if (lambda i: i == k if k < 3 else i >= 3)([u for u in rc["rw"] if u not in (rc["h"], rc["t"])].index(rc["am"]))]
            line("モデルの▲がワイドの順で{}".format(nm), [(rc["y"], trio(rc, rc["am"])) for rc in sel])

    print("\n■ 買うレース: 今の買い方(▲が延長ならワイド)を、ワイドのオッズの見え方で分ける")
    line("今の買い方 全部", [(rc["y"], now_pay(rc)) for rc in B])
    wi = lambda rc, u: rc["rw"].index(u)
    for nm, f in (("◎がワイドで1番強い", lambda rc: wi(rc, rc["h"]) == 0), ("◎がワイドで2番目以下", lambda rc: wi(rc, rc["h"]) >= 1),
                  ("○がワイドで3番目まで", lambda rc: wi(rc, rc["t"]) <= 2), ("○がワイドで4番目以下", lambda rc: wi(rc, rc["t"]) >= 3),
                  ("◎○がワイドの上位2頭", lambda rc: {wi(rc, rc["h"]), wi(rc, rc["t"])} == {0, 1}),
                  ("◎○▲がワイドの上位3頭", lambda rc: {wi(rc, rc["h"]), wi(rc, rc["t"]), wi(rc, rc["am"])} == {0, 1, 2}),
                  ("◎○▲がワイドの上位3頭でない", lambda rc: {wi(rc, rc["h"]), wi(rc, rc["t"]), wi(rc, rc["am"])} != {0, 1, 2}),
                  ("◎○▲が単勝の1〜3番人気", lambda rc: {rc["rn"].index(rc["h"]), rc["rn"].index(rc["t"]), rc["rn"].index(rc["am"])} == {0, 1, 2}),
                  ("◎○▲が単勝の1〜3番人気でない", lambda rc: {rc["rn"].index(rc["h"]), rc["rn"].index(rc["t"]), rc["rn"].index(rc["am"])} != {0, 1, 2})):
        line(nm, [(rc["y"], now_pay(rc)) for rc in B if f(rc)])
    print("  3連複のレースだけ(▲が延長でない)")
    T3 = [rc for rc in B if not rc["bet_wide"]]
    line("  モデルの▲", [(rc["y"], trio(rc, rc["am"])) for rc in T3])
    line("  ワイドの馬に替える", [(rc["y"], trio(rc, rc["aw"])) for rc in T3])
    line("  同じ馬のレースだけ買う", [(rc["y"], trio(rc, rc["am"])) for rc in T3 if rc["am"] == rc["aw"]])
    line("  違うレース モデルの▲", [(rc["y"], trio(rc, rc["am"])) for rc in T3 if rc["am"] != rc["aw"]])
    line("  違うレース ワイドの馬", [(rc["y"], trio(rc, rc["aw"])) for rc in T3 if rc["am"] != rc["aw"]])

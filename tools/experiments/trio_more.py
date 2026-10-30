"""買うレース(中央・◎80%以上)を、レースの条件・◎○▲それぞれの性質・前走・人気とのずれ・暦で分けて、
今の買い方(3連複◎○▲、▲が距離延長ならワイド◎-○)の回収率と的中を年ごとに出す。2024年1月〜。予想はアプリと同じ
  python tools/experiments/trio_more.py
  分け方ごとに [2024 2025 2026] の回収率を出す。3年そろって 100% を大きく上下するものだけが傾向の候補"""
import glob, json, math, os, re, sys
from collections import defaultdict, Counter
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, R)
import keiba as K
YS = ("2024", "2025", "2026")
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    day = None
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or len(res.get("order") or []) < 3 or not res.get("odds"):
            continue
        if p.get("verdict") not in ("A", "B"):
            continue
        hs = {h["umaban"]: h for h in p["horses"] if h.get("p3") is not None}
        mk = {h.get("mark"): u for u, h in hs.items() if h.get("mark") in ("◎", "○", "▲", "△")}
        od = {int(u): v for u, v in res["odds"].items() if v}
        if not all(m in mk and mk[m] in od for m in "◎○▲△") or hs[mk["◎"]]["p3"] < .8:
            continue
        if day is None:
            day = K.load_day(d)                       # 馬柱(前走)は別ファイル
        card = {h["umaban"]: h for h in day["races"][rid].get("horses") or []}
        races.append(dict(d=d, y=d[:4], m=int(d[4:6]), rid=rid, r=r, p=p, hs=hs, card=card, od=od, pay=res.get("payout") or {},
                          top3=set(res["order"][:3]), h=mk["◎"], t=mk["○"], a=mk["▲"], b=mk["△"]))
print("◎80%以上の買うレース", len(races))


def key3(us):
    return "-".join(str(x) for x in sorted(us))


def wide(r):
    return (r["pay"].get("ワイド") or {}).get(key3((r["h"], r["t"])), 0)


def trio(r):
    return (r["pay"].get("3連複") or {}).get(key3((r["h"], r["t"], r["a"])), 0)


def now(r):
    return wide(r) if "距離延長" in (r["hs"][r["a"]].get("axis_ng") or []) else trio(r)


def yr(xs, g):
    return " ".join("{:4.0f}".format(sum(g(r) for r in xs if r["y"] == y) / max(1, sum(1 for r in xs if r["y"] == y)))
                    if sum(1 for r in xs if r["y"] == y) >= 8 else "   -" for y in YS)


FLAG = []


def show(title, key, order=None, min_n=25):
    g = defaultdict(list)
    for r in races:
        k = key(r)
        if k is not None:
            g[k].append(r)
    print("\n  [{}]".format(title))
    for k in order or sorted(g, key=lambda k: -len(g[k])):
        s = g.get(k, [])
        if len(s) < min_n:
            continue
        roi = sum(map(now, s)) / len(s); hit = sum(1 for r in s if now(r)) / len(s)
        ys = [sum(now(r) for r in s if r["y"] == y) / max(1, sum(1 for r in s if r["y"] == y)) for y in YS]
        ok = [sum(1 for r in s if r["y"] == y) >= 8 for y in YS]
        mark = ""
        if all(ok) and all(v >= 115 for v in ys):
            mark = " ◀ 3年とも高い"
        elif all(ok) and all(v <= 90 for v in ys):
            mark = " ◀ 3年とも低い"
        if mark:
            FLAG.append("{}: {} {}R 今 {:.0f}% [{}]".format(title, k, len(s), roi, " ".join("%.0f" % v for v in ys)))
        print("    {:24s} {:4d}R 今 {:4.0f}% [{}] 的中{:3.0%} ‖ 3連複 {:4.0f}% ワイド {:4.0f}%{}".format(
            str(k)[:24], len(s), roi, yr(s, now), hit, sum(map(trio, s)) / len(s), sum(map(wide, s)) / len(s), mark))


def band(v, cuts, fmt="{:g}"):
    if v is None:
        return None
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if lo <= v < hi:
            return (fmt + "〜" + fmt).format(lo, hi) if hi < 9e8 else (fmt + "〜").format(lo)
    return None


def last(r, u):
    ps = (r["card"].get(u) or {}).get("past") or []
    return ps[0] if ps else {}


# 騎手の格: 2024〜2026年の中央で勝ち数の多い順(このデータの中だけ。あくまで目安)
wins = Counter()
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    for r in (json.load(open(f)).get("races") or {}).values():
        o = (r.get("result") or {}).get("order") or []
        if r.get("center") == "jra" and o:
            j = next((h.get("jockey") for h in r.get("horses") or [] if h["umaban"] == o[0]), None)
            if j:
                wins[re.sub(r"^[▲△☆◇★]", "", j)] += 1
top = [j for j, _ in wins.most_common(10)]
jk = lambda r, u: "上位10人" if re.sub(r"^[▲△☆◇★]", "", (r["card"].get(u) or {}).get("jockey") or "") in top else "ほか"
print("騎手の上位10人(勝ち数):", " ".join(top))

print("\n■ ぜんぶ  {}R 今 {:.0f}% [{}] 的中 {:.0%}".format(len(races), sum(map(now, races)) / len(races), yr(races, now),
                                                    sum(1 for r in races if now(r)) / len(races)))
print("\n■ レースの条件")
CLS = {0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝"}
show("クラス", lambda r: CLS.get(r["r"].get("cls"), "OP・重賞"))
show("年齢", lambda r: "2歳" if "2歳" in (r["r"].get("cond") or "") else "3歳" if re.search(r"3歳(?!以上)", r["r"].get("cond") or "") else "古馬(3歳以上・4歳以上)")
show("牝馬限定", lambda r: "牝馬限定" if "牝" in (r["r"].get("cond") or "") else "混合")
show("馬場", lambda r: r["r"].get("baba") or "?", ["良", "稍", "重", "不"])
show("天気", lambda r: r["r"].get("tenko") or "?")
show("競馬場", lambda r: r["r"].get("venue"))
show("レース番号", lambda r: "1〜4R" if r["r"]["rno"] <= 4 else "5〜8R" if r["r"]["rno"] <= 8 else "9〜12R", ["1〜4R", "5〜8R", "9〜12R"])
show("頭数", lambda r: band(r["r"].get("n") or len(r["hs"]), [0, 9, 12, 15, 99], "{:g}頭"))
show("季節", lambda r: "冬(12〜2月)" if r["m"] in (12, 1, 2) else "春(3〜5月)" if r["m"] <= 5 else "夏(6〜8月)" if r["m"] <= 8 else "秋(9〜11月)",
     ["冬(12〜2月)", "春(3〜5月)", "夏(6〜8月)", "秋(9〜11月)"])
show("曜日", lambda r: "日月火水木金土"[__import__("datetime").date(int(r["d"][:4]), int(r["d"][4:6]), int(r["d"][6:])).isoweekday() % 7])

print("\n■ ◎")
show("◎の確率", lambda r: band(r["hs"][r["h"]]["p3"], [.8, .85, .9, 2], "{:.0%}"))
show("◎の人気", lambda r: "1番人気" if r["hs"][r["h"]].get("ninki") == 1 else "2番人気" if r["hs"][r["h"]].get("ninki") == 2 else "3番人気以下")
show("◎の単勝オッズ", lambda r: band(r["od"][r["h"]], [1, 1.5, 2, 3, 999], "{:g}倍"))
show("◎の騎手", lambda r: jk(r, r["h"]))
show("◎の前走着順", lambda r: band(last(r, r["h"]).get("rank"), [1, 2, 4, 99], "{:g}着") if last(r, r["h"]) else "前走なし")
show("◎の前走人気", lambda r: band(last(r, r["h"]).get("ninki"), [1, 2, 4, 99], "{:g}番人気") if last(r, r["h"]) else "前走なし")
show("◎の性別", lambda r: ((r["card"].get(r["h"]) or {}).get("sex_age") or "?")[0])
show("◎の休み", lambda r: band(r["hs"][r["h"]].get("rest_days"), [0, 22, 50, 100, 9999], "{:g}日"))

print("\n■ ○")
show("○の確率", lambda r: band(r["hs"][r["t"]]["p3"], [0, .4, .5, .6, .7, 2], "{:.0%}"))
show("○の人気", lambda r: "1番人気" if r["hs"][r["t"]].get("ninki") == 1 else "2番人気" if r["hs"][r["t"]].get("ninki") == 2 else "3番人気" if r["hs"][r["t"]].get("ninki") == 3 else "4番人気以下")
show("◎と○の確率の差", lambda r: band(r["hs"][r["h"]]["p3"] - r["hs"][r["t"]]["p3"], [0, .2, .3, .4, 2], "{:.0%}"))
show("○の騎手", lambda r: jk(r, r["t"]))
show("○の前走着順", lambda r: band(last(r, r["t"]).get("rank"), [1, 2, 4, 99], "{:g}着") if last(r, r["t"]) else "前走なし")

print("\n■ ▲")
show("▲の騎手", lambda r: jk(r, r["a"]))
show("▲の前走着順", lambda r: band(last(r, r["a"]).get("rank"), [1, 2, 4, 6, 99], "{:g}着") if last(r, r["a"]) else "前走なし")
show("▲の前走人気", lambda r: band(last(r, r["a"]).get("ninki"), [1, 2, 4, 7, 99], "{:g}番人気") if last(r, r["a"]) else "前走なし")
show("▲の前走からの距離", lambda r: band((r["r"].get("dist") or 0) - (last(r, r["a"]).get("dist") or 0), [-9999, -1, 1, 9999], "{:g}m")
     if last(r, r["a"]) else None)
show("▲の前走の着差", lambda r: band(last(r, r["a"]).get("margin"), [-99, 0.01, 0.5, 1.0, 99], "{:g}秒") if last(r, r["a"]) else None)
show("▲の性別", lambda r: ((r["card"].get(r["a"]) or {}).get("sex_age") or "?")[0])
show("▲の馬番", lambda r: "内(1〜4)" if r["a"] <= 4 else "中(5〜10)" if r["a"] <= 10 else "外(11〜)")
show("▲の出走回数", lambda r: band(r["hs"][r["a"]].get("nruns"), [0, 3, 6, 11, 999]))

print("\n■ 人気とのずれ(◎○▲の人気の順位の合計。小さいほど人気どおり)")
show("◎○▲の人気の合計", lambda r: band(sum(r["hs"][u].get("ninki") or 9 for u in (r["h"], r["t"], r["a"])), [0, 7, 9, 12, 99]))
show("◎○▲に1番人気がいるか", lambda r: "いる" if 1 in [r["hs"][u].get("ninki") for u in (r["h"], r["t"], r["a"])] else "いない")


# 市場の3連複の確率(単勝オッズから Harville)と、▲の人気とのずれ
def harville3(q, us):
    p = 0.0
    from itertools import permutations
    for a, b, c in permutations(us):
        p += q[a] * q[b] / (1 - q[a]) * q[c] / (1 - q[a] - q[b])
    return p


for r in races:
    inv = sum(1 / v for v in r["od"].values() if v > 1)
    q = {u: 1 / v / inv for u, v in r["od"].items() if v > 1}
    r["pm"] = harville3(q, (r["h"], r["t"], r["a"])) if all(u in q for u in (r["h"], r["t"], r["a"])) else None
show("市場の見込みの3連複◎○▲の確率", lambda r: band(r["pm"], [0, .03, .06, .1, .15, 1], "{:.0%}"))

print("\n■ 3年そろって 100% から離れるもの(◀ の印)")
for x in FLAG:
    print("  " + x)

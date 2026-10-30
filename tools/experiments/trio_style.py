"""▲の脚質と3着内の傾向(中央、2024年1月〜。予想は日別ファイルの pred = アプリと同じ印・脚質・展開・コース傾向)
  python tools/experiments/trio_style.py
  ▲の脚質を、展開予想(ハイ・スロー・平均)・コースの脚質傾向・◎○の脚質・芝ダート・距離・頭数と組み合わせて、
  ▲の3着内率(実際 / 確率の見込み)と、◎○が両方3着内のときに▲が来た率、3連複◎○▲の回収率(◎80%以上)を年ごとに出す"""
import os, sys
sys.argv = sys.argv[:1]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trio_third.py")
exec(open(__file__).read().split("# ---------------------------------------------------------------- 1.")[0])
from collections import defaultdict
import glob, json
# 展開・コース傾向を足す(trio_third は持っていないので読み直す)
extra = {}
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p = r.get("pred") or {}
        if r.get("center") == "jra" and p.get("model"):
            extra[rid] = ((p.get("pace") or {}).get("pace") or "?", (p.get("course") or {}).get("lean") or "?",
                          (p.get("pace") or {}).get("nige"))
for r in races:
    r["pace"], r["lean"], r["nige"] = extra.get(r["rid"], ("?", "?", None))
A = lambda r: r["hs"][r["a"]].get("style") or "?"
ST = ("逃げ", "先行", "差し", "追込")
YS = ("2024", "2025", "2026")


def cell(xs):
    """▲3着内 実際/見込み、年ごとの差、◎○来た時の▲、3連複回収率"""
    n = len(xs)
    a3 = sum(r["a"] in r["top3"] for r in xs) / n
    ex = sum(r["hs"][r["a"]]["p3"] for r in xs) / n
    dy = []
    for y in YS:
        s = [r for r in xs if r["y"] == y]
        dy.append("{:+3.0f}".format((sum(r["a"] in r["top3"] for r in s) - sum(r["hs"][r["a"]]["p3"] for r in s)) / len(s) * 100) if len(s) >= 15 else "  -")
    bb = [r for r in xs if both(r)]
    tw = sum(third(r) == r["a"] for r in bb) / max(len(bb), 1)
    return n, a3, ex, dy, tw


def table(title, xs, key, keys, min_n=30, roi=False):
    print("  [{}]".format(title))
    g = defaultdict(list)
    for r in xs:
        g[key(r)].append(r)
    for k in keys:
        s = g.get(k, [])
        if len(s) < min_n:
            continue
        n, a3, ex, dy, tw = cell(s)
        line = "    {:22s} {:5d}R  ▲3着内 {:4.0%}(見込み{:4.0%} 差{:+3.0f}  年ごとの差 {})  ◎○来た時に▲ {:4.0%}".format(
            str(k)[:22], n, a3, ex, (a3 - ex) * 100, " ".join(dy), tw)
        if roi:
            line += "  3連複 {:4.0f}% [{}]".format(sum(trio(r) for r in s) / n, " ".join(
                "{:4.0f}".format(sum(trio(r) for r in s if r["y"] == y) / max(1, sum(1 for r in s if r["y"] == y))) for y in YS))
        print(line)


for lab, xs, mn, roi in (("全レース", races, 100, False), ("◎80%以上(買うレース)", buy, 20, True)):
    print("\n■ {} {}レース".format(lab, len(xs)))
    table("▲の脚質", xs, A, ST, mn, roi)
    table("▲の脚質 × 展開予想", xs, lambda r: "{}ペース・▲{}".format(r["pace"], A(r)), ["{}ペース・▲{}".format(p, s) for p in ("ハイ", "平均", "スロー") for s in ST], mn, roi)
    table("▲の脚質 × コースの脚質傾向", xs, lambda r: "{}・▲{}".format(r["lean"], A(r)), ["{}・▲{}".format(l, s) for l in ("前有利", "脚質の偏り小", "差し有利") for s in ST], mn, roi)
    fs = lambda r: "前" if r["hs"][r["h"]].get("style") in ("逃げ", "先行") else "後"
    ts = lambda r: "前" if r["hs"][r["t"]].get("style") in ("逃げ", "先行") else "後"
    table("◎○の脚質(前=逃げ先行 / 後=差し追込) × ▲の脚質", xs, lambda r: "◎{}○{}・▲{}".format(fs(r), ts(r), A(r)),
          ["◎{}○{}・▲{}".format(a, b, s) for a in "前後" for b in "前後" for s in ST], mn, roi)
    table("▲の脚質 × 芝・ダート", xs, lambda r: "{}・▲{}".format(r["surf"], A(r)), ["{}・▲{}".format(sf, s) for sf in ("芝", "ダ") for s in ST], mn, roi)
    table("▲の脚質 × 距離", xs, lambda r: "{}・▲{}".format("短距離(〜1400)" if r["dist"] < 1600 else "マイル〜中距離" if r["dist"] < 2200 else "長距離(2200〜)", A(r)),
          ["{}・▲{}".format(dd, s) for dd in ("短距離(〜1400)", "マイル〜中距離", "長距離(2200〜)") for s in ST], mn, roi)
    table("▲の脚質 × 頭数", xs, lambda r: "{}・▲{}".format("〜12頭" if r["n"] <= 12 else "13頭〜", A(r)), ["{}・▲{}".format(nn, s) for nn in ("〜12頭", "13頭〜") for s in ST], mn, roi)
    table("▲の脚質 × 逃げ馬の数", xs, lambda r: "逃げ{}頭・▲{}".format(min(r["nige"], 3) if r["nige"] is not None else "?", A(r)),
          ["逃げ{}頭・▲{}".format(k, s) for k in (0, 1, 2, 3) for s in ST], mn, roi)

# ◎○が両方3着内のとき、3着内のもう1頭の脚質(▲かどうかに関わらず)と、▲の脚質
print("\n■ ◎○が両方3着内のとき、3着内のもう1頭の脚質(全レース)")
bb = [r for r in races if both(r)]
from collections import Counter
c3 = Counter(r["hs"].get(third(r), {}).get("style") or "?" for r in bb)
ca = Counter(A(r) for r in bb)
cr = Counter(r["hs"][u].get("style") or "?" for r in bb for u in r["rest"])
nr = sum(cr.values())
for s in ST:
    print("  {}: 3着内のもう1頭 {:4.0%} / ▲ {:4.0%} / ◎○を除く全部の馬 {:4.0%}".format(s, c3[s] / len(bb), ca[s] / len(bb), cr[s] / nr))

"""11Rだけの解析(中央、2024年1月〜。予想はアプリと同じ印・確率)
  python tools/experiments/r11.py
  1) 11Rでモデルの確率は当たっているか(◎○▲の3着内 実際 / 見込み)
  2) 券種 × ◎の確率の下限 × ◎と○の差 の回収率を年ごとに
  3) クラス(重賞・オープン・3勝以下)、ハンデ戦かどうかで分ける
  4) 券種 × 条件 を2024・2025年で選び(2年で60R以上)、2026年で確かめる"""
import os, sys, glob, json
from itertools import product
sys.argv = sys.argv[:1]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trio_third.py")
exec(open(__file__).read().split("# ---------------------------------------------------------------- 1.")[0])
YS = ("2024", "2025", "2026")
cond = {}
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        if int(rid[10:12]) == 11:
            cond[rid] = r.get("cond") or ""
R11 = [r for r in races if int(r["rid"][10:12]) == 11]
for r in R11:
    c = cond.get(r["rid"], "")
    r["grade"] = {8: "G1", 7: "G2", 6: "G3", 5: "オープン・L"}.get(r["cls"], "3勝以下")
    r["hc"] = "ハンデ" if "ハンデ" in c else "定量・別定・馬齢"
print("11R {}レース {}".format(len(R11), {y: sum(1 for r in R11 if r["y"] == y) for y in YS}))

key = lambda typ, c: "-".join(str(x) for x in (c if typ in ("馬単", "3連単") else sorted(c)))
pay = lambda r, typ, c: (r["pay"].get(typ) or {}).get(key(typ, c), 0)
wide = lambda r: pay(r, "ワイド", (r["h"], r["t"]))
ext = lambda r: "距離延長" in (r["hs"][r["a"]].get("axis_ng") or [])
BETS = {
    "今(3連複◎○▲・延長ならワイド)": lambda r: [("ワイド", (r["h"], r["t"]))] if ext(r) else [("3連複", (r["h"], r["t"], r["a"]))],
    "単勝◎": lambda r: [("単勝", (r["h"],))],
    "ワイド◎-○": lambda r: [("ワイド", (r["h"], r["t"]))],
    "ワイド◎-○▲(2点)": lambda r: [("ワイド", (r["h"], r["t"])), ("ワイド", (r["h"], r["a"]))],
    "ワイド◎○▲BOX(3点)": lambda r: [("ワイド", c) for c in ((r["h"], r["t"]), (r["h"], r["a"]), (r["t"], r["a"]))],
    "馬連◎-○": lambda r: [("馬連", (r["h"], r["t"]))],
    "馬単◎→○": lambda r: [("馬単", (r["h"], r["t"]))],
    "3連複◎○▲": lambda r: [("3連複", (r["h"], r["t"], r["a"]))],
    "3連複◎○-▲△(2点)": lambda r: [("3連複", (r["h"], r["t"], x)) for x in (r["a"], r["b"])],
    "3連複◎-○▲△(3点)": lambda r: [("3連複", (r["h"], x, y)) for x, y in ((r["t"], r["a"]), (r["t"], r["b"]), (r["a"], r["b"]))],
    "3連複◎○▲△BOX(4点)": lambda r: [("3連複", c) for c in ((r["h"], r["t"], r["a"]), (r["h"], r["t"], r["b"]), (r["h"], r["a"], r["b"]), (r["t"], r["a"], r["b"]))],
    "3連単◎→○→▲": lambda r: [("3連単", (r["h"], r["t"], r["a"]))],
    "3連単◎○▲BOX(6点)": lambda r: [("3連単", (x, y, z)) for x in (r["h"], r["t"], r["a"]) for y in (r["h"], r["t"], r["a"]) for z in (r["h"], r["t"], r["a"]) if len({x, y, z}) == 3],
}
for r in R11:
    r["ret"] = {b: (sum(pay(r, t, c) for t, c in f(r)), 100 * len(f(r))) for b, f in BETS.items()}


def roi(xs, b):
    st = sum(r["ret"][b][1] for r in xs)
    return sum(r["ret"][b][0] for r in xs) / st * 100 if st else 0


def yr(xs, b):
    return " ".join("{:4.0f}".format(roi([r for r in xs if r["y"] == y], b)) if sum(1 for r in xs if r["y"] == y) >= 5 else "   -" for y in YS)


def pl(xs, b):
    return int(sum(r["ret"][b][0] - r["ret"][b][1] for r in xs))


print("\n■ 1. 11Rで確率は当たっているか(3着内 実際 / 見込み)")
for m, k in (("◎", "h"), ("○", "t"), ("▲", "a"), ("△", "b")):
    for lab, xs in (("11R", R11), ("ほかのレース", [r for r in races if int(r["rid"][10:12]) != 11])):
        a = sum(r[k] in r["top3"] for r in xs) / len(xs); e = sum(r["hs"][r[k]]["p3"] for r in xs) / len(xs)
        print("  {} {:8s} 実際 {:4.0%} / 見込み {:4.0%} (差 {:+.0f})".format(m, lab, a, e, (a - e) * 100))

print("\n■ 2. 券種 × ◎の確率の下限(11R全部 / ◎50% / 60% / 70% / 80%以上)  回収率% [2024 2025 2026]")
for b in BETS:
    cells = []
    for t1 in (0, .5, .6, .7, .8):
        xs = [r for r in R11 if r["p1"] >= t1]
        cells.append("{:3d}R {:4.0f}%".format(len(xs), roi(xs, b)))
    print("  {:26s} {}".format(b, " | ".join(cells)))

print("\n■ 3. クラス・ハンデで分ける(◎60%以上)")
for lab, g in (("クラス", lambda r: r["grade"]), ("ハンデ", lambda r: r["hc"]), ("頭数", lambda r: "〜12頭" if r["n"] <= 12 else "13〜15頭" if r["n"] <= 15 else "16頭〜"),
               ("芝・ダート", lambda r: r["surf"])):
    print("  [{}]".format(lab))
    ks = sorted({g(r) for r in R11})
    for k in ks:
        xs = [r for r in R11 if r["p1"] >= .6 and g(r) == k]
        if len(xs) < 15:
            continue
        print("    {:14s} {:3d}R  今 {:4.0f}% [{}] | ワイド◎-○ {:4.0f}% | 3連複◎○▲ {:4.0f}% | 単勝◎ {:4.0f}% | 3連複◎-○▲△ {:4.0f}%".format(
            k, len(xs), roi(xs, "今(3連複◎○▲・延長ならワイド)"), yr(xs, "今(3連複◎○▲・延長ならワイド)"), roi(xs, "ワイド◎-○"),
            roi(xs, "3連複◎○▲"), roi(xs, "単勝◎"), roi(xs, "3連複◎-○▲△(3点)")))

print("\n■ 4. 券種 × ◎の下限 × ◎と○の差 × クラス を2024・2025年で選ぶ(60R以上)→ 2026年")
cands = []
GR = {"全部": lambda r: True, "重賞": lambda r: r["grade"] in ("G1", "G2", "G3"), "重賞以外": lambda r: r["grade"] not in ("G1", "G2", "G3"),
      "ハンデ以外": lambda r: r["hc"] != "ハンデ"}
for b, t1, g, (gl, gf) in product(BETS, (0, .5, .6, .7, .8), (-1, .1, .15, .2), GR.items()):
    f = lambda r: r["p1"] >= t1 and r["p1"] - r["hs"][r["t"]]["p3"] >= g and gf(r)
    sel = [r for r in R11 if r["y"] != "2026" and f(r)]
    if len(sel) < 60:
        continue
    te = [r for r in R11 if r["y"] == "2026" and f(r)]
    cands.append((roi(sel, b), b, t1, g, gl, len(sel), len(te), roi(te, b), pl(te, b), sum(1 for r in te if r["ret"][b][0]) / max(1, len(te))))
cands.sort(key=lambda x: -x[0])
for c in cands[:15]:
    print("  {:26s} ◎{:.0%}以上 差{:>4s} {:6s} 2024-25 {:3d}R {:4.0f}% / 2026 {:3d}R {:4.0f}% 的中{:3.0%} 収支{:+,}円".format(
        c[1], c[2], "なし" if c[3] < 0 else "{:.0%}".format(c[3]), c[4], c[5], c[0], c[6], c[7], c[9], c[8]))
import numpy as np
a = np.array([c[0] for c in cands]); t = np.array([c[7] for c in cands])
print("  組 {} 個。2024-25の回収率と2026年の相関 {:.2f}。2024-25で100%以上の組の2026年の平均 {:.0f}%".format(len(cands), np.corrcoef(a, t)[0, 1], t[a >= 100].mean() if (a >= 100).any() else 0))

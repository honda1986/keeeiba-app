"""3連複◎-○-▲が当たるときの▲の傾向(中央、2024年1月〜。予想は日別ファイルの pred = アプリと同じ印・評価・確率)
  python tools/experiments/trio_third.py
  ◎○が両方3着内に来たとき、3着内のもう1頭が▲かどうかを、▲の確率・人気・脚質・評価(加点・減点・消し・軸不可)などで分ける。
  ほかの馬の評価が当たり方にどう効いているか(評価ごとの3着内率と、確率の見込みとの差)と、▲の選び方を変えたときの回収率も出す"""
import glob, json, math, os, re, sys
from collections import defaultdict, Counter
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
tag = lambda t: re.sub(r"\(.*?\)|[0-9.+\-]+[mkg日]*$", "", t).strip()
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = os.path.basename(f)[:8]
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        p, res = r.get("pred") or {}, r.get("result") or {}
        if r.get("center") != "jra" or not p.get("model") or len(res.get("order") or []) < 3 or not res.get("odds"):
            continue
        pay = res.get("payout") or {}
        if not pay.get("3連複"):
            continue
        hs = {h["umaban"]: h for h in p["horses"] if h.get("p3") is not None}
        mk = {h.get("mark"): u for u, h in hs.items() if h.get("mark") in ("◎", "○", "▲", "△")}
        if not all(m in mk for m in "◎○▲△"):
            continue
        od = {int(u): v for u, v in res["odds"].items() if v}
        if not all(mk[m] in od for m in "◎○▲△"):          # 取消など
            continue
        rest = sorted((u for u in hs if u not in (mk["◎"], mk["○"]) and u in od), key=lambda u: -hs[u]["p3"])
        pop = sorted((u for u in hs if u in od), key=lambda u: od[u])
        waku = {h["umaban"]: h.get("waku") for h in r.get("horses") or []}
        races.append(dict(d=d, y=d[:4], rid=rid, hs=hs, h=mk["◎"], t=mk["○"], a=mk["▲"], b=mk["△"], rest=rest, od=od,
                          nk={u: i + 1 for i, u in enumerate(pop)}, order=res["order"], top3=set(res["order"][:3]), pay=pay,
                          n=len(hs), surf=r.get("surface"), dist=r.get("dist") or 0, cls=r.get("cls"), baba=r.get("baba"),
                          waku=waku, buy=p.get("verdict") in ("A", "B"), p1=hs[mk["◎"]]["p3"]))
buy = [r for r in races if r["p1"] >= .8]
print("中央 {} レース(2024〜) / ◎80%以上 {} レース".format(len(races), len(buy)))


def trio(r, a=None):
    a = a or r["a"]
    return r["pay"]["3連複"].get("-".join(str(x) for x in sorted((r["h"], r["t"], a))), 0)


def tri(r, o):
    return (r["pay"].get("3連単") or {}).get("-".join(str(x) for x in o), 0)


both = lambda r: r["h"] in r["top3"] and r["t"] in r["top3"]
third = lambda r: next(u for u in r["order"][:3] if u not in (r["h"], r["t"]))

# ---------------------------------------------------------------- 1. 3着内のもう1頭はだれか
for lab, xs in (("◎80%以上", buy), ("全レース", races)):
    bb = [r for r in xs if both(r)]
    print("\n■ 1. {}: ◎○が両方3着内 {}/{} ({:.0%})。そのとき3着内のもう1頭は".format(lab, len(bb), len(xs), len(bb) / len(xs)))
    c1, c2 = Counter(), Counter()
    for r in bb:
        x = third(r)
        c1[min(r["rest"].index(x) + 1, 6) if x in r["rest"] else 9] += 1
        rk = sorted(r["rest"], key=lambda u: r["od"][u]).index(x) + 1 if x in r["rest"] else 9
        c2[min(rk, 6)] += 1
    print("  確率の順(◎○を除く) " + " ".join("{}{}: {:.0%}".format("▲△"[k - 1] if k <= 2 else str(k) + "番目", "以下" if k == 6 else "", c1[k] / len(bb)) for k in range(1, 7)))
    print("  人気の順(◎○を除く) " + " ".join("{}番手{}: {:.0%}".format(k, "以下" if k == 6 else "", c2[k] / len(bb)) for k in range(1, 7)))
    aa = [r for r in bb if r["a"] == r["rest"][0]]
    print("  ▲が当たる(3着内のもう1頭が▲) {:.0%} / ▲の確率の平均 {:.0%} / ▲がそのレースで◎○を除いた1番人気 {:.0%}".format(
        sum(third(r) == r["a"] for r in bb) / len(bb), sum(r["hs"][r["a"]]["p3"] for r in bb) / len(bb),
        sum(sorted(r["rest"], key=lambda u: r["od"][u])[0] == r["a"] for r in bb) / len(bb)))


# ---------------------------------------------------------------- 2. ▲の性質ごと
def show(title, xs, key, order=None, min_n=25):
    """key(r) -> 分ける値。◎○が両方3着内のとき▲が来た率(実際 / ▲の確率から見込み)と、3連複◎○▲の回収率"""
    g = defaultdict(list)
    for r in xs:
        for k in (key(r) if isinstance(key(r), (list, set, tuple)) else [key(r)]):
            g[k].append(r)
    ks = order or sorted(g, key=lambda k: -len(g[k]))
    print("  [{}]".format(title))
    for k in ks:
        s = g.get(k, [])
        if len(s) < min_n:
            continue
        bb = [r for r in s if both(r)]
        hit = sum(third(r) == r["a"] for r in bb)
        a3 = sum(r["a"] in r["top3"] for r in s) / len(s)
        exp = sum(r["hs"][r["a"]]["p3"] for r in s) / len(s)
        roi = sum(trio(r) for r in s) / len(s)
        by = " ".join("{:4.0f}".format(sum(trio(r) for r in s if r["y"] == y) / max(1, sum(1 for r in s if r["y"] == y))) for y in ("2024", "2025", "2026"))
        print("    {:22s} {:4d}R  ▲3着内 {:4.0%}(見込み{:4.0%} 差{:+4.0f}) ◎○来た時に▲ {:4.0%}  3連複 {:4.0f}% [{}]".format(
            str(k)[:22], len(s), a3, exp, (a3 - exp) * 100, hit / max(len(bb), 1), roi, by))


def band(v, cuts, fmt="{:.0%}"):
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if lo <= v < hi:
            return (fmt + "〜" + fmt).format(lo, hi) if hi < 9e9 else (fmt + "〜").format(lo)
    return "?"


A = lambda r: r["hs"][r["a"]]
for lab, xs in (("◎80%以上(買うレース)", buy), ("全レース(▲の傾向を数で見る)", races)):
    print("\n■ 2. {}: ▲の性質ごと  [3連複の回収率 2024 2025 2026]".format(lab))
    mn = 25 if xs is buy else 150
    show("▲の確率", xs, lambda r: band(A(r)["p3"], [0, .3, .4, .5, .6, 1.01]), min_n=mn)
    show("▲と△の確率の差", xs, lambda r: band(A(r)["p3"] - r["hs"][r["b"]]["p3"], [0, .03, .08, .15, 1.01]), min_n=mn)
    show("▲の人気", xs, lambda r: "{}番人気".format(r["nk"][r["a"]]) if r["nk"][r["a"]] <= 5 else "6〜9番人気" if r["nk"][r["a"]] <= 9 else "10番人気〜", min_n=mn)
    show("▲の単勝オッズ", xs, lambda r: band(r["od"][r["a"]], [1, 5, 10, 20, 50, 9e9], "{:g}倍"), min_n=mn)
    show("▲の脚質", xs, lambda r: A(r).get("style") or "?", min_n=mn)
    show("▲の脚質が◎・○と同じか", xs, lambda r: "◎とも○とも違う" if A(r).get("style") not in (r["hs"][r["h"]].get("style"), r["hs"][r["t"]].get("style")) else "どちらかと同じ", min_n=mn)
    show("▲が消しか", xs, lambda r: "消し: " + (A(r).get("cut_reason") or "?") if A(r).get("cut") else "消しでない", min_n=mn)
    show("▲の軸不可", xs, lambda r: ["軸不可: " + x for x in A(r).get("axis_ng") or []] or ["軸不可でない"], min_n=mn)
    show("▲の弱点", xs, lambda r: ["弱点: " + x for x in A(r).get("weak") or []] or ["弱点なし"], min_n=mn)
    show("▲の減点", xs, lambda r: ["減点: " + tag(x["t"]) for x in A(r).get("minus") or []] or ["減点なし"], min_n=mn)
    show("▲の加点", xs, lambda r: ["加点: " + tag(x["t"]) for x in A(r).get("plus") or []] or ["加点なし"], min_n=mn)
    show("▲の加点−減点", xs, lambda r: band(sum(x["p"] for x in A(r).get("plus") or []) + sum(x["p"] for x in A(r).get("minus") or []), [-99, -4, -2, 0, 1, 3, 99], "{:g}"), min_n=mn)
    show("▲の人気と確率のずれ", xs, lambda r: A(r).get("odds_note") or "なし", min_n=mn)
    show("▲の休み", xs, lambda r: band(A(r).get("rest_days") or 0, [0, 15, 36, 71, 181, 9999], "{:g}日"), min_n=mn)
    show("▲の枠", xs, lambda r: "{}枠".format(r["waku"].get(r["a"])) if r["waku"].get(r["a"]) else "?", min_n=mn)
    show("頭数", xs, lambda r: band(r["n"], [0, 10, 13, 16, 99], "{:g}頭"), min_n=mn)
    show("芝・ダート", xs, lambda r: r["surf"] or "?", min_n=mn)
    show("距離", xs, lambda r: band(r["dist"], [0, 1400, 1800, 2200, 9999], "{:g}m"), min_n=mn)

# ---------------------------------------------------------------- 3. ほかの馬の評価: 評価ごとの3着内率と見込みのずれ(◎○を除く全部の馬)
print("\n■ 3. ◎○を除く全部の馬: 評価ごとの3着内率(実際 / 確率の見込み)。差が大きいほど、確率がその評価を見落としている")
acc = defaultdict(lambda: [0, 0.0, 0.0])
for r in races:
    for u in r["rest"]:
        h = r["hs"][u]; y = u in r["top3"]
        keys = (["減点: " + tag(x["t"]) for x in h.get("minus") or []] + ["加点: " + tag(x["t"]) for x in h.get("plus") or []]
                + ["軸不可: " + x for x in h.get("axis_ng") or []] + ["弱点: " + x for x in h.get("weak") or []]
                + (["消し: " + (h.get("cut_reason") or "?")] if h.get("cut") else []) + (["人気とのずれ: " + h["odds_note"]] if h.get("odds_note") else [])
                + ["脚質: " + (h.get("style") or "?"), "全部"])
        for k in keys:
            a = acc[k]; a[0] += 1; a[1] += y; a[2] += h["p3"]
out = []
for k, (n, y, e) in acc.items():
    if n >= 300:
        z = (y - e) / math.sqrt(max(e * (1 - e / n), 1))
        out.append((z, k, n, y / n, e / n))
for z, k, n, a, e in sorted(out, key=lambda x: -x[0]):
    print("  {:26s} {:6d}頭  実際 {:5.1%} / 見込み {:5.1%}  差 {:+5.1f}ポイント (z {:+5.1f})".format(k[:26], n, a, e, (a - e) * 100, z))

# ---------------------------------------------------------------- 4. ▲の選び方を変えたら(◎80%以上、3連複1点)
print("\n■ 4. ▲の選び方を変えたら(◎80%以上、3連複◎-○-x の1点)  2024 2025 2026 計 / 的中")
pick = {
    "今(確率で◎○の次)": lambda r: r["a"],
    "△(確率で4番目)": lambda r: r["b"],
    "◎○を除いた1番人気": lambda r: min(r["rest"], key=lambda u: r["od"][u]),
    "消しでない中で確率最上位": lambda r: next((u for u in r["rest"] if not r["hs"][u].get("cut")), r["a"]),
    "軸不可でない中で確率最上位": lambda r: next((u for u in r["rest"] if not r["hs"][u].get("axis_ng")), r["a"]),
    "▲と△で人気の上の方": lambda r: min((r["a"], r["b"]), key=lambda u: r["od"][u]),
    "▲と△で人気の下の方": lambda r: max((r["a"], r["b"]), key=lambda u: r["od"][u]),
    "確率×市場で最上位": lambda r: max(r["rest"], key=lambda u: math.log(r["hs"][u]["p3"] + 1e-6) + math.log(1 / r["od"][u])),
    "加点−減点が最上位(▲△☆)": lambda r: max(r["rest"][:3], key=lambda u: sum(x["p"] for x in r["hs"][u].get("plus") or []) + sum(x["p"] for x in r["hs"][u].get("minus") or [])),
}
for lab, f in pick.items():
    cols, tot, hit = [], 0, 0
    for y in ("2024", "2025", "2026"):
        s = [r for r in buy if r["y"] == y]
        cols.append("{:4.0f}".format(sum(trio(r, f(r)) for r in s) / len(s)))
    tot = sum(trio(r, f(r)) for r in buy) / len(buy); hit = sum(1 for r in buy if trio(r, f(r))) / len(buy)
    same = sum(1 for r in buy if f(r) == r["a"]) / len(buy)
    print("  {:24s} {} 計{:4.0f}% 的中{:3.0%}  (▲と同じ馬 {:3.0%})".format(lab, " ".join(cols), tot, hit, same))

# ---------------------------------------------------------------- 5. 3連単: 当たるときの着順
print("\n■ 5. 3連単(◎80%以上)")
hits = [r for r in buy if trio(r)]
c = Counter("→".join("◎" if u == r["h"] else "○" if u == r["t"] else "▲" for u in r["order"][:3]) for r in hits)
print("  3連複◎○▲が当たった {} レースの着順: ".format(len(hits)) + " / ".join("{} {:.0%}".format(k, v / len(hits)) for k, v in c.most_common()))
FORMS = {
    "◎→○→▲(1点)": lambda r: [(r["h"], r["t"], r["a"])],
    "◎→○▲→○▲(2点)": lambda r: [(r["h"], r["t"], r["a"]), (r["h"], r["a"], r["t"])],
    "◎○→◎○→▲(2点)": lambda r: [(r["h"], r["t"], r["a"]), (r["t"], r["h"], r["a"])],
    "◎→○→▲△(2点)": lambda r: [(r["h"], r["t"], r["a"]), (r["h"], r["t"], r["b"])],
    "◎○▲ボックス(6点)": lambda r: [(x, y, z) for x in (r["h"], r["t"], r["a"]) for y in (r["h"], r["t"], r["a"]) for z in (r["h"], r["t"], r["a"]) if len({x, y, z}) == 3],
}
for lab, f in FORMS.items():
    cols = []
    for y in ("2024", "2025", "2026", None):
        s = [r for r in buy if y is None or r["y"] == y]
        st = sum(100 * len(f(r)) for r in s); rt = sum(sum(tri(r, o) for o in f(r)) for r in s)
        cols.append("{:4.0f}".format(rt / st * 100))
    hit = sum(1 for r in buy if any(tri(r, o) for o in f(r))) / len(buy)
    print("  {:20s} {} {} {} 計{}% 的中{:3.0%}".format(lab, *cols, hit))
print("  (比べる: 3連複◎○▲ 1点 {:.0f}%)".format(sum(trio(r) for r in buy) / len(buy)))

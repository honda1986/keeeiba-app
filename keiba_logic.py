#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keiba_logic.py -- 競馬の「買うレースを選ぶ」予想ロジック(取得とは切り離した純粋な計算)

■ 流れ(README の 30 か条をこの順に実装)
  1. 能力を作る(人気・オッズは使わない)            … 規則 5, 20
  2. 加点と減点(コース・距離・馬場・展開・枠・斤量・休み明け・クラス差)
                                                     … 規則 9〜19
  3. 減点が重なった馬を消す(負けパターンの最大公約数) … 規則 11〜13
  4. ◎○を決める(延長は軸にしない / 弱点が同じなら○を替える) … 規則 6〜8, 15
  5. このレースを買うか決める(2強だけ買う。1強・混戦は見送り) … 規則 1〜4
  6. 買い目(◎○2頭固定の3連複・3連単、馬連・ワイド、◎飛びの保険、3頭目に穴)
                                                     … 規則 21〜29
  オッズは最後に「人気との食い違い」を見るだけに使う。

■ 数字はすべて目安。history(収支)で条件別の回収率を見て、閾値を締めていく(規則 30)。
"""

import json
import math
import os
import re
import unicodedata
from datetime import date

# ---------------------------------------------------------------- 競馬場
# 名前 -> (中央/地方, 回り, 坂)
VENUES = {
    "札幌": ("jra", "右", "平坦"), "函館": ("jra", "右", "平坦"), "福島": ("jra", "右", "小坂"),
    "新潟": ("jra", "左", "平坦"), "東京": ("jra", "左", "坂"), "中山": ("jra", "右", "急坂"),
    "中京": ("jra", "左", "急坂"), "京都": ("jra", "右", "平坦"), "阪神": ("jra", "右", "急坂"),
    "小倉": ("jra", "右", "平坦"),
    "門別": ("nar", "右", "平坦"), "盛岡": ("nar", "左", "坂"), "水沢": ("nar", "右", "平坦"),
    "浦和": ("nar", "左", "平坦"), "船橋": ("nar", "左", "平坦"), "大井": ("nar", "右", "平坦"),
    "川崎": ("nar", "左", "平坦"), "金沢": ("nar", "右", "平坦"), "笠松": ("nar", "右", "平坦"),
    "名古屋": ("nar", "右", "平坦"), "園田": ("nar", "右", "平坦"), "姫路": ("nar", "右", "平坦"),
    "高知": ("nar", "右", "平坦"), "佐賀": ("nar", "右", "平坦"), "帯広": ("nar", "直", "坂"),
}
JRA_CODE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
NAR_CODE = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
            "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
            "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

HEAVY = ("重", "不")
NAR_CLS = {"A1": 4.5, "A2": 4.0, "A3": 3.75, "A4": 3.6, "B1": 3.5, "B2": 3.0, "B3": 2.75,
           "B4": 2.5, "C1": 2.25, "C2": 1.75, "C3": 1.25, "C4": 1.0}          # 道悪とみなす馬場

# ---------------------------------------------------------------- コース傾向(中央)
# keiba_course.json … titanic-online.com「競馬場の特徴」の2021〜2025年の集計(10場78コース)。
# 脚質の数字は「そのレースでの位置取り」なので、どのコースでも逃げ・先行の成績が良く出る。
# そのまま使わず、同じ芝/ダートの全コース平均と比べた「このコースではどれだけ得か」を使う。
COURSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba_course.json")
COURSE_NAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba_course_nar.json")
STYLE_MAP = {"逃げ": "逃げ", "先行": "先行", "差し": "中団", "追込": "後方"}
COURSE_STYLES = ("逃げ", "先行", "中団", "後方")


def _shrink(top3, n, base, k):
    """3着内率を平均へ寄せてから平均との比にする(出走が少ないコースは1に近づく)"""
    return ((top3 + k * base) / (n + k)) / base if base else 1.0


def _course_index(c, g, gf, exact):
    style = {s: round(_shrink(c["style"][s][2], c["style"][s][0], g[s], 100), 3)
             for s in COURSE_STYLES if s in c["style"]}
    tn = sum(v[0] for v in c["waku"].values())
    tt = sum(v[2] for v in c["waku"].values())
    avg = tt / tn if tn else 0
    waku = {int(w): round(_shrink(v[2], v[0], avg, 150), 3) for w, v in c["waku"].items()} if exact and avg else {}
    n12 = sum(c["ninki"].get(k, [0, 0, 0])[0] for k in ("1", "2"))
    t12 = sum(c["ninki"].get(k, [0, 0, 0])[2] for k in ("1", "2"))
    return {"name": "{}{}{}".format(c["venue"], "芝" if c["surface"] == "芝" else "ダ",
                                     c["dist"] if exact else "(場平均)"),
            "exact": exact, "text": c.get("text", {}) if exact else {}, "style_idx": style, "waku_idx": waku,
            "kata": round(_shrink(t12, n12, gf, 100), 3), "races": c["ninki"].get("1", [0])[0],
            "fav_top3": round(t12 / n12, 3) if n12 else None, "url": c.get("url")}


def load_courses(path=COURSE_FILE):
    """(場, 芝/ダ, 距離) -> 傾向。距離が無いときのために (場, 芝/ダ, None) に場の合計も入れる"""
    try:
        raw = json.load(open(path, encoding="utf-8"))["courses"]
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for surf in ("芝", "ダ"):
        cs = [c for c in raw if c["surface"] == surf]
        if not cs:
            continue
        g = {}
        for st in COURSE_STYLES:
            n = sum(c["style"].get(st, [0, 0, 0])[0] for c in cs)
            g[st] = sum(c["style"].get(st, [0, 0, 0])[2] for c in cs) / n if n else 0
        n12 = sum(c["ninki"][k][0] for c in cs for k in ("1", "2") if k in c["ninki"])
        gf = sum(c["ninki"][k][2] for c in cs for k in ("1", "2") if k in c["ninki"]) / n12 if n12 else 0
        agg = {}
        for c in cs:
            out[(c["venue"], surf, c["dist"])] = _course_index(c, g, gf, True)
            a = agg.setdefault(c["venue"], {"venue": c["venue"], "surface": surf, "dist": None,
                                            "style": {}, "ninki": {}, "waku": {}})
            for part in ("style", "ninki"):
                for k, v in c[part].items():
                    x = a[part].setdefault(k, [0, 0, 0])
                    for i in range(3):
                        x[i] += v[i]
        for v, a in agg.items():
            out[(v, surf, None)] = _course_index(a, g, gf, False)
    return out


def _nar_index(c, avg_style, avg_fav, venue_note, exact, source):
    """地方の1コース(または1場)の傾向。脚質は記事の率(実際の位置取り)なので、地方全体の平均との比を
    出走数で 1 に寄せる。枠は枠ごとの3着内率をコース平均へ寄せてから比にする"""
    n_all = sum(v[0] for v in c["waku"].values())
    w = n_all / (n_all + 2000.0)
    style = {}
    for st in COURSE_STYLES:
        r = (c["style"].get(st) or {}).get("top3")
        if r and avg_style.get(st):
            style[st] = round(1 + w * (r / avg_style[st] - 1), 3)
    tt = sum(v[2] for v in c["waku"].values())
    avg = tt / n_all if n_all else 0
    waku = {int(k): round(_shrink(v[2], v[0], avg, 150), 3) for k, v in c["waku"].items()} if avg else {}
    n12 = sum(c["ninki"].get(k, [0, 0, 0])[0] for k in ("1", "2"))
    t12 = sum(c["ninki"].get(k, [0, 0, 0])[2] for k in ("1", "2"))
    text = {}
    if c.get("note"):
        text["特徴"] = c["note"]
    if exact and venue_note:
        text["場"] = venue_note
    return {"name": "{}{}{}".format(c["venue"], "芝" if c["surface"] == "芝" else "ダ", c["dist"] or "(場全体)"),
            "exact": exact, "text": text, "style_idx": style, "waku_idx": waku,
            "kata": round(_shrink(t12, n12, avg_fav, 100), 3) if n12 else 1.0,
            "races": c["ninki"].get("1", [0])[0], "fav_top3": round(t12 / n12, 3) if n12 else None,
            "url": c.get("url"), "source": source}


def load_courses_nar(path=COURSE_NAR_FILE):
    """地方の (場, 芝/ダ, 距離) -> 傾向。場全体は (場, ダ, None)。記事に無い場全体は距離ごとの数字を足して作る"""
    try:
        raw = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    courses, venues, source = raw.get("courses", []), {v["venue"]: v for v in raw.get("venues", [])}, raw.get("source", "")
    for v, vc in venues.items():
        if not vc["waku"] or not vc["style"]:
            agg = {"waku": {}, "ninki": {}, "style": {}}
            wsum = {}
            for c in courses:
                if c["venue"] == v and c["surface"] == "ダ" and c["waku"]:
                    for part in ("waku", "ninki"):
                        for k, x in c[part].items():
                            y = agg[part].setdefault(k, [0, 0, 0])
                            for i in range(3):
                                y[i] += x[i]
                    n = sum(x[0] for x in c["waku"].values())
                    for st, r in c["style"].items():
                        if r.get("top3") is not None:
                            a = wsum.setdefault(st, [0.0, 0])
                            a[0] += r["top3"] * n
                            a[1] += n
            agg["style"] = {st: {"top3": round(a[0] / a[1], 1)} for st, a in wsum.items() if a[1]}
            for k in ("waku", "ninki", "style"):
                if not vc[k]:
                    vc[k] = agg[k]
    allc = [c for c in courses if c["waku"]] or list(venues.values())
    avg_style = {}
    for st in COURSE_STYLES:
        num = sum((c["style"].get(st) or {}).get("top3", 0) * sum(x[0] for x in c["waku"].values()) for c in allc
                  if (c["style"].get(st) or {}).get("top3"))
        den = sum(sum(x[0] for x in c["waku"].values()) for c in allc if (c["style"].get(st) or {}).get("top3"))
        avg_style[st] = num / den if den else 0
    n12 = sum(c["ninki"].get(k, [0, 0, 0])[0] for c in allc for k in ("1", "2"))
    avg_fav = sum(c["ninki"].get(k, [0, 0, 0])[2] for c in allc for k in ("1", "2")) / n12 if n12 else 0
    out = {}
    for c in courses:
        vc = venues.get(c["venue"]) or {}
        if c.get("stats") == "場全体":       # 数字の無い距離は場全体の数字に、この距離の説明を付ける
            c = dict(c, **{k: vc.get(k) or {} for k in ("waku", "ninki", "style")})
            out[(c["venue"], c["surface"], c["dist"])] = dict(_nar_index(c, avg_style, avg_fav, vc.get("note"), False, source),
                                                             name="{}ダ{}(数字は場全体)".format(c["venue"], c["dist"]))
            continue
        out[(c["venue"], c["surface"], c["dist"])] = _nar_index(c, avg_style, avg_fav, vc.get("note"), True, source)
    for v, vc in venues.items():
        out[(v, "ダ", None)] = _nar_index(vc, avg_style, avg_fav, None, False, source)
    return out


COURSES = load_courses()
COURSES.update(load_courses_nar())
# 地方のコース傾向を予想に使うか(2026-10-19 から使う。KEIBA_NAR_COURSE=0 で切って比べられる)。
# 切り替えたらモデルを学び直すこと(モデルは30か条の加減点も材料にしているので)
USE_NAR_COURSE = os.environ.get("KEIBA_NAR_COURSE", "1") == "1"


def course_of(race):
    """コース傾向(中央は titanic-online.com、地方はうまめし.com の記事から)。無ければ None"""
    if race.get("surface") not in ("芝", "ダ"):
        return None
    if race.get("center") != "jra" and not USE_NAR_COURSE:
        return None
    return (COURSES.get((race.get("venue"), race["surface"], race.get("dist")))
            or COURSES.get((race.get("venue"), race["surface"], None)))


def course_summary(cs):
    """アプリに出す要約"""
    si = cs["style_idx"]
    front = (si.get("逃げ", 1) + si.get("先行", 1)) / 2
    back = (si.get("中団", 1) + si.get("後方", 1)) / 2
    lean = "前有利" if front - back >= 0.12 else ("差し有利" if back - front >= 0.12 else "脚質の偏り小")
    wi = cs["waku_idx"]
    if wi:
        inner = (wi.get(1, 1) + wi.get(2, 1)) / 2
        outer = (wi.get(7, 1) + wi.get(8, 1)) / 2
        wlean = "内枠有利" if inner - outer >= 0.08 else ("外枠有利" if outer - inner >= 0.08 else "枠の差小")
    else:
        wlean = None
    kata = "堅い" if cs["kata"] >= 1.05 else ("荒れやすい" if cs["kata"] <= 0.95 else "標準")
    return {"name": cs["name"], "exact": cs["exact"], "lean": lean, "waku": wlean, "kata": kata,
            "kata_idx": cs["kata"], "fav_top3": cs["fav_top3"], "races": cs["races"],
            "style_idx": si, "text": cs["text"], "url": cs["url"], "source": cs.get("source")}


# ---------------------------------------------------------------- 閾値(目安)
TH = {
    "min_runners": 7,        # これ未満の頭数は配当が付かないので見送り
    "max_nodata": 0.34,      # 過去走が無い馬がこの割合を超えたら能力が測れないので見送り
    "gap_a": 6.0,            # ◎○と3番手の差。これ以上で「A 2強」
    "gap_b": 3.5,            # これ以上で「B 2強」。未満は見送り
    "one_strong": 8.0,       # ◎と○の差がこれ以上で、2強差が足りなければ「1強」
    "cut_gap": 9.0,          # 3番手の馬からこれ以上離れた馬は消す
    "cut_neg": 3,            # 重い減点(-2以下)がこの数以上重なったら消す
    "alt_taikou": 4.0,       # ○を替えるとき、元の○からこの差までの馬を候補にする
    "aite_max": 5,           # 3連複の相手(3頭目候補)の最大数
    "tan3_aite_max": 4,      # 3連単の3着候補の最大数
    "ana_ninki": 6,          # この人気以下を穴とみなす
    "kata_adj": 0.5,         # 堅いコースは2強の基準をこれだけ下げ、荒れやすいコースは上げる
    "bet_mode": "wide1",     # wide1=ワイド◎-○の1点 / wide=ワイド◎-○×5+相手4点 / full=3連複・3連単・馬連・ワイド
    "course_style_k": 5.0,   # コースの脚質傾向の効き(比の対数 × これ)
    "course_waku_k": 5.0,    # コースの枠順傾向の効き
}
UNIT = 100                   # 1点あたりの金額(円)


# ---------------------------------------------------------------- 文字の整理
def nfkc(s):
    return unicodedata.normalize("NFKC", s or "")


def class_level(text, center="jra"):
    """レース名・条件の文字列からクラスの高さを数にする(中央と地方は別の物差し)"""
    t = nfkc(text).replace(" ", "")
    m = re.search(r"Jpn(III|II|I|3|2|1)", t)
    if m:
        return {"I": 8, "1": 8, "II": 7, "2": 7, "III": 6, "3": 6}[m.group(1)]
    m = re.search(r"G(III|II|I|3|2|1)(?![0-9])", t)
    if m:
        return {"I": 8, "1": 8, "II": 7, "2": 7, "III": 6, "3": 6}[m.group(1)]
    if center == "jra":
        if re.search(r"\(L\)|リステッド|オープン|OP", t):
            return 5
        if re.search(r"3勝|1600万", t):
            return 4
        if re.search(r"2勝|1000万", t):
            return 3
        if re.search(r"1勝|500万", t):
            return 2
        if re.search(r"未勝利|新馬", t):
            return 1
        return None
    if re.search(r"重賞|S[1-3P]|M[1-3]|H[1-3]|オープン|OP", t):
        return 5
    m = re.search(r"([ABC])([1-4])", t)
    if m:
        return NAR_CLS[m.group(0)]
    if re.search(r"新馬|未格付|未勝利|ルーキー", t):
        return 1.0
    return None


def venue_info(name):
    for k, v in VENUES.items():
        if k in (name or ""):
            return k, v
    return None, None


def parse_date(s):
    try:
        y, m, d = (int(x) for x in re.findall(r"\d+", s)[:3])
        return date(y, m, d)
    except Exception:
        return None


# ---------------------------------------------------------------- 1走ごとの評価
def run_score(run, now_center):
    """過去1走のでき。着差(秒)を主に、無ければ着順/頭数。クラス差で補正する。"""
    rank, n = run.get("rank"), run.get("n") or 0
    if not rank or n < 2:
        return None
    m = run.get("margin")
    if m is not None:
        if m <= 0:      # 勝った(カッコは2着との差。マイナス)
            s = 60 + min(-m, 1.0) * 8
        else:
            s = 60 - min(m, 3.0) * 13
    else:
        s = 60 - (rank - 1) / (n - 1) * 36
    # クラス差: 上のクラスでの好走ほど価値が高い
    lv = run.get("cls")
    if lv is not None and run.get("now_cls") is not None:
        c = run.get("center") or now_center
        adj = lv
        if c == "jra" and now_center == "nar":
            adj += 1.5
        elif c == "nar" and now_center == "jra":
            adj -= 1.5
        s += max(-12.0, min(12.0, (adj - run["now_cls"]) * 4))
    return s


def run_style(run):
    """通過順の最初のコーナーで脚質を決める"""
    p = run.get("pass") or []
    n = run.get("n") or 0
    if not p or n < 2:
        return None
    r = (p[0] - 1) / max(n - 1, 1)
    if p[0] == 1:
        return "逃げ"
    if r <= 0.33 or p[0] <= 3:
        return "先行"
    if r <= 0.7:
        return "差し"
    return "追込"


# ---------------------------------------------------------------- 馬ごとの評価
W_RECENT = [1.0, 0.8, 0.65, 0.5, 0.4]


def evaluate_horse(h, race, today):
    """能力(加点の土台)と、減点・加点の理由を出す。人気・オッズは見ない。"""
    center = race["center"]
    past = [dict(r) for r in (h.get("past") or []) if r.get("surface") != "障"][:5]
    for r in past:
        r["now_cls"] = race.get("cls")
        vn, vi = venue_info(r.get("venue"))
        r["center"] = vi[0] if vi else center
        r["dir"] = vi[1] if vi else None
        r["score"] = run_score(r, center)
    scored = [r for r in past if r["score"] is not None]

    # 能力: 加重平均と、良い2走の平均を半々。走数が少なければ平均(45)へ寄せる
    if scored:
        ws = W_RECENT[:len(scored)]
        wmean = sum(r["score"] * w for r, w in zip(scored, ws)) / sum(ws)
        best2 = sorted((r["score"] for r in scored[:4]), reverse=True)[:2]
        raw = 0.5 * wmean + 0.5 * sum(best2) / len(best2)
        k = max(0, 3 - len(scored))
        ability = (raw * len(scored) + 45 * k) / (len(scored) + k)
    else:
        ability = 40.0
    ev = {"umaban": h["umaban"], "ability": round(ability, 1), "plus": [], "minus": [],
          "nruns": len(scored), "axis_ng": [], "weak": set()}

    def plus(t, p):
        ev["plus"].append({"t": t, "p": round(p, 1)})

    def minus(t, p, weak=None):
        ev["minus"].append({"t": t, "p": round(-abs(p), 1)})
        if weak and abs(p) >= 2:      # 軽い減点は「弱点」に数えない
            ev["weak"].add(weak)

    # 脚質
    styles = [run_style(r) for r in past]
    styles = [s for s in styles if s]
    nige = sum(1 for s in styles if s == "逃げ")
    if styles:
        cnt = {}
        for s, w in zip(styles, W_RECENT):
            cnt[s] = cnt.get(s, 0) + w
        style = max(cnt, key=cnt.get)
        if nige / len(styles) >= 0.4:
            style = "逃げ"
    else:
        style = None
    ev["style"] = style

    if not scored:
        minus("過去走なし(能力不明)", 3)
        ev["axis_ng"].append("能力不明")
        return ev
    avg = sum(r["score"] for r in scored) / len(scored)
    last = past[0] if past else {}

    # 規則 14: コース適性(同じ競馬場 / 同じ回り)
    same_v = [r["score"] for r in scored if r.get("venue") and race["venue"] in r["venue"]]
    if len(same_v) >= 1:
        d = sum(same_v) / len(same_v) - avg
        if d >= 3:
            plus("当地実績", min(d * 0.4, 3))
        elif d <= -5:
            minus("当地で凡走", min(-d * 0.3, 3), "コース")
    if race.get("direction") in ("右", "左"):
        same = [r["score"] for r in scored if r.get("dir") == race["direction"]]
        opp = [r["score"] for r in scored if r.get("dir") and r["dir"] != race["direction"]]
        if same and opp and sum(opp) / len(opp) - sum(same) / len(same) >= 6:
            minus("{}回り△".format(race["direction"]), 2, "回り")
        elif not same and len(opp) >= 3:
            minus("{}回り未経験".format(race["direction"]), 1)

    # 芝・ダートの替わり
    surf = [r for r in scored if r.get("surface") == race["surface"]]
    if not surf and race["surface"] in ("芝", "ダ"):
        minus("初{}".format("芝" if race["surface"] == "芝" else "ダート"), 3, "替わり")
        ev["axis_ng"].append("条件替わり")

    # 規則 15: 距離変更(延長は軸にしない)
    dist = race.get("dist") or 0
    if last.get("dist") and dist:
        dd = dist - last["dist"]
        if dd >= 200:
            proven = any(r.get("dist", 0) >= dist - 100 and (r["rank"] or 99) <= 3
                         for r in scored)
            minus("距離延長(+{}m)".format(dd), 1.5 if proven else 3, "延長")
            ev["axis_ng"].append("距離延長")
        elif dd <= -200:
            minus("距離短縮({}m)".format(dd), 1)
    near = [r for r in scored if r.get("dist") and abs(r["dist"] - dist) <= 200]
    if len(scored) >= 3:
        if not near:
            minus("距離実績なし", 2, "距離")
        elif max(r["score"] for r in near) < avg - 4:
            minus("この距離で凡走", 2, "距離")
        elif any((r["rank"] or 99) <= 2 for r in near):
            plus("距離実績", 1.5)

    # 規則 16: 馬場(良と道悪は別物)
    heavy_now = (race.get("baba") or "良")[:1] in HEAVY
    hv = [r["score"] for r in scored if (r.get("baba") or "良")[:1] in HEAVY]
    gd = [r["score"] for r in scored if (r.get("baba") or "良")[:1] not in HEAVY]
    if heavy_now:
        if hv and sum(hv) / len(hv) <= avg - 5:
            minus("道悪△", 4, "道悪")
        elif hv and sum(hv) / len(hv) >= avg + 3:
            plus("道悪巧者", 2)
        elif not hv:
            minus("道悪未経験", 1, "道悪")
    elif hv and gd and sum(gd) / len(gd) <= sum(hv) / len(hv) - 6:
        minus("良馬場△(道悪向き)", 2)

    # 規則 10: 休み明け
    if last.get("date") and today:
        d0 = parse_date(last["date"])
        if d0:
            days = (today - d0).days
            ev["rest_days"] = days
            if days >= 180:
                minus("長期休み明け({}日)".format(days), 5, "休み明け")
                ev["axis_ng"].append("長期休養明け")
            elif days >= 90:
                minus("休み明け({}日)".format(days), 3, "休み明け")

    # 規則 10: 斤量
    if h.get("kinryo") and last.get("kinryo"):
        dk = h["kinryo"] - last["kinryo"]
        if dk >= 3:
            minus("斤量増(+{:g}kg)".format(dk), 3, "斤量")
        elif dk >= 2:
            minus("斤量増(+{:g}kg)".format(dk), 2, "斤量")
        elif dk <= -2:
            plus("斤量減({:g}kg)".format(dk), 1)

    # 規則 10: クラス差(昇級・壁)
    now_cls = race.get("cls")
    if now_cls is not None:
        lv_last = last.get("cls")
        if lv_last is not None and last.get("center", center) == center:
            if now_cls > lv_last and (last.get("rank") or 99) == 1:
                minus("昇級初戦", 2, "クラス")
        good = [r.get("cls") for r in scored
                if r.get("cls") is not None and (r["rank"] or 99) <= 3
                and r.get("center", center) == center]
        if scored and len(scored) >= 3 and (not good or max(good) < now_cls - 0.9):
            minus("クラスの壁", 2, "クラス")
        if good and max(good) >= now_cls + 0.9:
            plus("格上実績", 2)

    # 近走不振
    if len(scored) >= 3 and all((r["rank"] or 99) > max(3, (r["n"] or 0) / 2)
                                for r in scored[:3]):
        minus("近走不振", 3, "近走")

    # 馬体重の大きな増減
    if h.get("wdiff") is not None and abs(h["wdiff"]) >= 10:
        minus("馬体重{:+d}kg".format(h["wdiff"]), 1)

    ev["avg"] = round(avg, 1)
    return ev


def pace_of(evs):
    """規則 17: 逃げ馬の数で展開を読む"""
    nige = sum(1 for e in evs if e.get("style") == "逃げ")
    senko = sum(1 for e in evs if e.get("style") == "先行")
    if nige >= 3:
        return {"pace": "ハイ", "nige": nige, "senko": senko,
                "note": "逃げ馬{}頭。競り合って前が苦しく、差し有利".format(nige)}
    if nige == 0:
        return {"pace": "スロー", "nige": 0, "senko": senko,
                "note": "逃げ馬不在。先行馬の前残りに注意"}
    if nige == 1 and senko <= 2:
        return {"pace": "スロー", "nige": nige, "senko": senko,
                "note": "単騎逃げ濃厚。前残り"}
    return {"pace": "平均", "nige": nige, "senko": senko, "note": "平均ペース"}


def apply_race_context(ev, h, race, pace, n, cs=None):
    """展開・枠(脚質とセット)・コース傾向の加減点"""
    st = ev.get("style")
    p = pace["pace"]

    def plus(t, v):
        ev["plus"].append({"t": t, "p": round(v, 1)})

    def minus(t, v, weak=None):
        ev["minus"].append({"t": t, "p": round(-abs(v), 1)})
        if weak and abs(v) >= 2:
            ev["weak"].add(weak)

    # 規則 17・18: 展開と同型
    if p == "ハイ":
        if st == "逃げ":
            minus("同型多数で競り合い", 4, "前崩れ")
        elif st == "先行":
            minus("ハイペースで先行不利", 2, "前崩れ")
        elif st in ("差し", "追込"):
            plus("ハイペースで差し有利", 1.5)
    elif p == "スロー":
        if st == "逃げ":
            plus("単騎逃げ", 3)
        elif st == "先行":
            plus("スローで前残り", 1.5)
        elif st == "追込":
            minus("スローで届かない", 3, "届かず")
    elif st == "逃げ" and pace["nige"] == 2:
        minus("逃げ馬2頭で競り合い", 2, "前崩れ")

    # 規則 14・17: コースの脚質傾向(中央。全コース平均と比べてこのコースで得か損か)
    key = STYLE_MAP.get(st)
    idx = cs["style_idx"].get(key) if cs and key else None
    if idx:
        v = max(-4.0, min(3.0, TH["course_style_k"] * math.log(idx))) * (1.0 if cs["exact"] else 0.5)
        front = key in ("逃げ", "先行")
        if v >= 0.5:
            plus("コース傾向: " + ("前が残りやすい" if front else "差しが届きやすい"), v)
        elif v <= -0.5:
            minus("コース傾向: " + ("前が残りにくい" if front else "差しが届きにくい"), -v,
                  "コース脚質")

    # 規則 19: 枠は脚質とセットで(地方は1枠が基本不利)
    um, wk = h.get("umaban") or 0, h.get("waku") or 0
    wi = cs["waku_idx"].get(wk) if cs and wk and cs.get("waku_idx") else None
    if wi:
        # コースの枠順成績があるときはそれを使う(下の一般則は使わない)。地方は場全体の成績でも使う
        v = max(-2.5, min(2.5, TH["course_waku_k"] * math.log(wi)))
        if v >= 0.5:
            plus("コース傾向: {}枠有利".format(wk), v)
        elif v <= -0.5:
            minus("コース傾向: {}枠不利".format(wk), -v, "枠")
    elif race["center"] == "nar":
        if wk == 1:
            minus("地方の1枠", 2, "枠")
        elif st in ("逃げ", "先行") and n >= 10 and um >= n - 1:
            minus("外枠の先行", 1, "枠")
    else:
        if st in ("逃げ", "先行") and um > n * 0.7 and (race.get("dist") or 0) <= 1600:
            minus("外枠の先行(短距離)", 2, "枠")
        elif st in ("差し", "追込") and um <= 3 and n >= 14:
            minus("内枠で包まれる懸念", 1, "枠")
        elif st in ("逃げ", "先行") and um <= 3:
            plus("内枠の先行", 1)


def finalize(ev):
    """規則 11〜13: 重い減点が重なるほど評価を下げ、消す"""
    heavy = [m for m in ev["minus"] if m["p"] <= -2]
    stack = 0.0
    if len(heavy) >= 2:
        stack = 2.0 * (len(heavy) - 1)
        ev["minus"].append({"t": "負け要因{}つ重複".format(len(heavy)), "p": -stack})
    ev["n_heavy"] = len(heavy)
    ev["score"] = round(ev["ability"] + sum(x["p"] for x in ev["plus"])
                        + sum(x["p"] for x in ev["minus"]), 1)


# ---------------------------------------------------------------- ◎○と判定
def overlap(a, b):
    """規則 8: ◎○が同じ負け方をしそうか"""
    same = set(a["weak"]) & set(b["weak"])
    if a.get("style") and a.get("style") == b.get("style") and a["style"] in ("逃げ", "追込"):
        same.add("両方{}".format(a["style"]))
    return sorted(same)


def pick_marks(evs):
    """◎=崩れにくい最上位、○=◎と弱点が重ならない次点。
    戻り値: (◎, ○, メモ, 残った弱点の重なり)"""
    ok = [e for e in evs if not e["cut"] and not e["axis_ng"]]
    ok.sort(key=lambda e: -e["score"])
    if len(ok) < 2:
        return None, None, [], []
    hon, first = ok[0], ok[1]
    ovl = overlap(hon, first)
    if not ovl:
        return hon, first, [], []
    for c in ok[2:]:
        if c["score"] < first["score"] - TH["alt_taikou"]:
            break
        if not overlap(hon, c):
            return hon, c, ["○を{}番→{}番に変更(◎と弱点「{}」が重なる)".format(
                first["umaban"], c["umaban"], "・".join(ovl))], []
    return hon, first, [], ovl


def predict(race, today=None, use_model=True, probs=None):
    """1レースの予想。買う/見送りの判定と、買い目まで出す。
    use_model … 3着以内の確率モデル(keiba_model)で◎○と買うレースを決める(無ければ手作りの評価)
    probs     … 馬番 -> 3着以内の確率 を外から渡す(過去の日の検証用。先の結果で学んでいない確率)"""
    if isinstance(today, str):
        today = parse_date(today)
    horses = [h for h in race["horses"] if not h.get("scratched")]
    n = len(horses)
    out = {"verdict": "見送り", "reason": "", "horses": [], "bets": [], "points": 0,
           "stake": 0, "notes": []}
    if race.get("surface") == "障" or race.get("venue") == "帯広":
        out["reason"] = "障害・ばんえいは対象外"
        return out

    evs = [evaluate_horse(h, race, today) for h in horses]
    pace = pace_of(evs)
    out["pace"] = pace
    cs = course_of(race)
    if cs:
        out["course"] = course_summary(cs)
    for e, h in zip(evs, horses):
        apply_race_context(e, h, race, pace, n, cs)
        finalize(e)
    # 能力で消すのは「3番手以下の集団」から大きく離れた馬だけ(2強が抜けていても3着は要る)
    srt = sorted((e["score"] for e in evs), reverse=True)
    third = srt[2] if len(srt) >= 3 else (srt[-1] if srt else 0)
    for e in evs:
        e["cut"] = False
        e["cut_reason"] = ""
        if e["n_heavy"] >= TH["cut_neg"]:
            e["cut"], e["cut_reason"] = True, "負け要因が{}つ重なる".format(e["n_heavy"])
        elif e["score"] <= third - TH["cut_gap"]:
            e["cut"], e["cut_reason"] = True, "能力差が大きい"
    by = {e["umaban"]: e for e in evs}

    def pub(e):
        return {"umaban": e["umaban"], "score": e["score"], "ability": e["ability"],
                "style": e.get("style"), "plus": e["plus"], "minus": e["minus"],
                "cut": e["cut"], "cut_reason": e["cut_reason"], "axis_ng": e["axis_ng"],
                "weak": sorted(e["weak"]), "rest_days": e.get("rest_days"),
                "nruns": e["nruns"], "mark": ""}

    res = [pub(e) for e in sorted(evs, key=lambda e: -e["score"])]
    out["horses"] = res
    rb = {r["umaban"]: r for r in res}

    # 規則 1・4: まず買うレースか
    nodata = sum(1 for e in evs if e["nruns"] == 0)
    if n < TH["min_runners"]:
        out["reason"] = "{}頭立て(配当妙味なし)".format(n)
        return out
    if nodata / n > TH["max_nodata"]:
        out["reason"] = "過去走のない馬が多い(新馬・能力不明)"
        return out

    # 3着以内の確率モデル(手作りの評価・加減点も材料にしている)
    live = use_model and probs is None
    if live:
        import keiba_model as KM            # keiba_model は keiba_logic を読むので、ここで読む
        probs = KM.probs(race, today, {"horses": res})
    if probs:
        out = _predict_model(out, res, rb, horses, race, cs, pace, probs)
        if live:
            KM.add_jockey_info(race, out)   # 騎手の得意な競馬場・コース・距離(アプリの表に出す)
        return out

    hon, tai, memo, ovl = pick_marks(evs)
    if not hon:
        out["reason"] = "軸にできる馬がいない"
        return out
    out["notes"].extend(memo)
    rest = [e for e in evs if e is not hon and e is not tai]
    rest_best = max((e["score"] for e in rest), default=0)
    gap = round(min(hon["score"], tai["score"]) - rest_best, 1)
    gap12 = round(hon["score"] - tai["score"], 1)
    out.update({"gap": gap, "gap12": gap12, "honmei": hon["umaban"], "taikou": tai["umaban"]})
    rb[hon["umaban"]]["mark"] = "◎"
    rb[tai["umaban"]]["mark"] = "○"

    # 規則 2・3: 2強だけ買う(1強・混戦は見送り)
    # コースの堅さで基準を少し動かす(上位人気が崩れにくいコースは2強を信じやすい)
    adj = 0.0
    if cs and cs["exact"]:
        if cs["kata"] >= 1.05:
            adj = -TH["kata_adj"]
            out["notes"].append("{}は堅いコース(1・2番人気の3着内率{:.0%})。2強の基準を{}下げた".format(
                cs["name"], cs["fav_top3"], TH["kata_adj"]))
        elif cs["kata"] <= 0.95:
            adj = TH["kata_adj"]
            out["notes"].append("{}は荒れやすいコース(1・2番人気の3着内率{:.0%})。2強の基準を{}上げ、穴を2頭まで拾う".format(
                cs["name"], cs["fav_top3"], TH["kata_adj"]))
    if ovl:
        out["reason"] = "◎○の弱点が重なる({})".format("・".join(ovl))
        verdict = "見送り"
    elif gap >= TH["gap_a"] + adj:
        verdict, out["reason"] = "A", "2強(3番手と{:.1f}差)".format(gap)
    elif gap >= TH["gap_b"] + adj:
        verdict, out["reason"] = "B", "2強(3番手と{:.1f}差)".format(gap)
    elif gap12 >= TH["one_strong"]:
        verdict, out["reason"] = "見送り", "1強(◎だけが抜けている)"
    else:
        verdict, out["reason"] = "見送り", "混戦(上位の差が小さい)"
    out["verdict"] = verdict

    # 規則 9・13: 消せない馬だけを相手に
    alive = [e for e in sorted(rest, key=lambda e: -e["score"]) if not e["cut"]]
    for r in res:
        if r["cut"]:
            r["mark"] = "×"
    aite = [e["umaban"] for e in alive[:TH["aite_max"]]]
    # 規則 27・28: 3頭目に人気薄を混ぜる(オッズは最後に見る)
    ana = []
    ninki = {h["umaban"]: h.get("ninki") for h in horses}
    for e in alive:
        nk = ninki.get(e["umaban"])
        if nk and nk >= TH["ana_ninki"] and e["n_heavy"] <= 1:
            ana.append(e["umaban"])
    ana = ana[:2 if cs and cs["exact"] and cs["kata"] <= 0.95 else 1]
    if ana:
        # 穴は必ず相手に入れる(あふれた分は評価の低い相手から外す)
        aite = [x for x in aite if x not in ana][:TH["aite_max"] - len(ana)] + ana
    for i, u in enumerate(aite):
        rb[u]["mark"] = "☆" if u in ana else ("▲" if i == 0 else "△")
    out["aite"], out["ana"] = aite, ana

    # 規則 5・20: 人気との食い違い(参考)
    for r in res:
        nk = ninki.get(r["umaban"])
        r["ninki"] = nk
        if nk:
            rank = res.index(r) + 1
            if nk <= 3 and rank >= 6:
                r["odds_note"] = "過剰人気?"
            elif nk >= 6 and rank <= 3:
                r["odds_note"] = "妙味"
    hn = ninki.get(hon["umaban"])
    if hn and hn >= 4:
        out["notes"].append("◎が{}番人気(市場の評価と違う。根拠を確認)".format(hn))
    if hn == 1 and hon.get("style") == "逃げ" and pace["nige"] >= 2:
        out["notes"].append("◎は同型の多い人気の逃げ馬。過信禁物")

    if verdict != "見送り":
        out["bets"] = (make_bets_wide(hon["umaban"], tai["umaban"], aite) if TH["bet_mode"].startswith("wide")
                       else make_bets(hon["umaban"], tai["umaban"], aite, verdict, gap12))
        out["points"] = sum(len(b["combos"]) for b in out["bets"])
        out["stake"] = sum(len(b["combos"]) * b["unit"] for b in out["bets"])
    return out


# モデルで買うレースの基準(keiba_model_meta.json の select。無ければこの値)
MODEL_SELECT = {"t2": 0.60, "gap": 0.0, "a": 0.65}
# 利用者が決めた基準。中央と地方は別物なので別々に決める(2026-10-19)。学び直しても変えない。None なら学び直しで選ぶ
#   地方 … ○の確率75%以上(Aは80%以上)で、9頭以上。8頭以下はワイドの配当が安く、月ごとの確率で
#          8頭以下 2〜8月 85%(219R)/ 9月〜 90%(48R)、9頭以上 96%(323R)/ 104%(65R)(2026-10-20)
#   中央 … ◎の確率80%以上(Aは85%以上)。○の確率・3番手との差では絞らない(gap=-1)(2026-10-25 見直し)。
#          2024・2025年で選び、2026年で確かめた。ワイド◎-○は月ごとの確率で 2024年 93% / 2025年 92% / 2026年 97%、
#          的中5割。前の基準(◎60%以上・○40〜50%)は 76% / 82% / 93%。どの年も100%には届いていない
MODEL_SELECT_FIXED = {
    "nar": {"t2": 0.75, "n_min": 9},
    "jra": {"t1": 0.80, "t2": 0.0, "gap": -1.0, "a_p1": 0.85},
}


def model_select(center="nar"):
    """中央(jra)・地方(nar)の買う基準。t1=◎の下限 / t2=○の下限 / t2_max=○の上限 / gap=○と3番手の差 /
    n_min=頭数の下限 / a=Aにする○の下限(a_p1 があれば◎の下限でAにする)"""
    if MODEL_SELECT_FIXED:
        s = dict(MODEL_SELECT, t1=0.0, t2_max=1.01)
        s.update(MODEL_SELECT_FIXED.get(center) or MODEL_SELECT_FIXED["nar"])
        s["a"] = s["t2"] + 0.05
        return s
    try:
        import keiba_model as KM
        _, meta = KM.load()
        sel = (meta or {}).get("select") or {}
        sel = sel.get(center, sel)
        if sel.get("t2") is not None:
            s = dict(MODEL_SELECT, t1=0.0, t2_max=1.01)
            s.update({k: v for k, v in sel.items() if k in ("t1", "t2", "t2_max", "gap", "n_min", "a_p1")})
            s["a"] = s["t2"] + 0.05
            return s
    except Exception:
        pass
    return dict(MODEL_SELECT, t1=0.0, t2_max=1.01)


def select_text(sel):
    """基準の短い説明(◎80%以上 / ◎60%以上・○40〜50% など)"""
    t = []
    if sel.get("t1"):
        t.append("◎{:.0%}以上".format(sel["t1"]))
    if sel.get("t2_max", 1.01) <= 1:
        t.append("○{:.0%}〜{:.0%}".format(sel["t2"], sel["t2_max"]))
    elif sel.get("t2"):
        t.append("○{:.0%}以上".format(sel["t2"]))
    return "・".join(t)


def select_ok(sel, p1, p2, p3, n=99):
    """基準に入るか(n=頭数)"""
    return (p1 >= sel["t1"] and sel["t2"] <= p2 < sel["t2_max"] and p2 - p3 >= sel["gap"]
            and n >= sel.get("n_min", 0))


def _predict_model(out, res, rb, horses, race, cs, pace, probs):
    """確率で◎○と買うレースを決める。
    ◎○ … 軸にできる馬(規則 15 延長などでない)のうち、3着以内の確率の上位2頭
    相手 … 残りの確率上位2頭
    買う … 中央・地方で別の基準(model_select)。地方は○の確率、中央は◎の確率と○の確率の帯で決める
    買い方 … ワイド(TH["bet_mode"]。wide1 は◎-○の1点)"""
    center = "jra" if race.get("center") == "jra" else "nar"
    sel = model_select(center)
    for r in res:
        r["p3"] = round(float(probs.get(r["umaban"], 0.0)), 3)
    res.sort(key=lambda r: -r["p3"])
    out["horses"] = res
    out["model"] = True
    ok = [r for r in res if not r["axis_ng"]]
    if len(res) < 5 or len(ok) < 2:
        out["reason"] = "軸にできる馬がいない"
        return out
    hon, tai = ok[0], ok[1]
    rest = [r for r in res if r is not hon and r is not tai]
    p1, p2, p3 = hon["p3"], tai["p3"], rest[0]["p3"]
    out.update({"honmei": hon["umaban"], "taikou": tai["umaban"], "gap": round(p2 - p3, 3),
                "gap12": round(p1 - p2, 3), "p3": [p1, p2, p3]})
    hon["mark"], tai["mark"] = "◎", "○"
    for r, m in zip(rest[:2], ("▲", "△")):
        r["mark"] = m
    for r in res:
        if r["cut"] and not r["mark"]:
            r["mark"] = "×"
    skipped = [r for r in res[:2] if r["axis_ng"]]
    if skipped:
        out["notes"].append("{}番は確率では上位だが軸にしない({})".format(
            skipped[0]["umaban"], "・".join(skipped[0]["axis_ng"])))
    w = sorted(set(hon["weak"]) & set(tai["weak"]))
    if w:
        out["notes"].append("◎○の弱点が重なる({})。同じ負け方に注意".format("・".join(w)))
    if select_ok(sel, p1, p2, p3, len(horses)):
        a = p1 >= sel["a_p1"] if sel.get("a_p1") else p2 >= sel["a"]
        out["verdict"] = "A" if a else "B"
        out["reason"] = ("◎が堅い(◎の3着内{:.0%}・○{:.0%}。中央の基準 {})".format(p1, p2, select_text(sel)) if center == "jra"
                         else "2強(○の3着内{:.0%}・3番手{:.0%})".format(p2, p3))
    else:
        out["verdict"] = "見送り"
        if len(horses) < sel.get("n_min", 0) and sel["t2"] <= p2:
            out["reason"] = "少頭数({}頭。{}頭以下はワイドの配当が安いので買わない)".format(len(horses), sel["n_min"] - 1)
        elif p1 < sel["t1"]:
            out["reason"] = "◎が基準未満(◎の3着内{:.0%}で基準{:.0%}に届かない)".format(p1, sel["t1"])
        elif p2 >= sel["t2_max"]:
            out["reason"] = "人気どおり(○の3着内{:.0%}。中央は○{:.0%}〜{:.0%}だけ買う)".format(p2, sel["t2"], sel["t2_max"])
        elif p1 >= sel["t2"] and p1 - p2 >= 0.15:
            out["reason"] = "1強(◎{:.0%}・○{:.0%})".format(p1, p2)
        else:
            out["reason"] = "混戦(○の3着内{:.0%}で基準{:.0%}に届かない)".format(p2, sel["t2"])
    # 規則 5・20: 人気は最後に照らすだけ
    ninki = {h["umaban"]: h.get("ninki") for h in horses}
    for i, r in enumerate(res):
        nk = ninki.get(r["umaban"])
        r["ninki"] = nk
        if nk:
            if nk <= 3 and i >= 5:
                r["odds_note"] = "過剰人気?"
            elif nk >= 6 and i <= 2:
                r["odds_note"] = "妙味"
    hn = ninki.get(hon["umaban"])
    if hn and hn >= 4:
        out["notes"].append("◎が{}番人気(市場の評価と違う。根拠を確認)".format(hn))
    out["aite"] = [r["umaban"] for r in rest[:2]]
    out["ana"] = []
    if out["verdict"] != "見送り":
        out["bets"] = make_bets_wide(hon["umaban"], tai["umaban"], out["aite"])
        out["points"] = sum(len(b["combos"]) for b in out["bets"])
        out["stake"] = sum(len(b["combos"]) * b["unit"] for b in out["bets"])
    return out


def make_bets_wide(h, t, aite):
    """ワイドだけ。wide1 は◎-○の1点(2026年の再現で回収率がいちばん高かった買い方)"""
    if TH["bet_mode"] == "wide1":
        return [{"type": "ワイド", "role": "◎○が両方3着内", "kind": "◎-○", "combos": [sorted([h, t])], "unit": UNIT}]
    a = [x for x in aite if x not in (h, t)][:2]
    bets = [{"type": "ワイド", "role": "◎○が両方3着内", "kind": "◎-○", "combos": [sorted([h, t])], "unit": UNIT * 5}]
    if a:
        bets.append({"type": "ワイド", "role": "◎と相手が3着内", "kind": "◎-相手",
                     "combos": [sorted([h, x]) for x in a], "unit": UNIT})
        bets.append({"type": "ワイド", "role": "◎が飛んだときの保険", "kind": "○-相手",
                     "combos": [sorted([t, x]) for x in a], "unit": UNIT})
    return bets


def make_bets(h, t, aite, verdict, gap12):
    """規則 21〜29: ◎○2頭固定を中心に、券種ごとに役割を分ける"""
    bets = []
    a3 = [x for x in aite if x not in (h, t)]
    # 3連複 ◎○2頭軸流し(規則 21・22)
    if a3:
        bets.append({"type": "3連複", "role": "◎○が両方3着内+3頭目",
                     "kind": "◎○2頭軸流し",
                     "combos": [sorted([h, t, x]) for x in a3], "unit": UNIT})
    # 3連単 ◎○の1・2着入れ替え(規則 23)。◎が抜けていれば ◎→○ を厚く
    t3 = a3[:TH["tan3_aite_max"]]
    if t3:
        combos = [[h, t, x] for x in t3] + [[t, h, x] for x in t3]
        bets.append({"type": "3連単", "role": "◎○で1・2着(入れ替えも)",
                     "kind": "◎⇄○→相手", "combos": combos, "unit": UNIT})
    # 馬連・ワイド ◎-○(規則 25・26)
    bets.append({"type": "馬連", "role": "◎○で1・2着", "kind": "◎-○",
                 "combos": [sorted([h, t])], "unit": UNIT * (3 if verdict == "A" else 2)})
    bets.append({"type": "ワイド", "role": "◎○が両方3着内(保険)", "kind": "◎-○",
                 "combos": [sorted([h, t])], "unit": UNIT * 3})
    # ◎が飛んだとき(規則 24): ○-相手上位2頭のワイド
    if a3:
        bets.append({"type": "ワイド", "role": "◎が飛んだときの保険", "kind": "○-相手",
                     "combos": [sorted([t, x]) for x in a3[:2]], "unit": UNIT})
    return bets


# ---------------------------------------------------------------- 結果の照合
def settle(pred, result):
    """買い目と結果(着順・払戻)を照らし合わせる。払戻が取れなくても的中は着順で判定。"""
    order = result.get("order") or []
    pay = result.get("payout") or {}
    if len(order) < 3:
        return None
    top2, top3 = set(order[:2]), set(order[:3])
    ret = 0
    for b in pred.get("bets", []):
        b["hit"] = []
        b["ret"] = 0
        paid = pay.get(b["type"]) or {}
        for c in b["combos"]:
            key = "-".join(str(x) for x in (c if b["type"] in ("3連単", "馬単") else sorted(c)))
            hit = False
            if paid:
                # 払戻表があればそれで判定(同着で2組に払戻が出ることがある)
                hit = key in paid
            elif b["type"] == "3連複":
                hit = set(c) == top3
            elif b["type"] == "3連単":
                hit = list(c) == order[:3]
            elif b["type"] == "馬連":
                hit = set(c) == top2
            elif b["type"] == "馬単":
                hit = list(c) == order[:2]
            elif b["type"] == "ワイド":
                hit = set(c) <= top3
            elif b["type"] == "単勝":
                hit = c[0] == order[0]
            if hit:
                yen = paid.get(key)
                b["hit"].append(key)
                if yen:
                    r = int(yen * b["unit"] / 100)
                    b["ret"] += r
                    ret += r
    return {"stake": pred.get("stake", 0), "ret": ret,
            "hit": any(b["hit"] for b in pred.get("bets", [])),
            "honmei_rank": rank_of(order, pred.get("honmei")),
            "taikou_rank": rank_of(order, pred.get("taikou"))}


def rank_of(order, u):
    try:
        return order.index(u) + 1
    except ValueError:
        return None

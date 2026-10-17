#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""競馬場の特徴の記事(read_site.py で読んだ本文)から、コースごとの傾向を取り出して JSON にする。
   python tools/parse_course.py 出力.json 記事1.txt 記事2.txt ...
取り出すもの(2021〜2025年の集計):
  text  … 脚質傾向・枠順傾向・ペース傾向・求められる能力・向いている馬・不向きな馬・最初のコーナーまで・直線・坂
  waku  … 枠番別 [出走, 1着, 3着内]
  style … 脚質別(逃げ・先行・中団・後方・マクリ) [出走, 1着, 3着内]
  ninki … 人気別 [出走, 1着, 3着内]"""
import json
import re
import sys
import unicodedata

TEXT_KEYS = ["コース形態", "スタート地点", "最初のコーナーまで", "直線の長さ", "坂", "脚質傾向", "枠順傾向",
             "ペース傾向", "求められる能力", "向いている馬", "不向きな馬"]
STYLE = {"逃げ": "逃げ", "先行": "先行", "中団": "中団", "後方": "後方", "マクリ": "マクリ"}


def nf(s):
    return unicodedata.normalize("NFKC", s).strip()


def counts(s):
    """'23-21-23-322 / 389' -> [出走, 1着, 3着内]"""
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*/\s*(\d+)", nf(s).replace(",", ""))
    if not m:
        return None
    a, b, c, d = (int(x) for x in m.groups()[:4])
    return [a + b + c + d, a, a + b + c]


def parse(path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    url = next((l.split(" ", 2)[2] for l in lines if l.startswith("##### 記事")), "")
    courses, cur, table = [], None, None
    for l in lines:
        t = nf(l)
        m = re.match(r"^## (.+?)競馬場\s*(芝|ダート|障害)\s*(\d{3,4})\s*m\s*コースの特徴", t)
        if m:
            cur = {"venue": m.group(1), "surface": {"芝": "芝", "ダート": "ダ", "障害": "障"}[m.group(2)],
                   "dist": int(m.group(3)), "text": {}, "waku": {}, "style": {}, "ninki": {}, "url": url}
            courses.append(cur)
            table = None
            continue
        if t.startswith("## "):
            cur, table = None, None
            continue
        if not cur or not t.startswith("|"):
            if t.startswith("###"):
                table = None
            continue
        cells = [nf(c) for c in t.strip("|").split("|")]
        head = cells[0]
        # コースの特徴の表(「脚質傾向」は脚質別の表の見出しと紛れるので先に見る)
        if head in TEXT_KEYS and len(cells) >= 2 and not counts(cells[1]):
            cur["text"][head] = cells[1]
            table = None
            continue
        # 表の見出し行
        if head in ("枠番", "枠"):
            table = "waku"
            continue
        if head == "人気":
            table = "ninki"
            continue
        if head.startswith("脚質"):
            table = "style"
            continue
        if head in ("騎手", "調教師", "種牡馬"):
            table = "skip"
            continue
        if table in (None, "skip") or len(cells) < 2:
            continue
        c = counts(cells[1])
        if not c:
            continue
        if table == "waku":
            m = re.match(r"(\d)枠", head)
            if m:
                cur["waku"][m.group(1)] = c
        elif table == "ninki":
            m = re.match(r"(\d+)番人気", head)
            if m:
                cur["ninki"][m.group(1)] = c
        elif table == "style":
            key = head.split("・")[-1].replace("ﾏｸﾘ", "マクリ")
            key = unicodedata.normalize("NFKC", key)
            if key in STYLE:
                cur["style"][STYLE[key]] = c
    return courses


if __name__ == "__main__":
    out, allc = sys.argv[1], []
    for p in sys.argv[2:]:
        cs = parse(p)
        allc += cs
        for c in cs:
            print("{}{}{} 枠{} 脚質{} 人気{} 文{}".format(c["venue"], c["surface"], c["dist"], len(c["waku"]),
                                                     len(c["style"]), len(c["ninki"]), len(c["text"])))
    json.dump({"source": "titanic-online.com 競馬場の特徴(2021〜2025年の集計)", "courses": allc},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(allc), "コース")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""モデルの◎○が市場(1・2番人気)と食い違うレースを、食い違う理由ごとに分けてワイド◎-○の成績を見る。
   KEIBA_CACHE=... python tools/check_disagree.py
確率は keiba/model_wf.json.gz(その月より前だけで学んだモデル)を使う。特徴量は train_model のキャッシュから読む。
理由: タイム … 馬場差で補正したタイム指数(3走平均)が人気馬より2以上上
      上がり … 上がりの指数が人気馬より3以上上
      減点   … 人気馬に、延長・昇級・休み明け・斤量増・コースの脚質が合わない、のどれか
2026年10月に試した結果は README の「市場と食い違う馬」。"""
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keiba as K            # noqa: E402
import train_model as T      # noqa: E402

PERIODS = [("2〜8月", "20260201", "20260901"), ("9月〜", "20260901", "20990101")]


def neg(fy):
    out = []
    if (fy.get("dist_change") or 0) >= 200:
        out.append("延長")
    if (fy.get("cls_diff_last") or 0) > 0:
        out.append("昇級")
    if (fy.get("rest_days") or 0) >= 120:
        out.append("休み明け")
    if (fy.get("kinryo_change") or 0) >= 2:
        out.append("斤量増")
    if fy.get("course_style_idx") is not None and fy["course_style_idx"] <= -0.15:
        out.append("脚質")
    return out


def diff(fx, fy, k):
    return fx[k] - fy[k] if fx.get(k) is not None and fy.get(k) is not None else None


def reasons(fx, fy):
    r = set()
    if (diff(fx, fy, "spd_mean3") or -9) >= 2:
        r.add("タイム")
    if (diff(fx, fy, "l3fig_best") or -9) >= 3:
        r.add("上がり")
    if neg(fy):
        r.add("減点")
    return r


def main():
    t2 = float(os.environ.get("T2", "0.6"))
    with gzip.open(os.path.join(K.OUT, "model_wf.json.gz"), "rt", encoding="utf-8") as f:
        wf = json.load(f)["probs"]
    _, rows, _, _ = pickle.load(open(T.CACHE, "rb"))
    rb = {rc["id"]: rc for _, rc in T.load_races()}
    acc = defaultdict(lambda: [0, 0, 0])

    def add(key, per, sel, pay):
        y = pay.get("-".join(str(x) for x in sorted(sel)))
        a = acc[(key, per)]
        a[0] += 1
        a[1] += 1 if y else 0
        a[2] += y or 0

    for rid, rs in T.group(rows).items():
        pw = wf.get(rid)
        if not pw:
            continue
        d = rs[0]["date"]
        per = next((n for n, lo, hi in PERIODS if lo <= d < hi), None)
        if not per:
            continue
        o = sorted(rs, key=lambda r: -pw.get(str(r["u"]), 0))
        ok = [r for r in o if not r["f"].get("rule_axis_ng")]
        mk = sorted((r for r in rs if r["ninki"]), key=lambda r: r["ninki"])
        if len(o) < 5 or len(ok) < 2 or len(mk) < 4:
            continue
        h, t = ok[0], ok[1]
        if pw.get(str(t["u"]), 0) < t2:
            continue
        pay = (rb[rid]["result"].get("payout") or {}).get("ワイド") or {}
        mset = {mk[0]["u"], mk[1]["u"]}
        X = [r for r in (h, t) if r["u"] not in mset]
        Y = [r for r in mk[:2] if r["u"] not in (h["u"], t["u"])]
        add("モデル", per, (h["u"], t["u"]), pay)
        add("市場(1・2番人気)", per, (mk[0]["u"], mk[1]["u"]), pay)
        if not X:
            add("一致したレース", per, (h["u"], t["u"]), pay)
            continue
        add("食い違うレース", per, (h["u"], t["u"]), pay)
        add("食い違うレースを市場で買う", per, (mk[0]["u"], mk[1]["u"]), pay)
        rsn = [reasons(x["f"], y["f"]) for x, y in zip(X, Y)]
        common = set.intersection(*rsn)
        if not all(rsn):            # 食い違う馬のどれかに理由が無い
            add("食い違い 理由なし", per, (h["u"], t["u"]), pay)
        else:
            add("食い違い 理由あり", per, (h["u"], t["u"]), pay)
            for k in sorted(common):
                add("食い違い 理由: " + k, per, (h["u"], t["u"]), pay)
            if len(set.union(*rsn)) >= 2:
                add("食い違い 理由2つ以上", per, (h["u"], t["u"]), pay)
    keys = []
    for k, _ in acc:
        if k not in keys:
            keys.append(k)
    print("○の確率 {} 以上のレース・ワイド◎-○ 100円".format(t2))
    for k in keys:
        cells = []
        for p, _, _ in PERIODS:
            n, hit, ret = acc[(k, p)]
            cells.append("{} {:>5}R 的中{:>3.0f}% 回収率{:>4.0f}%".format(p, n, hit / n * 100 if n else 0,
                                                                     ret / n if n else 0))
        print("  {:<16} ".format(k) + " | ".join(cells))


if __name__ == "__main__":
    main()

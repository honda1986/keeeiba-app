#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""keiba_logic / keiba のパースの確認。  python -m unittest test_keiba"""

import json
import os
import re
import unittest

import keiba as K
import keiba_logic as L


def run(rank, n=14, margin=None, dist=1600, surface="芝", baba="良", venue="東京",
        first=6, date="2026-09-20", kinryo=57.0, name="2勝クラス"):
    return {"date": date, "venue": venue, "rank": rank, "n": n, "margin": margin,
            "dist": dist, "surface": surface, "baba": baba, "pass": [first, first],
            "kinryo": kinryo, "race_name": name, "cls": L.class_level(name)}


def horse(u, runs, waku=None, kinryo=57.0, ninki=None):
    return {"umaban": u, "waku": waku or (u + 1) // 2, "name": "馬{}".format(u),
            "kinryo": kinryo, "past": runs, "ninki": ninki}


def race(horses, **kw):
    r = {"id": "202605040811", "center": "jra", "venue": "東京", "rno": 11,
         "surface": "芝", "dist": 1600, "direction": "左", "baba": "良",
         "cls": 3, "horses": horses}
    r.update(kw)
    return r


def strong(u, first=6, **kw):
    return horse(u, [run(1, margin=-0.4, first=first), run(1, margin=-0.3, first=first),
                     run(2, margin=0.1, first=first)], **kw)


def middle(u, first=8, m=0.9, **kw):
    return horse(u, [run(5, margin=m, first=first), run(6, margin=m + 0.2, first=first),
                     run(4, margin=m - 0.1, first=first)], **kw)


TODAY = "2026-10-18"


class Logic(unittest.TestCase):
    def test_two_strong_is_bought(self):
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        mode, L.TH["bet_mode"] = L.TH["bet_mode"], "full"
        try:
            p = L.predict(race(hs), TODAY, use_model=False)
        finally:
            L.TH["bet_mode"] = mode
        self.assertIn(p["verdict"], ("A", "B"), p["reason"])
        self.assertEqual({p["honmei"], p["taikou"]}, {1, 2})
        types = [b["type"] for b in p["bets"]]
        for t in ("3連複", "3連単", "馬連", "ワイド"):
            self.assertIn(t, types)
        # 3連複は ◎○ が全部の組に入る
        for c in p["bets"][0]["combos"]:
            self.assertTrue({1, 2} <= set(c))
        # 3連単は ◎→○ と ○→◎ の両方
        t3 = next(b for b in p["bets"] if b["type"] == "3連単")["combos"]
        self.assertTrue(any(c[:2] == [p["honmei"], p["taikou"]] for c in t3))
        self.assertTrue(any(c[:2] == [p["taikou"], p["honmei"]] for c in t3))
        self.assertLessEqual(p["points"], 20)

    def test_wide_only_bets(self):
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)
        self.assertEqual({b["type"] for b in p["bets"]}, {"ワイド"})
        # ワイド1点: ◎-○を100円だけ
        self.assertEqual([(b["combos"], b["unit"]) for b in p["bets"]], [([sorted([p["honmei"], p["taikou"]])], 100)])
        self.assertEqual(p["stake"], 100)
        mode, L.TH["bet_mode"] = L.TH["bet_mode"], "wide"
        try:
            p = L.predict(race(hs), TODAY, use_model=False)
        finally:
            L.TH["bet_mode"] = mode
        self.assertEqual((p["bets"][0]["unit"], p["stake"]), (500, 900))

    def test_model_mode(self):
        # 確率を外から渡す: ◎○は軸にできる馬の上位2頭。買う基準は中央と地方で別
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        probs = {u: 0.1 for u in range(1, 13)}
        # 地方: ○の確率が基準以上なら買う
        nar = race(hs, center="nar")
        sel = L.model_select("nar")       # 基準は学び直しで変わるので、今の基準で A になる確率にする
        probs.update({1: 0.95, 2: min(0.94, sel["a"] + 0.02), 3: 0.40, 4: 0.30})
        p = L.predict(nar, TODAY, probs=probs)
        self.assertTrue(p["model"])
        self.assertEqual((p["honmei"], p["taikou"], p["aite"]), (1, 2, [3, 4]))
        self.assertEqual(p["verdict"], "A")
        self.assertEqual({b["type"] for b in p["bets"]}, {"ワイド"})
        small = race(hs[:sel["n_min"] - 1], center="nar")   # 少頭数は確率が高くても買わない
        p = L.predict(small, TODAY, probs=probs)
        self.assertEqual(p["verdict"], "見送り")
        self.assertIn("少頭数", p["reason"])
        probs[2] = 0.45                   # ○が基準に届かない → 見送り(1強)
        p = L.predict(nar, TODAY, probs=probs)
        self.assertEqual(p["verdict"], "見送り")
        self.assertIn("1強", p["reason"])
        # 中央: ◎の下限(○の帯があればその中)だけ買う
        jra = race(hs)
        sj = L.model_select("jra")
        mid = (sj["t2"] + min(sj["t2_max"], 1.0)) / 2
        probs.update({1: 0.95, 2: mid, 3: 0.20, 4: 0.15})
        p = L.predict(jra, TODAY, probs=probs)
        self.assertEqual(p["verdict"], "A" if 0.95 >= sj.get("a_p1", 1) else "B")
        # 中央の買い方: 3連複◎-○-▲の1点(▲=◎○の次に確率が高い馬)
        self.assertEqual([(b["type"], b["combos"], b["unit"]) for b in p["bets"]], [("3連複", [[1, 2, 3]], 100)])
        self.assertEqual(p["stake"], 100)
        st = L.settle(p, {"order": [3, 1, 2], "payout": {"3連複": {"1-2-3": 780}}})
        self.assertEqual((st["hit"], st["ret"]), (True, 780))
        self.assertFalse(L.settle(p, {"order": [1, 2, 4], "payout": {"3連複": {"1-2-4": 500}}})["hit"])
        # ▲が距離延長(軸不可)なら、3連複ではなくワイド◎-○の1点
        ext3 = horse(3, [run(1, margin=-0.6, dist=1200)] * 3)
        p = L.predict(race([strong(1, first=2), strong(2, first=9), ext3] + [middle(u) for u in range(4, 13)]), TODAY, probs=probs)
        self.assertIn("距離延長", next(h for h in p["horses"] if h["umaban"] == 3)["axis_ng"])
        self.assertEqual([(b["type"], b["combos"]) for b in p["bets"]], [("ワイド", [[1, 2]])])
        self.assertTrue(any("ワイド◎-○" in n for n in p["notes"]))
        if sj["t2_max"] <= 1:
            probs[2] = min(0.94, sj["t2_max"] + 0.05)   # ○が帯より強い → 見送り
            self.assertEqual(L.predict(jra, TODAY, probs=probs)["verdict"], "見送り")
        if sj.get("g12", -1) > 0:                     # ◎と○が近い → 見送り(人気どおりで配当が安い)
            probs.update({1: 0.95, 2: 0.95 - sj["g12"] + 0.01})
            p = L.predict(jra, TODAY, probs=probs)
            self.assertEqual(p["verdict"], "見送り")
            self.assertIn("◎と○が近い", p["reason"])
        probs.update({1: max(0.0, sj["t1"] - 0.05), 2: mid - 0.06})   # ◎が下限に届かない → 見送り
        self.assertEqual(L.predict(jra, TODAY, probs=probs)["verdict"], "見送り")
        # 延長の馬は確率が高くても軸にしない
        ext = horse(5, [run(1, margin=-0.6, dist=1200)] * 3)
        r2 = race([ext] + [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in (3, 4, 6, 7, 8, 9, 10)])
        probs = {u: 0.1 for u in range(1, 11)}
        probs.update({5: 0.9, 1: 0.8, 2: 0.7})
        p = L.predict(r2, TODAY, probs=probs)
        self.assertEqual((p["honmei"], p["taikou"]), (1, 2))
        self.assertIn(5, p["aite"])
        self.assertIn("5番(距離延長)は確率では上位だが軸にしない。○2番は確率3位", p["notes"])

    def test_predict_only(self):
        # 高知の最終レースのアプリ: 確率と印は出すが、買わない
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        probs = {u: 0.1 for u in range(1, 13)}
        probs.update({1: 0.95, 2: 0.5, 3: 0.4, 4: 0.3})
        old = L.PREDICT_ONLY
        try:
            L.PREDICT_ONLY = True
            p = L.predict(race(hs), TODAY, probs=probs)
        finally:
            L.PREDICT_ONLY = old
        self.assertEqual((p["verdict"], p["reason"], p["bets"]), ("見送り", L.PREDICT_ONLY_REASON, []))
        self.assertEqual((p["honmei"], p["taikou"], p["aite"][0]), (1, 2, 3))

    def test_final_model(self):
        # 高知の最終レース用: 単勝オッズの確率が強く効く式なら、印は人気の順に付け直す。オッズが無ければそのまま
        import keiba_final as F
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        rc = race(hs, center="nar")
        p = L.predict(rc, TODAY, probs={u: 0.6 - u * 0.04 for u in range(1, 13)})
        model = {"feats": F.FEATS, "mu": [0.0] * len(F.FEATS), "sd": [1.0] * len(F.FEATS),
                 "w": [0.0] + [3.0 if k == "lmt3" else 0.0 for k in F.FEATS]}
        self.assertFalse(F.apply(rc, p, model, {}))                       # オッズが無い
        odds = {u: 2.0 + abs(u - 9) * 3 for u in range(1, 13)}            # 9番が1番人気、8・10番が次
        self.assertTrue(F.apply(rc, p, model, odds))
        marks = {h["umaban"]: h["mark"] for h in p["horses"] if h["mark"]}
        self.assertEqual(marks[9], "◎")
        self.assertEqual(p["honmei"], 9)
        self.assertLessEqual(sum(h["p3"] for h in p["horses"]), 3.01)        # レースで合計3(1頭は97%まで)
        self.assertTrue(all("p3_model" in h for h in p["horses"]))
        # 荒れ度: 1番人気のオッズが高いほど、1番人気が3着に来ない確率が高い
        model["chaos"] = {"mu": [1.0, 0.9], "sd": [0.5, 0.05], "w": [0.0, 0.8, 0.3]}
        low = F.chaos(model, {1: 1.3, 2: 8.0, 3: 12.0, 4: 20.0, 5: 30.0})
        high = F.chaos(model, {1: 4.5, 2: 5.0, 3: 6.0, 4: 7.0, 5: 8.0})
        self.assertLess(low, high)
        self.assertIsNone(F.chaos(model, {1: 2.0}))

    def test_chaos_model_jra(self):
        # 中央の荒れ度の式(keiba/chaos_model.json。アプリが出す): 1番人気が堅いほど低く、段の分け方はアプリと同じ
        import keiba_final as F
        m = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba", "chaos_model.json"), encoding="utf-8"))
        low = F.chaos(m, {1: 1.3, 2: 8.0, 3: 12.0, 4: 20.0, 5: 30.0, 6: 50.0})
        high = F.chaos(m, {1: 4.5, 2: 5.0, 3: 6.0, 4: 7.0, 5: 8.0, 6: 9.0})
        self.assertLess(low, 0.3)
        self.assertGreater(high, 0.5)
        self.assertEqual([F.chaos_level(v) for v in (0.2, 0.4, 0.55)], ["低", "中", "高"])
        self.assertEqual(len(m["chaos"]["bands"]), 5)
        # 後押しも使う荒れ度(chaos_push。2026-11-26): 1番人気の後押し(2頭とも3着内の式)が小さい
        # (過去の成績が良く見えて買われすぎ)ほど荒れる。1番人気に後押しが無ければ出さない(アプリはオッズだけの式に戻す)。
        # 人気は (オッズ, 馬番) の順
        if m.get("chaos_push"):
            od = {1: 2.0, 2: 4.0, 3: 6.0, 4: 9.0, 5: 15.0, 6: 30.0}
            x = F.chaos_push_x(od, {1: 0.5, 2: 0.0}, {1: 0.2})
            self.assertEqual(len(x), len(m["chaos_push"]["x"]))
            self.assertAlmostEqual(x[2], __import__("math").log(4.0))
            self.assertEqual(x[3], 6.0)
            overbet = F.chaos_push(m, od, {1: 0.0}, {1: -0.5})
            underbet = F.chaos_push(m, od, {1: 0.0}, {1: 0.5})
            self.assertGreater(overbet, underbet)
            self.assertIsNone(F.chaos_push(m, od, {2: 1.0}, {2: 0.5}))
            self.assertEqual(F.chaos_push_x({1: 2.0, 2: 2.0, 3: 6.0, 4: 9.0, 5: 15.0}, {1: 9.0, 2: -9.0}, {1: 9.0, 2: -9.0})[4], 9.0)   # 同じオッズは馬番の小さい方
            self.assertEqual(len(m["chaos_push"]["bands"]), 5)
            self.assertGreaterEqual(m["chaos_push"]["auc"], m["chaos_push"]["auc_odds"])
        # 式の作り方: はっきり分かれる材料なら係数がプラス
        import numpy as np
        X = np.array([[x] for x in np.linspace(-2, 2, 200)]); y = (X[:, 0] > 0).astype(float)
        self.assertGreater(F.fit_logit(X, y)[1], 1.0)

    def test_muddle_is_skipped(self):
        hs = [middle(u, m=0.5 + u * 0.02) for u in range(1, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)
        self.assertEqual(p["verdict"], "見送り")
        self.assertEqual(p["bets"], [])

    def test_one_strong_is_skipped(self):
        hs = [strong(1, first=2)] + [middle(u, m=0.9 + u * 0.01) for u in range(2, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)
        self.assertEqual(p["verdict"], "見送り")

    def test_extension_is_not_axis(self):
        ext = horse(1, [run(1, margin=-0.6, dist=1200), run(1, margin=-0.5, dist=1200),
                        run(1, margin=-0.5, dist=1200)])
        hs = [ext, strong(2, first=2), strong(3, first=9)] + [middle(u) for u in range(4, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)
        self.assertNotIn(1, (p.get("honmei"), p.get("taikou")))
        h1 = next(h for h in p["horses"] if h["umaban"] == 1)
        self.assertIn("距離延長", h1["axis_ng"])

    def test_same_weakness_changes_taikou(self):
        # ◎も次点も逃げ馬 → ○は脚質の違う馬に替える
        hs = [strong(1, first=1), strong(2, first=1), strong(3, first=9)] + \
             [middle(u) for u in range(4, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)
        styles = {h["umaban"]: h["style"] for h in p["horses"]}
        self.assertFalse(styles[p["honmei"]] == styles[p["taikou"]] == "逃げ")

    def test_nar_waku1_penalty(self):
        hs = [strong(1, first=6, waku=1)] + [middle(u) for u in range(2, 11)]
        p = L.predict(race(hs, center="nar", venue="大井", direction="右", cls=3.0), TODAY, use_model=False)
        h1 = next(h for h in p["horses"] if h["umaban"] == 1)
        self.assertTrue(any("1枠" in m["t"] for m in h1["minus"]))

    def test_stacked_losses_are_cut(self):
        bad = horse(5, [run(9, margin=1.5, date="2026-03-01", dist=1200, baba="重"),
                        run(10, margin=1.8, dist=1200), run(9, margin=1.6, dist=1200)],
                    kinryo=60.0)
        hs = [strong(1, first=2), strong(2, first=9), bad] + \
             [middle(u) for u in (3, 4, 6, 7, 8, 9, 10)]
        p = L.predict(race(hs, baba="重"), TODAY, use_model=False)
        h5 = next(h for h in p["horses"] if h["umaban"] == 5)
        self.assertTrue(h5["cut"])
        self.assertEqual(h5["mark"], "×")

    def test_ana_in_third(self):
        hs = [strong(1, first=2, ninki=1), strong(2, first=9, ninki=2)] + \
             [middle(u, m=0.8 + u * 0.03, ninki=u) for u in range(3, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)
        if p["verdict"] != "見送り":
            self.assertTrue(p["ana"])
            self.assertIn(p["ana"][0], p["aite"])

    def test_course_style(self):
        # 函館ダ1700は前が残る(先行はダート平均の1.18倍、後方は半分ほど)
        hs = [strong(1, first=3), strong(2, first=12)] + [middle(u) for u in range(3, 13)]
        p = L.predict(race(hs, venue="函館", surface="ダ", dist=1700, direction="右", cls=2), TODAY, use_model=False)
        self.assertEqual(p["course"]["name"], "函館ダ1700")
        h = {x["umaban"]: x for x in p["horses"]}
        self.assertTrue(any("前が残りやすい" in m["t"] for m in h[1]["plus"]))
        self.assertTrue(any("差しが届きにくい" in m["t"] for m in h[2]["minus"]))
        self.assertIn("コース脚質", h[2]["weak"])

    def test_course_waku(self):
        # 新潟芝1000(千直)は外枠有利・内枠不利
        hs = [horse(u, [run(3, margin=0.3, dist=1000, venue="新潟", first=5)] * 3, waku=(u + 1) // 2)
              for u in range(1, 17)]
        p = L.predict(race(hs, venue="新潟", dist=1000, direction="直", cls=2), TODAY, use_model=False)
        h = {x["umaban"]: x for x in p["horses"]}
        self.assertTrue(any("8枠有利" in m["t"] for m in h[16]["plus"]))
        self.assertTrue(any("1枠不利" in m["t"] for m in h[1]["minus"]))

    def test_course_only_jra(self):
        hs = [strong(1), strong(2)] + [middle(u) for u in range(3, 11)]
        p = L.predict(race(hs, center="nar", venue="大井", direction="右", cls=3.0), TODAY, use_model=False)
        self.assertNotIn("course", p)

    def test_course_fallback(self):
        # 東京芝2300は記事に無い → 東京芝の場平均(脚質だけ半分の重み、枠は使わない)
        cs = L.course_of({"center": "jra", "venue": "東京", "surface": "芝", "dist": 2300})
        self.assertFalse(cs["exact"])
        self.assertEqual(cs["waku_idx"], {})

    def test_course_kata(self):
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        p = L.predict(race(hs), TODAY, use_model=False)                       # 東京芝1600は堅い
        self.assertTrue(any("堅いコース" in n for n in p["notes"]))
        p = L.predict(race(hs, venue="京都", dist=1400, direction="右"), TODAY, use_model=False)  # 京都芝1400は荒れやすい
        self.assertTrue(any("荒れやすいコース" in n for n in p["notes"]))

    def test_settle(self):
        pred = {"honmei": 1, "taikou": 2, "stake": 0, "bets": L.make_bets(1, 2, [3, 4], "A", 2)}
        pred["stake"] = sum(len(b["combos"]) * b["unit"] for b in pred["bets"])
        res = {"order": [2, 1, 4, 3],
               "payout": {"3連複": {"1-2-4": 1500}, "3連単": {"2-1-4": 6000},
                          "馬連": {"1-2": 400}, "ワイド": {"1-2": 200, "1-4": 500, "2-4": 600}}}
        s = L.settle(pred, res)
        self.assertTrue(s["hit"])
        # 3連複1500 + 3連単6000 + 馬連400*3 + ワイド200*3 + 保険ワイド(2-4)600
        self.assertEqual(s["ret"], 1500 + 6000 + 1200 + 600 + 600)
        self.assertEqual(s["honmei_rank"], 2)

    def test_settle_dead_heat(self):
        # 2着同着: 馬連 5-7 と 5-10 の両方に払戻
        pred = {"honmei": 5, "taikou": 10, "bets": [
            {"type": "馬連", "role": "", "kind": "", "combos": [[5, 10]], "unit": 100}], "stake": 100}
        res = {"order": [5, 7, 10], "payout": {"馬連": {"5-7": 3200, "5-10": 2020}}}
        s = L.settle(pred, res)
        self.assertTrue(s["hit"])
        self.assertEqual(s["ret"], 2020)

    def test_class_level(self):
        self.assertEqual(L.class_level("毎日王冠(GⅡ)"), 7)
        self.assertEqual(L.class_level("3歳以上2勝クラス"), 3)
        self.assertEqual(L.class_level("3歳未勝利"), 1)
        self.assertEqual(L.class_level("C2五六", "nar"), 1.75)
        self.assertEqual(L.class_level("東京大賞典(GI)", "nar"), 8)


PAST_HTML = """<table class="Shutuba_Past5_Table"><tr class="HorseList">
<td class="Waku2 Txt_C">2</td><td class="Waku Txt_C">3</td><td>x</td>
<td class="Past Ranking_1"><div class="Data_Item">
 <div class="Data01"><span>2026.09.20 中山</span><span class="Num">1</span></div>
 <div class="Data02"><a>セントライト記念(GII)</a></div>
 <div class="Data05">芝2200 2:11.8 良</div>
 <div class="Data03">15頭 7番 2人 ルメール 57.0</div>
 <div class="Data06">3-3-2-2 (34.2) 486(+4)</div>
 <div class="Data07">ライバル(-0.2)</div></div></td>
<td class="Rest">休養</td></tr></table>"""

RESULT_HTML = """<table class="RaceTable01"><tr class="HorseList"><td><div class="Rank">1</div></td>
<td>2</td><td>3</td><td class="Odds Txt_C">8</td><td class="Odds Txt_R">15.8</td></tr><tr class="HorseList"><td>2</td><td>1</td><td>1</td></tr>
<tr class="HorseList"><td>3</td><td>5</td><td>9</td></tr></table>
<table class="Payout_Detail_Table"><tr class="Fukusho"><th>複勝</th><td class="Result"><div><span>3</span></div>
<div><span>1</span></div><div><span>9</span></div></td><td class="Payout"><span>170円<br>120円<br>310円</span></td></tr>
<tr class="Wakuren"><th>枠連</th><td class="Result"><ul><li><span>1</span></li><li><span>2</span></li></ul></td>
<td class="Payout"><span>560円</span></td></tr>
<tr class="Umaren"><th>馬連</th>
<td class="Result"><ul><li><span>1</span></li><li><span>3</span></li></ul></td>
<td class="Payout"><span>1,230円</span></td></tr>
<tr class="Wide"><th>ワイド</th><td class="Result"><ul><li><span>1</span></li><li><span>3</span></li></ul>
<ul><li><span>3</span></li><li><span>9</span></li></ul><ul><li><span>1</span></li><li><span>9</span></li></ul></td>
<td class="Payout"><span>400円<br>900円<br>700円</span></td></tr>
<tr class="Umatan"><th>馬単</th><td class="Result"><ul><li><span>3</span></li><li><span>1</span></li></ul></td>
<td class="Payout"><span>2,560円</span></td></tr>
<tr class="Tan3"><th>3連単</th><td class="Result"><ul><li><span>3</span></li><li><span>1</span></li><li><span>9</span></li></ul></td>
<td class="Payout"><span>12,340円</span></td></tr></table>"""


ODDS_JRA = """<table><tr><th>枠</th></tr>
<tr><td class="Waku1 W31">1</td><td class="W31">1</td><td class="Mark_User"></td><td class="Horse_Select"></td>
<td class="Horse_Name">シーラーステート</td><td class="Odds Popular">12.5</td></tr>
<tr><td class="Waku1 W31">1</td><td class="W31">2</td><td class="Mark_User"></td><td class="Horse_Select"></td>
<td class="Horse_Name">ブルーリボン</td><td class="Odds Popular">3.1</td></tr></table>"""

ODDS_NAR = """<table><tr><th>人気</th></tr>
<tr><td class="Ninki">1</td><td class="Waku7">8</td><td class="Mark_User"></td>
<td class="Horse_Name">エイシンシームレス</td><td class="Odds">2.0</td></tr>
<tr><td class="Ninki">2</td><td class="Waku4">4</td><td class="Mark_User"></td>
<td class="Horse_Name">スティールシャトル</td><td class="Odds">3.0</td></tr></table>"""

ODDS_BEFORE = ODDS_JRA.replace("12.5", "---.-").replace("3.1", "---.-")


RESULT_NAR = """<table class="RaceTable01 ResultMain"><tr><th>着<br>順</th><th>枠</th><th>馬<br>番</th><th>馬名</th>
<th>人<br>気</th><th>単勝<br>オッズ</th></tr>
<tr><td>1</td><td>4</td><td>5</td><td>A</td><td>3</td><td>6.2</td></tr>
<tr><td>2</td><td>1</td><td>1</td><td>B</td><td>1</td><td>1.8</td></tr>
<tr><td>3</td><td>6</td><td>8</td><td>C</td><td>7</td><td>25.0</td></tr>
<tr><td>取消</td><td>2</td><td>2</td><td>D</td><td></td><td></td></tr></table>"""


class Parse(unittest.TestCase):
    def test_result_nar(self):
        r = K.parse_result(RESULT_NAR)
        self.assertEqual(r["order"], [5, 1, 8])
        self.assertEqual((r["ninki"]["8"], r["odds"]["8"]), (7, 25.0))

    def test_odds(self):
        self.assertEqual(K.parse_odds_form(ODDS_JRA), {1: (12.5, None), 2: (3.1, None)})
        self.assertEqual(K.parse_odds_form(ODDS_NAR), {8: (2.0, 1), 4: (3.0, 2)})
        self.assertEqual(K.parse_odds_form(ODDS_BEFORE), {})

    def test_trio_odds(self):
        j = {"status": "middle", "data": {"odds": {"7": {"010203": ["3,602.0", "0", "299"], "020309": ["12.3", "0", "5"],
                                                          "030409": ["45.1", "0", "40"]}}}}
        self.assertEqual(K.parse_trio_odds(j), {"1-2-3": 3602.0, "2-3-9": 12.3, "3-4-9": 45.1})
        self.assertEqual(K.parse_trio_odds(j, (9, 2)), {"2-3-9": 12.3})       # ◎○を含む組だけ
        self.assertEqual(K.parse_trio_odds({"data": {"odds": {}}}), {})

    def test_oshi(self):
        # 後押しの買い目: 後押しが境目以上の馬 x(1番人気でない・単勝20倍以下)と、1番人気の馬連
        odds = {1: 2.0, 2: 5.0, 3: 9.0, 4: 20.0, 5: 20.1, 6: 40.0}
        push = {1: 0.9, 2: 0.3, 3: 0.1, 4: 0.25, 5: 0.9, 6: 0.9}
        pk = L.oshi_picks(push, odds, 0.2171)
        self.assertEqual([(x["u"], x["fav"]) for x in pk], [(2, 1), (4, 1)])          # 1番人気(1)・20倍より上(5・6)・境目より下(3)は入らない
        self.assertEqual(L.oshi_picks({"2": 0.3}, {"1": 2.0, "2": 5.0, "3": 9.0, "4": 20.0, "5": 30.0}, 0.2171)[0]["u"], 2)     # 馬番は文字でもよい
        self.assertEqual(L.oshi_picks({2: 0.2171}, {1: 2.0, 2: 5.0, 3: 9.0, 4: 20.0, 5: 30.0}, 0.2171)[0]["push"], 0.2171)      # ちょうど境目は入る
        self.assertEqual(L.oshi_picks({1: 0.5, 2: 0.5}, {1: 2.0, 2: 2.0, 3: 9.0, 4: 20.0, 5: 30.0}, 0.2)[0]["fav"], 1)          # 同じオッズなら馬番の小さい方が1番人気
        self.assertEqual(L.oshi_picks(push, {1: 2.0, 2: 5.0, 3: 9.0, 4: 20.0}, 0.2171), [])                                      # 5頭いない
        self.assertEqual(L.oshi_picks(push, odds, None), [])                                                                    # 境目が無い(式が無い)
        self.assertEqual(L.oshi_picks({2: None}, odds, 0.2171), [])
        # アプリの数字(index.html の OSHI・PRE)と同じ。消した買い目(割安ワイド・堅い買い目)がアプリに残っていないこと
        html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba", "index.html"), encoding="utf-8").read()
        m = re.search(r'const OSHI = \{q: "([\d.]+)", max_odds: ([\d.]+)\};', html)
        self.assertEqual((m.group(1), float(m.group(2))), (L.OSHI["q"], L.OSHI["max_odds"]))
        m = re.search(r"const PRE = \{pt: ([\d.]+), alpha: ([\d.]+), fresh: ([\d.]+)\};", html)
        self.assertEqual([float(x) for x in m.groups()], [L.PRE[k] for k in ("pt", "alpha", "fresh")])
        self.assertNotIn("割安", html)
        self.assertNotIn("堅い買い目", html)

    def test_oshi_history(self):
        # 後押しの買い目の日ごとの成績: PUSH_FROM より前は push_model.json の days、そこからは予想に入っている push と確定オッズで数える
        import tempfile, keiba_push as PU
        hs = [{"umaban": u} for u in (1, 2, 3, 4, 5, 6)]
        ph = [{"umaban": 1, "push": 0.9, "push2": 0.0}, {"umaban": 2, "push": 0.3, "push2": 0.0}, {"umaban": 3, "push": 0.25, "push2": 0.0},
              {"umaban": 4, "push": -0.1, "push2": 0.0}, {"umaban": 5, "push": 0.5, "push2": 0.0}, {"umaban": 6}]
        res = {"order": [1, 3, 2], "odds": {"1": 1.5, "2": 4.0, "3": 9.0, "4": 15.0, "5": 30.0, "6": 12.0}, "payout": {"馬連": {"1-3": 620}}}
        race = {"horses": hs, "result": res, "pred": {"verdict": "見送り", "horses": ph}}
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "days"))
            json.dump({"date": "20261115", "races": {"202605050811": race, "202605050812": dict(race, result={"order": [1]})}}, open(os.path.join(tmp, "days", "20261115.json"), "w"))
            json.dump({"date": "20261108", "races": {"202605050811": race}}, open(os.path.join(tmp, "days", "20261108.json"), "w"))     # PUSH_FROM より前は数え直さない
            keep = K.OUT
            K.OUT = tmp
            PU._M["m"] = {"cuts": {"win_top": {"0.95": 0.2171}}, "days": [["20261107", 9, 1, 1200], ["20261115", 99, 9, 9999]]}
            try:
                out = K.oshi_history()
            finally:
                K.OUT = keep
                PU._M.clear()
        # 1番人気は1番。2番(0.3)・3番(0.25)が後押しの馬 → 馬連 1-2(外れ)・1-3(620円)。5番は20倍より上、6番は push が無い
        self.assertEqual(out, [["20261107", 9, 1, 1200], ["20261115", 2, 1, 620]])

    def test_push(self):
        # 後押し: 評価値をレースの中で平均0・ばらつき1にそろえて、重みを掛けて足す(keiba_push.py)
        import keiba_push as PU
        z = PU.zs([1.0, 2.0, 3.0, None])
        self.assertEqual(z[3], 0.0)                                          # 無い値は 0(平均)
        self.assertAlmostEqual(z[0], -1.2247, places=3); self.assertAlmostEqual(z[1], 0.0); self.assertAlmostEqual(z[2], 1.2247, places=3)
        self.assertEqual(PU.zs([5, 5, 5]), [0.0, 0.0, 0.0])                  # 差が無ければ全部 0
        self.assertEqual(PU.zs([None, None]), [0.0, 0.0])
        self.assertEqual(max(PU.zs([0] * 30 + [100])), 3.0)                  # ±3 で切る
        fs = [{"a": 1, "style": 0}, {"a": 3, "style": 2}, {"a": None, "style": 2}, {"a": 2, "style": None}]
        self.assertEqual([round(v, 3) for v in PU.column(fs, "a")], [-1.225, 1.225, 0.0, 0.0])
        self.assertEqual([round(v, 3) for v in PU.column(fs, "style=2")], [-1.0, 1.0, 1.0, -1.0])   # 種類は 0/1 にしてからそろえる
        m = {"features": ["a", "style=2"], "win": [0.1, 0.0], "pair": [-0.2, 0.05]}
        sc = PU.score([(3, fs[0]), (5, fs[1]), (7, fs[2]), (9, fs[3])], m)
        self.assertEqual(sc[3], (round(0.1 * -1.2247, 4), round(-0.2 * -1.2247 + 0.05 * -1.0, 4)))
        self.assertEqual(sc[7], (0.0, 0.05))
        self.assertEqual(PU.score([], m), {})
        self.assertEqual([PU.level(v, [-0.1, 0.2]) for v in (-0.3, -0.1, 0.0, 0.2, 0.5, None)], [0, 0, 1, 2, 2, None])
        out = {"horses": [{"umaban": 3}, {"umaban": 5}, {"umaban": 8}]}
        PU._M["m"] = m
        PU.add(out, [(3, fs[0]), (5, fs[1])])
        PU._M.clear()
        self.assertEqual((out["horses"][0]["push"], out["horses"][0]["push2"]), PU.score([(3, fs[0]), (5, fs[1])], m)[3])
        self.assertNotIn("push", out["horses"][2])                            # 材料の無い馬には付けない
        names = PU.candidates([{"a": 1, "a_rank": 1, "venue": 3, "course_waku_idx": 1, "style": 1, "sex": 0}] * 60 + [{"a": 2, "style": 2, "sex": 0}])
        self.assertEqual(names, ["a", "style=1", "sex=0"])                    # そろえ直した値・場・コース傾向は入れない。少ない種類(2%以下)も入れない

    def test_push_history(self):
        # モデルの買い目の、後押しの強さごとの成績: 予想に push2 が入っているレースだけ数える(買い目の馬の push2 の平均を cuts.model で分ける)
        import tempfile, keiba_push as PU
        hs = [{"umaban": u} for u in (1, 2, 3, 4, 5)]
        ph = [{"umaban": 1, "push": 0.3, "push2": 0.2}, {"umaban": 2, "push": -0.5, "push2": -0.1}, {"umaban": 3, "push": 0.0, "push2": -0.4},
              {"umaban": 4, "push": 0.0, "push2": 0.0}, {"umaban": 5, "push": 0.0, "push2": 0.0}]
        res = {"order": [1, 3, 2], "odds": {"1": 1.5, "2": 4.0, "3": 9.0, "4": 15.0, "5": 30.0}, "payout": {"3連複": {"1-2-3": 900}}}
        race = {"horses": hs, "result": res,
                "pred": {"verdict": "B", "horses": ph, "bets": [{"type": "3連複", "combos": [[1, 2, 3]], "unit": 100}]}, "settle": {"stake": 100, "ret": 900, "hit": True}}
        old = dict(race, pred=dict(race["pred"], horses=[{"umaban": h["umaban"]} for h in ph]))       # push の無い予想は数えない
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "days"))
            json.dump({"date": "20261115", "races": {"202605050811": race, "202605050812": old}}, open(os.path.join(tmp, "days", "20261115.json"), "w"))
            json.dump({"date": "20261108", "races": {"202605050811": race}}, open(os.path.join(tmp, "days", "20261108.json"), "w"))     # PUSH_FROM より前
            keep = K.OUT
            K.OUT = tmp
            PU._M["m"] = {"cuts": {"model": [-0.15, 0.0]}}
            try:
                out = K.push_history()
            finally:
                K.OUT = keep
                PU._M.clear()
        self.assertEqual(out, {"from": "20261115", "model": [[0, 0, 0], [1, 1, 900], [0, 0, 0]]})     # 1-2-3 の push2 の平均 -0.1 → ふつう

    def test_course_own(self):
        # 自前のコース傾向: コースごとの 実際の3着内 ÷ 頭数から見た見込み を、芝ダ全体の同じ値で割る(少ないうちは寄せる)
        import keiba_model as M
        def race(v, s, d, order, styles):
            hs = [{"umaban": u, "waku": u, "past": [{"pass": {"逃げ": [1], "先行": [2], "差し": [6], "追込": [10]}[styles.get(u, "先行")], "n": 12, "surface": s}] * 3}
                  for u in range(1, 9)]
            return {"center": "jra", "venue": v, "surface": s, "dist": d, "horses": hs, "result": {"order": order}}
        acc = {}
        for _ in range(40):
            M.course_add(acc, race("新潟", "芝", 1000, [8, 7, 6], {}))          # 外の馬番ばかり来るコース
            M.course_add(acc, race("東京", "芝", 1600, [1, 4, 8], {}))
        M.course_add(acc, dict(race("東京", "芝", 1600, [1, 2, 3], {}), center="nar"))     # 地方は数えない
        M.course_add(acc, race("東京", "障", 3000, [1, 2, 3], {}))                       # 障害は数えない
        jt = {"course": {"by": acc["course"]}}
        r = race("新潟", "芝", 1000, [], {})
        self.assertGreater(M.course_own(jt, r, "gate", "外"), 1.2)
        self.assertLess(M.course_own(jt, r, "gate", "内"), 0.8)
        self.assertGreater(M.course_own(jt, r, "waku", 8), M.course_own(jt, r, "waku", 1))
        self.assertEqual(M.course_own(jt, race("中山", "芝", 2000, [], {}), "gate", "外"), 1.0)   # 数えていないコースは芝全体(= 1)
        self.assertIsNone(M.course_own(jt, r, "gate", None))
        self.assertIsNone(M.course_own({}, r, "gate", "外"))
        self.assertIsNone(M.course_own(jt, race("新潟", "ダ", 1200, [], {}), "gate", "外"))      # ダートはまだ数えていない(30より少ない)
        self.assertEqual(acc["course"]["C|東京芝1600|all|all"], [120, 120.0])                 # 8頭立て: 1頭あたり見込み 3/8
        self.assertNotIn("C|東京障3000|all|all", acc["course"])
        # ふだんの脚質(同じ数なら新しい走り)・馬番の位置
        self.assertEqual(M.usual_style([{"pass": [9], "n": 10}, {"pass": [1], "n": 10}]), "追込")
        self.assertEqual(M.usual_style([{"pass": [1], "n": 10}, {"pass": [9], "n": 10}, {"pass": [9], "n": 10}]), "追込")
        self.assertIsNone(M.usual_style([{"pass": [], "n": 10}]))
        self.assertEqual([M.gate_of(u, 12) for u in (1, 4, 5, 8, 9, 12)], ["内", "内", "中", "中", "外", "外"])
        self.assertIsNone(M.gate_of(None, 12))

    def test_ev_wide(self):
        # 期待値ワイド: 確率の高い順に6組と ◎-○ の中で、期待値(0.75×強さ×(下限+0.35×幅))が 1.0以上の組
        tp = {"1-2": 0.8, "1-3": 0.5, "2-3": 0.45, "1-4": 0.3, "2-4": 0.25, "3-4": 0.2, "1-5": 0.15, "5-6": 0.1}
        wide = {"1-2": [1.2, 1.4], "1-3": [2.6, 3.0], "2-3": [3.0, 3.4], "1-4": [3.8, 4.6], "2-4": [6.0, 7.0], "3-4": [5.0, 6.0],
                "1-5": [9.0, 10.0], "5-6": [20.0, 25.0]}
        pk = L.ev_wide_picks(tp, wide)
        self.assertEqual([x["key"] for x in pk], ["1-3", "2-3", "2-4"])                 # 1-2 は 0.73、1-4 は 0.94、3-4 は 0.80
        self.assertAlmostEqual(pk[0]["ev"], 0.75 * 0.5 * (2.6 + 0.35 * 0.4), places=6)
        self.assertEqual(pk[0]["pair"], [1, 3])
        self.assertEqual([x["key"] for x in L.ev_wide_picks(tp, wide, "5-6")], ["1-3", "2-3", "2-4", "5-6"])   # ◎-○ は7番目でも表に出る
        self.assertNotIn("1-5", [x["key"] for x in L.ev_wide_picks(tp, wide)])            # 7番目(表に出ない)
        self.assertEqual(L.ev_wide_picks({"1-2": 0.5}, {"1-2": [1.0, 1.0]}), [])           # 取消の返還(1.0倍)は数えない
        self.assertEqual(len(L.ev_wide_picks({"1-2": 0.4}, {"1-2": [3.1, 3.4]})), 1)        # 期待値 0.96: 境目(0.95)以上
        self.assertEqual(L.ev_wide_picks({"1-2": 0.4}, {"1-2": [3.0, 3.4]}), [])            # 期待値 0.94: 境目より下
        self.assertEqual(L.ev_wide_picks(None, wide), [])
        # アプリの数字(index.html の EV_WIDE)と同じ
        html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba", "index.html"), encoding="utf-8").read()
        m = re.search(r"const EV_WIDE = \{min_ev: ([\d.]+), top: (\d+), push_ev: ([\d.]+)\};", html)
        self.assertEqual((float(m.group(1)), int(m.group(2)), float(m.group(3))), (L.EV_WIDE["min_ev"], L.EV_WIDE["top"], L.EV_WIDE["push_ev"]))
        # 後押しで絞る(2026-11-27): 後押しの式の期待値(pp × 払戻の見込み)が push_ev 未満の組は外す。pp が無い(後押しの無いレース)なら絞らない
        eff = 2.6 + 0.35 * 0.4
        pp = {"1-3": 1.0 / eff - 0.01, "2-3": 0.5, "2-4": 0.3}
        self.assertEqual([x["key"] for x in L.ev_wide_picks(tp, wide, pp=pp)], ["2-3", "2-4"])          # 1-3 は後押し込み 0.99
        self.assertEqual([x["key"] for x in L.ev_wide_picks(tp, wide, pp={"2-3": 0.5})], ["2-3"])       # pp に無い組は外す
        self.assertEqual([x["key"] for x in L.ev_wide_picks(tp, wide, pp={})], ["1-3", "2-3", "2-4"])    # 後押しが無ければ絞らない
        rows = L.ev_wide_rows(tp, wide, pp=pp)
        self.assertAlmostEqual(next(x for x in rows if x["key"] == "2-3")["pev"], 0.5 * (3.0 + 0.35 * 0.4))
        self.assertIsNone(next(x for x in rows if x["key"] == "1-2")["pev"])

    def test_result_partial(self):
        # 結果が全頭そろっていない(発走 8分後の速報で上位5頭だけ)なら取り直す。そろっていれば取り直さない(2026-11-27)
        hs = [{"umaban": u} for u in range(1, 11)] + [{"umaban": 11, "scratched": True}]
        top5 = {"order": [3, 1, 4, 5, 9], "odds": {str(u): 2.0 + u for u in (3, 1, 4, 5, 9)}, "payout": {"単勝": {"3": 500}}}
        full = {"order": list(range(1, 10)), "odds": {str(u): 2.0 + u for u in range(1, 11)}, "payout": {}}
        self.assertTrue(K.result_partial({"horses": hs, "result": top5}))
        self.assertFalse(K.result_partial({"horses": hs, "result": full}))
        self.assertFalse(K.result_partial({"horses": hs, "result": dict(full, odds={str(u): 3.0 for u in range(1, 10)})}))   # 競走中止などで1頭欠け
        self.assertFalse(K.result_partial({"horses": hs}))                                    # 結果がまだ無い
        rc = {"horses": hs, "result": top5}
        self.assertTrue(K.take_result(rc, full))
        self.assertEqual(rc["result"]["payout"], {"単勝": {"3": 500}})                        # 払戻が無ければ前のを残す
        self.assertFalse(K.take_result(rc, {"order": [1], "odds": {}, "payout": {}}))

    def test_pc_notify_numbers(self):
        # PC の通知(pc/notify.js)の買い目の決め方の数字が keiba_logic と同じ(2026-11-27)
        js = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pc", "notify.js"), encoding="utf-8").read()
        m = re.search(r"const PRE = \{ pt: ([\d.]+), alpha: ([\d.]+) \};", js)
        self.assertEqual((float(m.group(1)), float(m.group(2))), (L.PRE["pt"], L.PRE["alpha"]))
        m = re.search(r"const EV_WIDE = \{ min_ev: ([\d.]+), top: (\d+), push_ev: ([\d.]+) \};", js)
        self.assertEqual((float(m.group(1)), int(m.group(2)), float(m.group(3))), (L.EV_WIDE["min_ev"], L.EV_WIDE["top"], L.EV_WIDE["push_ev"]))
        m = re.search(r"const OSHI = \{ q: \"([\d.]+)\", max_odds: ([\d.]+) \};", js)
        self.assertEqual((m.group(1), float(m.group(2))), (L.OSHI["q"], L.OSHI["max_odds"]))
        m = re.search(r"const OSHIW = \{ q: \"([\d.]+)\", max_odds: ([\d.]+) \};", js)
        self.assertEqual((m.group(1), float(m.group(2))), (L.OSHIW["q"], L.OSHIW["max_odds"]))

    def test_oshiw(self):
        # 後押しワイド(試し): 2頭とも3着内の式の上5%の馬(1番人気でない・20倍以下)× 2番人気(その馬が2番人気なら3番人気)
        od = {1: 2.0, 2: 4.0, 3: 6.0, 4: 12.0, 5: 25.0}
        pk = L.oshiw_picks({1: 0.5, 2: 0.3, 4: 0.2, 5: 0.4}, od, 0.15)
        self.assertEqual([(x["u"], x["mate"]) for x in pk], [(2, 3), (4, 2)])            # 1 は1番人気、5 は20倍超え
        self.assertEqual(L.oshiw_picks({4: 0.2}, {1: 2, 4: 12}, 0.15), [])                 # 5頭いない
        self.assertEqual(L.oshiw_picks({4: 0.2}, od, None), [])
        html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba", "index.html"), encoding="utf-8").read()
        m = re.search(r"const OSHIW = \{q: \"([\d.]+)\", max_odds: (\d+)\};", html)
        self.assertEqual((m.group(1), float(m.group(2))), (L.OSHIW["q"], L.OSHIW["max_odds"]))

    def test_pair_probs(self):
        # 後押しの式の組の確率(keiba_push.pair_probs): 土台(オッズ)+ 2頭の後押し。後押しが大きいほど高い。式が無ければ {}
        import math
        import keiba_push as PU
        hs = [(1, 2.0), (2, 4.0), (3, 6.0), (4, 10.0), (5, 20.0), (6, 40.0)]
        tp = {"1-2": 0.6, "1-3": 0.5, "2-3": 0.3}
        wide = {"1-2": [1.5, 1.8], "1-3": [2.0, 2.5], "2-3": [4.0, 5.0]}
        p2 = {u: 0.0 for u in range(1, 7)}
        pb = [0.1, 1.0, 0.05, -0.5, -0.05, 0.02, 0.15]
        m = {"pair_base": pb}
        out = PU.pair_probs(hs, tp, wide, p2, m)
        self.assertEqual(set(out), {"1-2", "1-3", "2-3"})
        # 手で計算した値と同じ(土台の材料は train_push.build の BP と同じ並び)
        inv = {u: 1 / o for u, o in hs}; tot = sum(inv.values()); lq = {u: math.log(inv[u] / tot) for u in inv}
        mean = sum(lq.values()) / 6; sd = (sum((v - mean) ** 2 for v in lq.values()) / 6) ** 0.5
        zq = {u: (lq[u] - mean) / sd for u in lq}
        lg = lambda p: math.log(p / (1 - p))
        x0 = lg(0.75 * 0.6); xw = lg(min(0.775 / 1.65, 0.98))
        z = sum(w * x for w, x in zip(pb, [1, x0, x0 * x0, xw, xw * xw, zq[1] + zq[2], lq[1] + lq[2]]))
        self.assertAlmostEqual(out["1-2"], 1 / (1 + math.exp(-z)))
        up = PU.pair_probs(hs, tp, wide, {**p2, 1: 0.3}, m)
        self.assertGreater(up["1-2"], out["1-2"])
        self.assertEqual(PU.pair_probs(hs, tp, wide, {1: 0.1}, m), {})                     # 後押しの無い馬の組は出さない
        self.assertEqual(PU.pair_probs(hs, tp, wide, p2, {}), {})                          # 式(pair_base)が無い
        # 本番の式に pair_base がある(アプリの pushPairProbs が使う)
        pm = PU.load()
        if pm:
            self.assertEqual(len(pm.get("pair_base") or []), 7)
        # 発走前のオッズは発走の30分前より後に取ったものだけ
        rc = {"start": "10:05", "trio_pair_at": "2026-11-21T09:50:00+09:00"}
        self.assertTrue(K.pre_fresh("20261121", rc))
        self.assertFalse(K.pre_fresh("20261121", dict(rc, trio_pair_at="2026-11-21T09:30:00+09:00")))
        self.assertFalse(K.pre_fresh("20261121", dict(rc, trio_pair_at="2026-11-21T10:06:00+09:00")))

    def test_apply_pc_odds(self):
        # PC が取った発走前のオッズは、手元の分より新しく、発走より前に取ったものだけ入れる
        day = {"races": {
            "202605010101": {"start": "10:05", "horses": [{"umaban": 1, "odds": 3.0, "ninki": 2}, {"umaban": 2, "odds": 2.0, "ninki": 1}],
                             "trio_pair_at": "2026-11-14T09:40:00+09:00", "trio_pair": {"1-2": 0.1}, "wide_odds": {"1-2": [9, 9]}},
            "202605010102": {"start": "10:35", "horses": [{"umaban": 1}], "trio_pair_at": "2026-11-14T10:20:00+09:00", "trio_pair": {"1-2": 0.5}},
            "202605010103": {"start": "11:05", "horses": [{"umaban": 1}]}}}
        s = lambda at: {"at": at, "odds": {"1": [2.5, 1], "2": [4.1, 2]}, "wide": {"1-2": [2.0, 2.4]}, "tp": {"1-2": 0.3}, "trio": {"1-2-3": 5.0}}
        snap = {"date": "20261114", "races": {"202605010101": s("2026-11-14T10:03:00+09:00"),      # 新しい → 入れる
                                              "202605010102": s("2026-11-14T10:10:00+09:00"),      # 手元の方が新しい → そのまま
                                              "202605010103": s("2026-11-14T11:06:00+09:00"),      # 発走のあと → 入れない
                                              "209999999999": s("2026-11-14T10:00:00+09:00")}}     # その日に無いレース
        self.assertEqual(K.apply_pc_odds("20261114", day, snap), 1)
        r = day["races"]["202605010101"]
        self.assertEqual((r["trio_pair"], r["wide_odds"], r["trio_pair_at"], r["odds_src"], r["trio_odds"]),
                         ({"1-2": 0.3}, {"1-2": [2.0, 2.4]}, "2026-11-14T10:03:00+09:00", "PC", {"1-2-3": 5.0}))
        self.assertEqual([(h["odds"], h["ninki"]) for h in r["horses"]], [(2.5, 1), (4.1, 2)])
        self.assertEqual(day["races"]["202605010102"]["trio_pair"], {"1-2": 0.5})
        self.assertNotIn("trio_pair", day["races"]["202605010103"])
        self.assertEqual(K.apply_pc_odds("20261114", day, None), 0)

    def test_trio_pair_sums(self):
        # 2頭を含む3連複の組の 1/オッズ の合計。小さい組は保存しない
        to = {"1-2-3": 5.0, "1-2-4": 10.0, "1-3-4": 20.0, "2-3-4": 40.0, "1-2-10": 1000.0}
        self.assertEqual(K.trio_pair_sums(to), {"1-2": 0.301, "1-3": 0.25, "2-3": 0.225, "1-4": 0.15, "2-4": 0.125, "3-4": 0.075})
        self.assertEqual(K.trio_pair_sums({}), {})

    def test_wide_odds(self):
        # API(type=5)の形(取得の診断で見たもの)。キーは馬番2桁ずつ、値は [下限, 上限, 人気]
        j = {"status": "middle", "data": {"odds": {"5": {"0102": ["133.5", "134.8", "67"], "0310": ["2.1", "2.4", "1"],
                                                         "0405": ["---.-", "---.-", "**"]}}}}
        self.assertEqual(K.parse_wide_odds(j), {"1-2": [133.5, 134.8], "3-10": [2.1, 2.4]})
        self.assertEqual(K.parse_wide_odds(None), {})
        self.assertEqual(K.parse_wide_odds({"data": {"odds": []}}), {})

    def test_past_skips_training_rows(self):
        oikiri = PAST_HTML.replace("</table>", """<tr class="OikiriDataHead3 HorseList">
<td class="Waku2">2</td><td class="Umaban">3</td><td class="Horse_Info">前走</td></tr></table>""")
        self.assertEqual(len(K.parse_past(oikiri)[3]), 1)

    def test_past(self):
        p = K.parse_past(PAST_HTML)
        self.assertIn(3, p)
        r = p[3][0]
        self.assertEqual((r["date"], r["venue"], r["rank"]), ("2026-09-20", "中山", 1))
        self.assertEqual((r["surface"], r["dist"], r["baba"], r["time"]), ("芝", 2200, "良", 131.8))
        r2 = K.parse_past(PAST_HTML.replace("芝2200 2:11.8 良", "芝1000 57.9 稍"))[3][0]
        self.assertEqual((r2["dist"], r2["time"], r2["baba"]), (1000, 57.9, "稍"))
        self.assertEqual((r["n"], r["umaban"], r["ninki"], r["kinryo"]), (15, 7, 2, 57.0))
        self.assertEqual(r["pass"], [3, 3, 2, 2])
        self.assertEqual(r["margin"], -0.2)
        self.assertEqual(r["cls"], 7)

    def test_result(self):
        r = K.parse_result(RESULT_HTML)
        self.assertEqual(r["order"], [3, 1, 9])
        self.assertEqual(r["payout"]["馬連"], {"1-3": 1230})
        self.assertEqual(r["payout"]["ワイド"], {"1-3": 400, "3-9": 900, "1-9": 700})
        self.assertEqual(r["payout"]["3連単"], {"3-1-9": 12340})
        self.assertEqual(r["payout"]["馬単"], {"3-1": 2560})                 # 馬単は着順どおり
        self.assertEqual(r["payout"]["複勝"], {"3": 170, "1": 120, "9": 310})   # 自分で買った券種の収支に使う
        self.assertEqual(r["payout"]["枠連"], {"1-2": 560})
        # ほかの買い方(参考): ◎3 ○1 なら馬連・馬単は的中、単勝は払戻表に無いので None
        self.assertEqual(K.alt_bets({"honmei": 3, "taikou": 1}, r), {"ワイド": 400, "馬連": 1230, "馬単": 2560, "単勝": None})
        self.assertEqual(K.alt_bets({"honmei": 1, "taikou": 3}, r)["馬単"], 0)  # 逆の順は外れ
        # 3連複◎○▲(▲=相手の1頭目)。払戻表に3連複が無ければ None
        self.assertIsNone(K.alt_bets({"honmei": 3, "taikou": 1, "aite": [9, 2]}, r)["3連複"])
        pay = {"payout": {"3連複": {"1-3-9": 2000}}}
        self.assertEqual(K.alt_bets({"honmei": 3, "taikou": 1, "aite": [9, 2]}, pay)["3連複"], 2000)
        self.assertEqual(K.alt_bets({"honmei": 3, "taikou": 1, "aite": [2, 9]}, pay)["3連複"], 0)
        self.assertEqual((r["ninki"]["3"], r["odds"]["3"]), (8, 15.8))

    def test_center(self):
        self.assertEqual(K.center_of("202605040811"), "jra")
        self.assertEqual(K.venue_of("202644101801"), "大井")


class Centers(unittest.TestCase):
    """2026-10-21 から中央だけ。一覧が取れない(メンテナンス)と開催が無い日を分ける"""

    def ids(self, pages):
        get, K.get = K.get, lambda url, params=None, as_json=False: pages.get(url.split("/")[2])
        try:
            return K.race_ids("20261025")
        finally:
            K.get = get

    def test_target_venues(self):
        # 高知のアプリ: 地方のうち KEIBA_NAR_VENUES の場だけ(高知 54 / 佐賀 55)
        old = (K.CENTERS, K.NAR_VENUES)
        try:
            K.CENTERS, K.NAR_VENUES = ("nar",), ("高知",)
            self.assertEqual([K.target(x) for x in ("202654110812", "202655110812", "202605040101")], [True, False, False])
            K.NAR_VENUES = ()
            self.assertTrue(K.target("202655110812"))
        finally:
            K.CENTERS, K.NAR_VENUES = old

    def test_jra_only(self):
        self.assertEqual(K.CENTERS, ("jra",))
        page = 'race_id=202605040811 race_id=202644102501 kaisai_id=2026441025'
        self.assertEqual(self.ids({"race.netkeiba.com": page, "nar.netkeiba.com": page}), ["202605040811"])

    def test_list_failure_is_not_empty_day(self):
        self.assertIsNone(self.ids({}))                                  # 503 などで取れない
        self.assertEqual(self.ids({"race.netkeiba.com": "<html></html>"}), [])   # 開催が無い日


class Weather(unittest.TestCase):
    """気象の材料(既定では使わないが、作り方は確かめておく)"""

    def test_feats(self):
        import datetime
        import keiba_weather as KW
        row = [[10 + i, 60, 1.0 if i < 2 else 0.0, 61 if i < 2 else 0] for i in range(10)]   # 8〜17時
        wx = {"hours": list(range(8, 18)), "data": {"東京": {"20261018": row, "20260801": [[30, 70, 0, 0]] * 10,
                                                          "20260105": [[5, 40, 0, 0]] * 10}}}
        self.assertEqual(KW.at(wx, "東京", "2026-10-18", "15:40"), (18, 60, 0, 2.0))   # 16時の値、朝からの雨 2mm
        self.assertEqual(KW.code_kind(61), 2)
        past = [{"venue": "東京", "date": "2026-08-01", "rank": 1, "n": 10},
                {"venue": "東京", "date": "2026-01-05", "rank": 10, "n": 10},
                {"venue": "大井", "date": "2026-01-06", "rank": 1, "n": 10}]       # 中央の外は数えない
        f = {}
        KW.feats(f, past, {"venue": "東京", "start": "15:40"}, datetime.date(2026, 10, 18), wx)
        self.assertEqual((f["wx_temp"], f["wx_n"]), (18, 2))
        self.assertAlmostEqual(f["wx_hot_rr"], 0.45)       # 暑い日は着順の比が平均より 0.45 良い
        self.assertAlmostEqual(f["wx_cold_rr"], -0.45)


class Speed(unittest.TestCase):
    def test_track_variant(self):
        """同じコースで、速い日に走った馬は馬場差の分だけ割り引かれる"""
        import keiba_speed as SP
        recs = []
        for day, shift in (("2026-05-01", 0.0), ("2026-05-02", -1.0)):     # 2日目は1kmあたり1秒速い馬場
            for i in range(30):
                recs.append({"date": day, "venue": "東京", "surface": "芝", "dist": 1600, "cls": 3,
                             "center": "jra", "margin": 0.5, "time": round(1.6 * (60 + shift) + (i % 5) * 0.1, 1),
                             "last3f": 34.0})
        tb = SP.fit(recs)
        slow = SP.figure({"date": "2026-05-01", "venue": "東京", "surface": "芝", "dist": 1600, "time": 96.0}, tb)
        fast = SP.figure({"date": "2026-05-02", "venue": "東京", "surface": "芝", "dist": 1600, "time": 94.4}, tb)
        self.assertAlmostEqual(slow, fast, delta=1.5)    # 1.6秒速いが、馬場差を引けばほぼ同じ指数
        # 表に無い日は、集めた走りから馬場差を足せる
        new = [dict(r, date="2026-05-03") for r in recs[30:]]
        SP.extend(tb, new)
        self.assertLess(tb["t"]["V"]["20260503|東京|芝"], -0.3)   # 基準は2日の真ん中なので約 -0.5


class Concurrency(unittest.TestCase):
    def test_model_load_in_threads(self):
        """並列の予想で、モデルを読み終わる前の呼び出しが「モデルなし」にならない"""
        import keiba_model as KM
        if not os.path.exists(KM.MODEL_FILE):
            self.skipTest("モデルが無い")
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm が無い")
        saved = dict(KM._MODEL)
        KM._MODEL.update({"booster": None, "meta": None, "tried": False})
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as ex:
                got = list(ex.map(lambda _: KM.load()[0] is not None, range(16)))
            self.assertTrue(all(got))
        finally:
            KM._MODEL.update(saved)


class ModelFiles(unittest.TestCase):
    def test_model_files(self):
        """種を変えた何本かのモデル(meta["models"])を全部読む。古い形は keiba_model.txt の1本"""
        import keiba_model as KM
        self.assertEqual(KM.model_files({}), [KM.MODEL_FILE])
        fs = KM.model_files({"models": ["keiba_model.txt", "keiba_model_s2.txt"]})
        self.assertEqual([os.path.basename(f) for f in fs], ["keiba_model.txt", "keiba_model_s2.txt"])
        self.assertEqual(os.path.dirname(fs[1]), KM.HERE)


class CourseNar(unittest.TestCase):
    def test_nar_course(self):
        """地方のコース傾向(うまめし.com の記事から)を読める。距離が無ければ場全体"""
        c = L.course_of({"center": "nar", "venue": "園田", "surface": "ダ", "dist": 1400})
        if not c:
            self.skipTest("keiba_course_nar.json が無い")
        self.assertTrue(c["exact"])
        self.assertGreater(c["waku_idx"][8], c["waku_idx"][1])          # 園田1400は外枠有利
        v = L.course_of({"center": "nar", "venue": "園田", "surface": "ダ", "dist": 9999})
        self.assertFalse(v["exact"])
        self.assertIn("場全体", v["name"])
        self.assertTrue(v["waku_idx"])                                   # 場全体の枠も使う


class ArmLive(unittest.TestCase):
    """開催日の更新を、次にいつ起動するか(tools/arm_live.py)"""

    def setUp(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
        import arm_live as A
        self.A = A

    def at(self, s):
        from datetime import datetime
        return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=self.A.JST)

    def test_decide(self):
        d = self.A.decide
        days = [("20261121", "16:25"), ("20261122", "16:20"), ("20261123", "15:30"), ("20261128", "16:25")]
        self.assertEqual(d(self.at("202611220800"), days), "wait 202611220831")     # 朝: 8:31 まで待つ
        self.assertEqual(d(self.at("202611220816"), days), "now")                   # 15分前から すぐ
        self.assertEqual(d(self.at("202611221300"), days), "now")                   # 開催中
        self.assertEqual(d(self.at("202611221700"), days), "now")                   # 最後の発走 +40分まで
        self.assertEqual(d(self.at("202611221701"), days), "wait 202611230831")     # 終わったら 明日の 8:31
        self.assertEqual(d(self.at("202611231615"), days), "none")                  # 次の開催(土曜)は20時間より先
        self.assertEqual(d(self.at("202611272040"), days), "wait 202611280831")     # 金曜の夜に 土曜の分を予約
        self.assertEqual(d(self.at("202611290900"), days), "none")                  # 開催が無い

    def test_guard(self):
        g, now = self.A.guard, self.at("202611221000")
        t = now.timestamp()
        self.assertEqual(g("now", now, t - 5 * 60), "wait 202611221030")           # すぐ終わった回は 30分後に
        self.assertEqual(g("now", now, t - 3 * 3600), "now")
        self.assertEqual(g("now", now, 0), "now")                                   # 「競馬の予想」からは そのまま
        self.assertEqual(g("wait 202611230831", now, t - 60), "wait 202611230831")

    def test_race_days(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "days"))
            races = {"a": {"center": "jra", "venue": "東京", "start": "15:30"}, "b": {"center": "jra", "venue": "東京", "start": "16:10"},
                     "c": {"center": "nar", "venue": "高知", "start": "20:30"}}
            for name, rs in (("20261121", races), ("20261122", races), ("20261124", {"c": races["c"]}), ("20261125", {})):
                json.dump({"races": rs}, open(os.path.join(td, "days", name + ".json"), "w"))
            self.assertEqual(self.A.race_days(td, "20261122"), [("20261122", "16:10")])     # 中央だけ・今日から


if __name__ == "__main__":
    unittest.main()

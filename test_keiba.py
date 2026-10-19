#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""keiba_logic / keiba のパースの確認。  python -m unittest test_keiba"""

import os
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
        # 確率を外から渡す: ◎○は軸にできる馬の上位2頭、買うのは○が基準以上のとき
        hs = [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in range(3, 13)]
        r = race(hs)
        probs = {u: 0.1 for u in range(1, 13)}
        sel = L.model_select()            # 基準は学び直しで変わるので、今の基準で A になる確率にする
        probs.update({1: 0.95, 2: min(0.94, sel["a"] + 0.02), 3: 0.40, 4: 0.30})
        p = L.predict(r, TODAY, probs=probs)
        self.assertTrue(p["model"])
        self.assertEqual((p["honmei"], p["taikou"], p["aite"]), (1, 2, [3, 4]))
        self.assertEqual(p["verdict"], "A")
        self.assertEqual({b["type"] for b in p["bets"]}, {"ワイド"})
        # ○が基準に届かない → 見送り(1強)
        probs[2] = 0.45
        p = L.predict(r, TODAY, probs=probs)
        self.assertEqual(p["verdict"], "見送り")
        self.assertIn("1強", p["reason"])
        # 延長の馬は確率が高くても軸にしない
        ext = horse(5, [run(1, margin=-0.6, dist=1200)] * 3)
        r2 = race([ext] + [strong(1, first=2), strong(2, first=9)] + [middle(u) for u in (3, 4, 6, 7, 8, 9, 10)])
        probs = {u: 0.1 for u in range(1, 11)}
        probs.update({5: 0.9, 1: 0.8, 2: 0.7})
        p = L.predict(r2, TODAY, probs=probs)
        self.assertEqual((p["honmei"], p["taikou"]), (1, 2))
        self.assertIn(5, p["aite"])

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
<table class="Payout_Detail_Table"><tr class="Umaren"><th>馬連</th>
<td class="Result"><ul><li><span>1</span></li><li><span>3</span></li></ul></td>
<td class="Payout"><span>1,230円</span></td></tr>
<tr class="Wide"><th>ワイド</th><td class="Result"><ul><li><span>1</span></li><li><span>3</span></li></ul>
<ul><li><span>3</span></li><li><span>9</span></li></ul><ul><li><span>1</span></li><li><span>9</span></li></ul></td>
<td class="Payout"><span>400円<br>900円<br>700円</span></td></tr>
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
        self.assertEqual((r["ninki"]["3"], r["odds"]["3"]), (8, 15.8))

    def test_center(self):
        self.assertEqual(K.center_of("202605040811"), "jra")
        self.assertEqual(K.venue_of("202644101801"), "大井")


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


if __name__ == "__main__":
    unittest.main()

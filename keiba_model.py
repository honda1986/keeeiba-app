#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keiba_model.py -- 各馬の「3着以内に来る確率」を出すモデル(LightGBM)

■ なぜ
  2026年1〜10月の再現(2,700レース)で、手作りの能力評価で選んだ ◎-○ のワイドは回収率69%、
  同じレースの1・2番人気のワイドは82%だった。手作りの評価は市場より弱い。
  そこで、過去走・出馬表・コース傾向・30か条の加減点を材料に、3着以内の確率を学習させる。

■ 使うもの(すべてレース前にわかるもの)
  過去5走(着差・着順・クラス・通過順・上がり・その時の人気)、出馬表(枠・斤量・馬体重・性齢)、
  コース傾向(中央)、30か条の評価(能力・加減点・軸不可・消し)。同じレースの他馬との比較も入れる。
■ 使わないもの
  そのレースのオッズ・人気(規則 20。人気は最後に照らすだけ)

■ ファイル
  keiba_model.txt       … 学習したモデル(tools/train_model.py が作る)
  keiba_model_meta.json … 特徴量の並び・上がり3Fの基準・学習の成績
"""

import json
import math
import os
import re
import threading

import keiba_logic as L
import keiba_speed as SP
import keiba_weather as KW

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "keiba_model.txt")
META_FILE = os.path.join(HERE, "keiba_model_meta.json")

BABA = {"良": 0, "稍": 1, "重": 2, "不": 3}
# 気象の材料(気温・湿度・雨量・天気)。2026-10-22 に試したが良くならなかったので、既定では使わない(1 で使う)
USE_WX = os.environ.get("KEIBA_WX", "0") == "1"
VENUE_ID = {v: i for i, v in enumerate(sorted(L.VENUES))}
STYLE_ID = {"逃げ": 0, "先行": 1, "差し": 2, "追込": 3}
# 同じレースの中で順位・最大との差を取る特徴量
REL = ["sc_mean", "sc_best2", "sc_last", "margin_min", "margin_mean", "rank_ratio_mean",
       "pop_mean", "pop_min", "rule_score", "rule_ability", "l3f_dev_best", "top3_rate",
       "jockey_top3", "trainer_top3", "spd_best", "spd_best_surf", "spd_mean3", "spd_last", "l3fig_best",
       "spd_adj_best", "spd_adj_mean3"]
JT_K = 30          # 騎手・厩舎の3着内率は、乗った数が少ないうちは全体の平均へ寄せる


def dist_band(d):
    return 0 if (d or 0) <= 1300 else 1 if d <= 1600 else 2 if d <= 2000 else 3


def l3f_key(surface, dist):
    return "{}{}".format(surface or "?", dist_band(dist))


def _num(x):
    return float(x) if x is not None else float("nan")


def jt_name(h, key):
    """騎手・厩舎の名前(印や記号を落とす)"""
    return re.sub(r"[▲△☆◇★\s]", "", h.get(key) or "") or None


def jt_rate(stats, name):
    """[乗った数, 3着内] -> 平均に寄せた3着内率"""
    if not stats or not name:
        return None, 0
    n, t = stats["by"].get(name, (0, 0))
    base = stats["base"]
    return (t + JT_K * base) / (n + JT_K), n


def center_of(venue):
    vi = L.venue_info(venue)[1]
    return vi[0] if vi else None


def speed_feats(f, past, race, sp):
    """馬場差で補正した走破タイム・上がり3Fの指数(keiba_speed)"""
    fig = [SP.figure(r, sp, "t") for r in past]
    lfg = [SP.figure(r, sp, "l") for r in past]
    v = [x for x in fig if x is not None]
    f["spd_best"] = max(v) if v else None
    f["spd_last"] = fig[0] if fig else None
    f["spd_mean3"] = sum(v[:3]) / len(v[:3]) if v else None
    surf = [x for x, r in zip(fig, past) if x is not None and r.get("surface") == race.get("surface")]
    f["spd_best_surf"] = max(surf) if surf else None
    dist = race.get("dist") or 0
    near = [x for x, r in zip(fig, past) if x is not None and r.get("surface") == race.get("surface")
            and abs((r.get("dist") or 0) - dist) <= 200]
    f["spd_best_dist"] = max(near) if near else None
    lv = [x for x in lfg if x is not None]
    f["l3fig_best"] = max(lv) if lv else None
    f["l3fig_mean"] = sum(lv[:3]) / len(lv[:3]) if lv else None
    # 競馬場ごとの目盛りのずれを足した指数(別の場から来た馬とも比べられる。地方競馬レベル順と力関係の考え方)
    if sp.get("lvl"):
        adj = [x + SP.level(r, sp) for x, r in zip(fig, past) if x is not None]
        f["spd_adj_best"] = max(adj) if adj else None
        f["spd_adj_mean3"] = sum(adj[:3]) / len(adj[:3]) if adj else None
        f["spd_adj_last"] = adj[0] if adj and fig and fig[0] is not None else None


def horse_feats(h, race, today, cs, n, l3f_norm, jt=None, sp=None):
    """1頭分(他馬との比較は race_rows で足す)。jt は騎手・厩舎の成績(そのレースより前の日まで)、
    sp は走破タイムの指数の表(keiba_speed)"""
    center = race["center"]
    past = [dict(r) for r in (h.get("past") or []) if r.get("surface") != "障"][:5]
    for r in past:
        r["now_cls"] = race.get("cls")
        vi = L.venue_info(r.get("venue"))[1]
        r["center"] = vi[0] if vi else center
        r["score"] = L.run_score(r, center)
    sc = [r["score"] for r in past if r["score"] is not None]
    rk = [r for r in past if r.get("rank") and (r.get("n") or 0) > 1]
    mg = [max(-1.0, min(5.0, r["margin"])) for r in past if r.get("margin") is not None]
    pops = [r["ninki"] for r in past if r.get("ninki")]
    f = {}
    f["n_runs"] = len(past)
    f["sc_mean"] = sum(sc) / len(sc) if sc else None
    f["sc_max"] = max(sc) if sc else None
    f["sc_last"] = sc[0] if sc else None
    f["sc_best2"] = sum(sorted(sc, reverse=True)[:2]) / min(2, len(sc)) if sc else None
    f["margin_last"] = mg[0] if mg else None
    f["margin_min"] = min(mg) if mg else None
    f["margin_mean"] = sum(mg[:3]) / len(mg[:3]) if mg else None
    rr = [(r["rank"] - 1) / (r["n"] - 1) for r in rk]
    f["rank_ratio_last"] = rr[0] if rr else None
    f["rank_ratio_mean"] = sum(rr) / len(rr) if rr else None
    f["win_rate"] = sum(1 for r in rk if r["rank"] == 1) / len(rk) if rk else None
    f["top3_rate"] = sum(1 for r in rk if r["rank"] <= 3) / len(rk) if rk else None
    # 前走までの人気(その時の市場の評価。今回のオッズではない)
    f["pop_last"] = pops[0] if pops else None
    f["pop_mean"] = sum(pops) / len(pops) if pops else None
    f["pop_min"] = min(pops) if pops else None
    beat = [r["ninki"] - r["rank"] for r in rk if r.get("ninki")]
    f["beat_pop"] = sum(beat) / len(beat) if beat else None
    # 上がり3F(芝ダ・距離帯の平均との差。小さいほど速い)
    dev = [r["last3f"] - l3f_norm[l3f_key(r.get("surface"), r.get("dist"))]
           for r in past if r.get("last3f") and l3f_key(r.get("surface"), r.get("dist")) in l3f_norm]
    f["l3f_dev_best"] = min(dev) if dev else None
    f["l3f_dev_mean"] = sum(dev) / len(dev) if dev else None
    # 位置取り
    pos = [(r["pass"][0] - 1) / max((r.get("n") or 2) - 1, 1) for r in past if r.get("pass")]
    f["pos_first"] = sum(pos) / len(pos) if pos else None
    f["nige_rate"] = sum(1 for r in past if r.get("pass") and r["pass"][0] == 1) / len(past) if past else None
    # 条件の変化
    last = past[0] if past else {}
    dist = race.get("dist") or 0
    f["dist"] = dist
    f["dist_change"] = dist - last["dist"] if last.get("dist") and dist else None
    best = max((r for r in past if r["score"] is not None), key=lambda r: r["score"], default=None)
    f["dist_from_best"] = dist - best["dist"] if best and best.get("dist") and dist else None
    f["same_surface"] = sum(1 for r in past if r.get("surface") == race.get("surface")) / len(past) if past else None
    f["same_venue_runs"] = sum(1 for r in past if r.get("venue") == race.get("venue"))
    # 移籍・場の変化(地方の記事: 中央からの移籍馬、他場から来た馬)
    f["same_venue_last"] = (1 if past[0].get("venue") == race.get("venue") else 0) if past else None
    f["jra_share"] = sum(1 for r in past if r.get("center") == "jra") / len(past) if past else None
    f["from_jra"] = (1 if center == "nar" and past[0].get("center") == "jra" else 0) if past else None
    f["n_venues"] = len({r.get("venue") for r in past}) if past else None
    d0 = L.parse_date(last.get("date")) if last.get("date") else None
    f["rest_days"] = (today - d0).days if d0 and today else None
    f["kinryo"] = h.get("kinryo")
    f["kinryo_change"] = h["kinryo"] - last["kinryo"] if h.get("kinryo") and last.get("kinryo") else None
    f["weight"] = h.get("weight")
    f["wdiff"] = h.get("wdiff")
    m = re.match(r"(牡|牝|セ|騸)\D*(\d+)", h.get("sex_age") or "")
    f["sex"] = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}[m.group(1)] if m else None
    f["age"] = int(m.group(2)) if m else None
    lv = [r.get("cls") for r in past if r.get("cls") is not None]
    f["cls_now"] = race.get("cls")
    f["cls_diff_last"] = race["cls"] - last["cls"] if race.get("cls") is not None and last.get("cls") is not None else None
    f["cls_diff_max"] = race["cls"] - max(lv) if race.get("cls") is not None and lv else None
    # レースの条件
    f["n"] = n
    f["waku"] = h.get("waku")
    f["umaban_ratio"] = (h.get("umaban") or 0) / max(n, 1)
    f["center"] = 0 if center == "jra" else 1
    f["surface"] = 0 if race.get("surface") == "芝" else 1
    f["baba"] = BABA.get((race.get("baba") or "良")[:1], 0)
    f["venue"] = VENUE_ID.get(race.get("venue"))
    f["final_race"] = 1 if "ファイナル" in (race.get("name") or "") else 0     # 高知ファイナルなど(荒れやすい)
    # コース傾向(中央)
    st = None
    if past:
        styles = [L.run_style(r) for r in past]
        styles = [s for s in styles if s]
        st = max(set(styles), key=styles.count) if styles else None
    f["style"] = STYLE_ID.get(st)
    key = L.STYLE_MAP.get(st)
    f["course_style_idx"] = cs["style_idx"].get(key) if cs and key else None
    f["course_waku_idx"] = cs["waku_idx"].get(h.get("waku")) if cs and cs.get("waku_idx") else None
    f["course_kata"] = cs["kata"] if cs and cs.get("exact") else None
    # 騎手・厩舎(そのレースより前の日までの3着内率)
    jt = jt or {}
    f["jockey_top3"], f["jockey_n"] = jt_rate(jt.get("jockey"), jt_name(h, "jockey"))
    f["trainer_top3"], f["trainer_n"] = jt_rate(jt.get("trainer"), jt_name(h, "trainer"))
    if sp:
        speed_feats(f, past, race, sp)
    if USE_WX:
        KW.feats(f, past, race, today)          # 気温・湿度・雨量・天気(keiba_weather.json)
    return f


def race_rows(race, today, pred, l3f_norm, jt=None, sp=None):
    """1レース分の特徴量(馬番の順)。pred は keiba_logic.predict の結果(30か条の評価)"""
    horses = [h for h in race["horses"] if not h.get("scratched")]
    n = len(horses)
    cs = L.course_of(race)
    pe = {x["umaban"]: x for x in (pred or {}).get("horses", [])}
    rows = []
    for h in horses:
        f = horse_feats(h, race, today, cs, n, l3f_norm, jt, sp)
        e = pe.get(h["umaban"], {})
        f["rule_score"] = e.get("score")
        f["rule_ability"] = e.get("ability")
        f["rule_minus"] = sum(m["p"] for m in e.get("minus", [])) if e else None
        f["rule_plus"] = sum(m["p"] for m in e.get("plus", [])) if e else None
        f["rule_axis_ng"] = 1 if e.get("axis_ng") else 0
        f["rule_cut"] = 1 if e.get("cut") else 0
        rows.append((h["umaban"], f))
    # 同じレースの他馬との比較(大きいほど良い/小さいほど良いを揃えずに、順位と最大・最小との差を両方入れる)
    for k in REL:
        vals = [f[k] for _, f in rows if f.get(k) is not None]
        if not vals:
            continue
        hi, lo = max(vals), min(vals)
        mean = sum(vals) / len(vals)
        srt = sorted(vals, reverse=True)
        for _, f in rows:
            v = f.get(k)
            f[k + "_dhi"] = v - hi if v is not None else None
            f[k + "_dlo"] = v - lo if v is not None else None
            f[k + "_dmean"] = v - mean if v is not None else None
            f[k + "_rank"] = srt.index(v) + 1 if v is not None else None
    return rows


# ---------------------------------------------------------------- 予測(アプリで使う)
_MODEL = {"booster": None, "meta": None, "tried": False}
_LOCK = threading.Lock()


def load():
    # 予想は並列に呼ばれる。読み終わる前に「モデルなし」と答えると手作りの評価に落ちるので、読み終わるまで待たせる
    with _LOCK:
        if _MODEL["tried"]:
            return _MODEL["booster"], _MODEL["meta"]
        _MODEL["tried"] = True
        try:
            import lightgbm as lgb
            _MODEL["booster"] = lgb.Booster(model_file=MODEL_FILE)
            _MODEL["meta"] = json.load(open(META_FILE, encoding="utf-8"))
        except Exception as e:          # モデルが無い/lightgbm が無いときは手作りの評価だけで動く
            print("モデルを読めません", type(e).__name__, e)
    return _MODEL["booster"], _MODEL["meta"]


def probs(race, today, pred):
    """馬番 -> 3着以内の確率(モデルが無ければ None)"""
    bst, meta = load()
    if not bst:
        return None
    sp = SP.load() if any(k.startswith(("spd_", "l3fig_")) for k in meta["features"]) else None
    rows = race_rows(race, today, pred, meta["l3f_norm"], meta.get("jt"), sp)
    if not rows:
        return {}
    X = [[_num(f.get(k)) for k in meta["features"]] for _, f in rows]
    with _LOCK:                         # 1つのモデルを並列に使うので、予測は1つずつ
        p = bst.predict(X)
    # 1レースの3着以内は3頭なので、合計が3になるように揃える(頭数が少ないときは頭数まで)
    tot = sum(p)
    want = min(3, len(p))
    return {u: min(0.99, float(v) * want / tot) if tot else float(v) for (u, _), v in zip(rows, p)}

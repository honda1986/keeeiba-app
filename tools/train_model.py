#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3着以内の確率モデルを学習し、買い方まで含めて成績を確かめる。
   python tools/train_model.py [--save]
期間で分ける(後ろの期間は学習に使わない):
  学習 … 〜6月 / 調整 … 7〜8月(早期終了と、買うレースの基準を決める)/ 試験 … 9月〜(最後に1回だけ見る)
比べるもの(買い方はアプリと同じ。選ぶときはワイド◎-○の1点。月ごとの確率の確かめは、中央は3連複◎-○-▲の1点):
  手作り … 今の30か条の評価で選んだ◎○(A/B)
  モデル … 確率の上位2頭を◎○。買うレースは確率で選ぶ
  市場   … 同じレースで1・2番人気(確定オッズ。参考。実際にはレース前に分からない)
--save で全期間で学習し直して keiba_model.txt / keiba_model_meta.json を保存する。"""
import argparse
import glob
import json
import math
import os
import pickle
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keiba as K            # noqa: E402
import keiba_logic as L      # noqa: E402
import keiba_model as M      # noqa: E402
import keiba_speed as SP     # noqa: E402

VALID_FROM, TEST_FROM = "20260701", "20260901"
WF_FROM = "202401"            # 月ごとの確率を作る最初の月(2022・2023年は学ぶだけ)
# 月ごとの確率で成績を確かめる期間(名前, 最初の日, 最後の日の次)。中央の買う基準は2024・2025年で選んだ(2026-10-25)
CHECK_PERIODS = (("2024年(基準を選んだ期間)", "20240101", "20250101"),
                 ("2025年(基準を選んだ期間)", "20250101", "20260101"),
                 ("2026年1〜8月(基準を選ぶのに使っていない)", "20260101", "20260901"),
                 ("2026年9月〜(基準を選ぶのに使っていない)", "20260901", "99999999"))
CACHE = os.environ.get("KEIBA_CACHE", "/tmp/keiba_dataset.pkl")


def load_races():
    out = []
    for f in sorted(glob.glob(K.OUT + "/days/*.json")):
        d = os.path.basename(f)[:8]
        day = K.load_day(d)
        for rid, rc in sorted(day["races"].items()):
            res = rc.get("result") or {}
            if (rc.get("horses") and rc.get("past_at") and len(res.get("order", [])) >= 3
                    and rc.get("surface") in ("芝", "ダ") and K.venue_of(rid) != "帯広"
                    and K.center_of(rid) in K.CENTERS):           # 2026-10-21 から中央だけ
                out.append((d, rc))
    return out


def l3f_norms(races):
    acc = {}
    for d, rc in races:
        if d >= VALID_FROM:
            continue
        for h in rc["horses"]:
            for r in h.get("past") or []:
                if r.get("last3f") and r.get("surface") in ("芝", "ダ"):
                    a = acc.setdefault(M.l3f_key(r["surface"], r.get("dist")), [0.0, 0])
                    a[0] += r["last3f"]
                    a[1] += 1
    return {k: round(s / n, 3) for k, (s, n) in acc.items() if n >= 50}


def jt_snapshot(acc):
    """騎手・厩舎の成績の、その時点での写し"""
    out = {}
    for k, by in acc.items():
        n = sum(v[0] for v in by.values())
        t = sum(v[1] for v in by.values())
        out[k] = {"base": t / n if n else 0.25, "by": {name: (v[0], v[1]) for name, v in by.items()}}
    return out


def jt_add(acc, rc):
    top3 = set(rc["result"]["order"][:3])
    for h in rc["horses"]:
        if h.get("scratched"):
            continue
        for k in ("jockey", "trainer"):
            name = M.jt_name(h, k)
            if name:
                v = acc[k].setdefault(name, [0, 0])
                v[0] += 1
                v[1] += h["umaban"] in top3
        name = M.jt_name(h, "jockey")
        if name:
            for k, ctx in M.jt_ctx(rc).items():        # 騎手の競馬場・コース・距離帯ごと
                v = acc.setdefault(k, {}).setdefault(name + "|" + ctx, [0, 0])
                v[0] += 1
                v[1] += h["umaban"] in top3


def speed_table(races, until=None):
    """走破タイム・上がり3Fの指数の表。基準・クラスの差は until より前の走りだけで求め、
    馬場差はすべての日について(その日のタイムだけで決まる)"""
    recs = SP.collect(races, M.center_of)
    base = [r for r in recs if not until or r["date"].replace("-", "") < until]
    tb = SP.fit(base)
    SP.extend(tb, recs)
    # 競馬場ごとの目盛りのずれ(同じ馬が別の場で続けて走ったときの指数の差)。基準と同じ期間だけで求める
    tb["lvl"], n = SP.fit_levels([(d, rc) for d, rc in races if not until or d < until], tb)
    print("競馬場のずれ", len(tb["lvl"]), "場 / 組", n)
    return tb


def build(races, norm, sp=None):
    """日付の順に作る。騎手・厩舎の成績は、その日より前のレースだけで数える(先の結果を使わない)"""
    rows = []
    acc = {"jockey": {}, "trainer": {}, "jockey_venue": {}, "jockey_course": {}, "jockey_dist": {}}
    cur, snap, todays = None, None, []
    for d, rc in races:
        if d != cur:
            for x in todays:
                jt_add(acc, x)
            todays, cur = [], d
            snap = jt_snapshot(acc)
        today = datetime.strptime(d, "%Y%m%d").date()
        pred = L.predict(rc, today, use_model=False)      # 材料は手作りの評価だけ
        top3 = set(rc["result"]["order"][:3])
        ninki = rc["result"].get("ninki") or {}
        for u, f in M.race_rows(rc, today, pred, norm, snap, sp):
            rows.append({"date": d, "rid": rc["id"], "u": u, "f": f, "y": int(u in top3),
                         "ninki": ninki.get(str(u))})
        todays.append(rc)
    for x in todays:
        jt_add(acc, x)
    return rows, jt_snapshot(acc)


def group(rows):
    """rid -> [row]"""
    g = {}
    for r in rows:
        g.setdefault(r["rid"], []).append(r)
    return g


def wide_bets(h, t, others):
    """アプリと同じ買い方(keiba_logic.TH["bet_mode"])"""
    if L.TH["bet_mode"] == "wide1":
        return [([h, t], 1)]
    a = others[:2]
    return ([([h, t], 5)] + [([h, x], 1) for x in a] + [([t, x], 1) for x in a])


def settle_wide(bets, pay):
    st = rt = 0
    hit = False
    for c, u in bets:
        st += 100 * u
        y = pay.get("-".join(str(x) for x in sorted(c)))
        if y:
            rt += y * u
            hit = True
    return st, rt, hit


BET_NAME = {"wide": "ワイド◎-○", "trio": "3連複◎○▲"}


def evaluate(rows, p, races_by_id, pick, label, bet="wide"):
    """pick(rid, rs, p) -> (◎, ○, 相手) か None(見送り)。bet … wide(アプリのワイド)/ trio(3連複◎-○-相手の1頭目の1点)"""
    tot = [0, 0, 0, 0]
    for rid, rs in group(rows).items():
        sel = pick(rid, rs, p)
        if not sel or (bet == "trio" and not sel[2]):
            continue
        pay = (races_by_id[rid]["result"].get("payout") or {}).get("3連複" if bet == "trio" else "ワイド") or {}
        st, rt, hit = settle_wide([([sel[0], sel[1], sel[2][0]], 1)] if bet == "trio" else wide_bets(*sel), pay)
        tot[0] += 1
        tot[1] += hit
        tot[2] += st
        tot[3] += rt
    n, hit, st, rt = tot
    return {"label": label, "races": n, "hit": round(hit / n, 3) if n else 0, "stake": st, "ret": rt,
            "roi": round(rt / st, 3) if st else 0, "bet": BET_NAME[bet]}


def pick_model(t2, gap, pr, axis_ok=True, t1=0.0, t2_max=1.01, n_min=0):
    """アプリと同じ選び方: ◎○=軸にできる馬(延長などでない)の確率上位2頭、
    相手=残りの確率上位2頭。◎の確率が t1 以上、○の確率が t2 以上 t2_max 未満で、
    残りの1番手との差が gap 以上、頭数が n_min 以上なら買う"""
    def f(rid, rs, p):
        if (rs[0]["rid"], rs[0]["u"]) not in pr:
            return None
        o = sorted(rs, key=lambda r: -pr[(r["rid"], r["u"])])
        ok = [r for r in o if not (axis_ok and r["f"].get("rule_axis_ng"))]
        if len(o) < max(5, n_min) or len(ok) < 2:
            return None
        h, t = ok[0], ok[1]
        rest = [r for r in o if r is not h and r is not t]
        ph, pt, p3 = pr[(h["rid"], h["u"])], pr[(t["rid"], t["u"])], pr[(rest[0]["rid"], rest[0]["u"])]
        if ph < t1 or pt < t2 or pt >= t2_max or pt - p3 < gap:
            return None
        return h["u"], t["u"], [r["u"] for r in rest]
    return f


def by_market(sel):
    """sel で選んだレースで、1・2番人気を◎○にする(確定人気。参考)"""
    def f(rid, rs, p):
        if not sel(rid, rs, p):
            return None
        o = sorted((r for r in rs if r["ninki"]), key=lambda r: r["ninki"])
        if len(o) < 4:
            return None
        return o[0]["u"], o[1]["u"], [r["u"] for r in o[2:]]
    return f


def load_wf():
    """月ごとの確率 (rid, 馬番) -> 確率。無ければ None"""
    import gzip
    try:
        with gzip.open(os.path.join(K.OUT, "model_wf.json.gz"), "rt", encoding="utf-8") as f:
            return {(rid, int(u)): v for rid, d in json.load(f)["probs"].items() for u, v in d.items()}
    except (OSError, ValueError, KeyError):
        return None


CENTERS = (("nar", "地方"), ("jra", "中央"))


def of_center(rows, c):
    """中央(jra)・地方(nar)の行だけ"""
    v = 0 if c == "jra" else 1
    return [x for x in rows if x["f"].get("center") == v]


def pick_sel(sel, pr):
    """model_select の形の基準で選ぶ"""
    return pick_model(sel["t2"], sel.get("gap", 0.0), pr, True, sel.get("t1", 0.0), sel.get("t2_max", 1.01),
                      sel.get("n_min", 0))


def check_wf(rows, races_by_id, pw, sel, bet="wide"):
    """月ごとの確率で、基準 sel の成績(CHECK_PERIODS ごと / 全期間の同じレースで人気上位)。表の行のリスト"""
    rs = [x for x in rows if (x["rid"], x["u"]) in pw]
    out = []
    for name, a, b in CHECK_PERIODS:
        sub = [x for x in rs if a <= x["date"] < b]
        if sub:
            out.append(evaluate(sub, None, races_by_id, pick_sel(sel, pw), name, bet))
    out.append(evaluate(rs, None, races_by_id, by_market(pick_sel(sel, pw)),
                        "同じレースで{}番人気({}〜。参考)".format("1〜3" if bet == "trio" else "1・2",
                                                         min(x["date"] for x in rs)[:4] if rs else ""), bet))
    return out


def center_bet(c):
    """アプリの買い方(中央は TH["jra_bet"] が trio1 なら3連複◎○▲)"""
    return "trio" if c == "jra" and L.TH.get("jra_bet") == "trio1" else "wide"


def check_centers(rows, races_by_id, pw, sels):
    """中央・地方それぞれの基準で、それぞれのレースだけを、アプリと同じ買い方で確かめる"""
    return {c: check_wf(of_center(rows, c), races_by_id, pw, sels[c], center_bet(c)) for c, _ in CENTERS}


def show_check(chk):
    for c, name in CENTERS:
        if c not in chk:
            continue
        print("  {}".format(name))
        for r in chk[c]:
            print("   {:<28} {:>5}R 的中{:>4.0%} 回収率{:>4.0%}".format(r["label"], r["races"], r["hit"], r["roi"]))


def fixed_selects():
    """利用者が決めた中央・地方の基準(keiba_logic.MODEL_SELECT_FIXED)を model_select の形で"""
    return {c: {k: v for k, v in L.model_select(c).items() if k in ("t1", "t2", "t2_max", "gap", "n_min", "a_p1")}
            for c, _ in CENTERS}


def update_select(rows, races_by_id):
    """学び直さずに、基準と月ごとの確率での成績だけを keiba_model_meta.json に書き直す"""
    meta = json.load(open(M.META_FILE, encoding="utf-8"))
    pw = load_wf()
    if L.MODEL_SELECT_FIXED:
        meta["select"] = fixed_selects()
        meta["select_from"] = "fixed"
    sels = meta["select"]
    if "nar" not in sels:                   # 古い形(中央・地方で同じ基準)
        sels = {c: sels for c, _ in CENTERS}
    meta["check_wf"] = check_centers(rows, races_by_id, pw, sels) if pw else None
    print("基準", json.dumps(sels, ensure_ascii=False))
    if pw:
        show_check(meta["check_wf"])
    json.dump(meta, open(M.META_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("保存", M.META_FILE)


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-speed", action="store_true", help="走破タイムの指数を使わない(比べるため)")
    ap.add_argument("--update-select", action="store_true",
                    help="学び直さずに、今の月ごとの確率で基準の成績だけ出し直して keiba_model_meta.json に書く")
    ap.add_argument("--walkforward", action="store_true",
                    help="月ごとに、その月より前だけで学んだモデルで確率を出して keiba/model_wf.json.gz に保存する"
                         "(過去の日の成績を、先の結果を知らないモデルで見るため)")
    a = ap.parse_args()
    import lightgbm as lgb

    races = load_races()
    print("レース", len(races), races[0][0], "〜", races[-1][0])
    cache = CACHE + (".nospeed" if a.no_speed else "")
    if os.path.exists(cache) and not a.rebuild:
        norm, rows, jt_all, sp = pickle.load(open(cache, "rb"))
    else:
        norm = l3f_norms(races)
        sp = None if a.no_speed else speed_table(races, VALID_FROM)
        rows, jt_all = build(races, norm, sp)
        pickle.dump((norm, rows, jt_all, sp), open(cache, "wb"))
    # 3年分になるとメモリが足りなくなるので、手作りの評価で比べる期間(調整・試験)より前は結果だけ残す
    races_by_id = {rc["id"]: (rc if d >= VALID_FROM else {"id": rc["id"], "center": rc.get("center"),
                                                         "result": rc["result"]}) for d, rc in races}
    span = (races[0][0], races[-1][0], len(races))
    del races
    if a.update_select:
        update_select(rows, races_by_id)
        return
    feats = sorted({k for r in rows for k in r["f"]})
    X = np.empty((len(rows), len(feats)))
    for i, r in enumerate(rows):
        X[i] = [M._num(r["f"].get(k)) for k in feats]
    y = np.array([r["y"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    tr, va, te = dates < VALID_FROM, (dates >= VALID_FROM) & (dates < TEST_FROM), dates >= TEST_FROM
    print("馬 学習{} 調整{} 試験{} / 特徴量{}".format(tr.sum(), va.sum(), te.sum(), len(feats)))
    cat = [feats.index(k) for k in ("venue", "style") if k in feats]
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 80,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
              "verbose": -1, "seed": 1}
    dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(X[va], y[va], categorical_feature=cat, reference=dtr)
    bst = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    best_iter = bst.best_iteration
    p = bst.predict(X, num_iteration=best_iter)
    pr = {(r["rid"], r["u"]): v for r, v in zip(rows, p)}
    rule = np.array([M._num(r["f"].get("rule_score")) for r in rows])
    mkt = np.array([-(r["ninki"] or 99) for r in rows], dtype=float)
    for name, m in (("調整", va), ("試験", te)):
        print("{} AUC モデル {:.3f} / 手作り {:.3f} / 市場(確定人気) {:.3f}".format(
            name, auc(y[m], p[m]), auc(y[m], rule[m]), auc(y[m], mkt[m])))

    # ---- 買い方の比較
    def by_rule(rid, rs, p):
        pred = L.predict(races_by_id[rid], datetime.strptime(rs[0]["date"], "%Y%m%d").date(), use_model=False)
        if pred.get("verdict") not in ("A", "B"):
            return None
        return pred["honmei"], pred["taikou"], [x for x in pred.get("aite", []) if x not in (pred["honmei"], pred["taikou"])]

    def by_model(t2, gap, axis_ok=True, pr=pr):
        return pick_model(t2, gap, pr, axis_ok)

    print("\n■ 調整期間で、モデルで買うレースの基準を探す(○の確率 t2 以上、○と3番手の差 gap 以上)")
    grid = []
    for t2 in (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        for gap in (0.0, 0.05, 0.10, 0.15, 0.20):
            for axis_ok in (True,):              # 規則 15(延長は軸にしない)は守る
                r = evaluate([x for x in rows if va[rows.index(x)]] if False else [x for x, m in zip(rows, va) if m],
                             p, races_by_id, by_model(t2, gap, axis_ok), "")
                if r["races"] >= 150:
                    grid.append((r["roi"], t2, gap, axis_ok, r))
    grid.sort(key=lambda g: -g[0])
    for roi, t2, gap, axis_ok, r in grid[:8]:
        print("  t2={:.2f} gap={:.2f} 延長を軸にしない={} → {}R 的中{:.0%} 回収率{:.0%}".format(t2, gap, axis_ok, r["races"], r["hit"], r["roi"]))
    # 回収率が最良から1ポイント以内の中で、いちばん多くのレースを買える基準にする(たまたまに合わせない)
    top = grid[0][0]
    _, t2, gap, axis_ok, _ = max((g for g in grid if g[0] >= top - 0.01), key=lambda g: g[4]["races"])
    print("  → 使う基準: t2={:.2f} gap={:.2f}".format(t2, gap))

    print("\n■ 結果(買い方はどれもワイドだけ)")
    out = {}
    for name, m in (("調整", va), ("試験", te)):
        rs = [x for x, k in zip(rows, m) if k]
        res = [evaluate(rs, p, races_by_id, by_rule, "手作り(今の◎○・A/B)"),
               evaluate(rs, p, races_by_id, by_model(0.0, 0.0, axis_ok), "モデル(全レース)"),
               evaluate(rs, p, races_by_id, by_model(t2, gap, axis_ok), "モデル(基準で選ぶ)"),
               evaluate(rs, p, races_by_id, by_market(by_model(t2, gap, axis_ok)), "市場 同じレースで1・2番人気"),
               evaluate(rs, p, races_by_id, by_market(lambda *a: True), "市場 全レースで1・2番人気")]
        out[name] = res
        print(" " + name)
        for r in res:
            print("   {:<26} {:>5}R 的中{:>4.0%} 投資{:>10,} 払戻{:>10,} 回収率{:>4.0%}".format(
                r["label"], r["races"], r["hit"], r["stake"], r["ret"], r["roi"]))

    imp = sorted(zip(bst.feature_importance("gain"), feats), reverse=True)[:15]
    print("\n■ 効いている特徴量(上位15)")
    print("  " + " / ".join("{} {:.0f}".format(k, v) for v, k in imp))

    wf_check, select_from = None, "valid"
    sels = {c: {"t2": t2, "gap": gap} for c, _ in CENTERS}
    if not a.walkforward:
        # 月ごとの確率で選んだ基準があれば、それを引き継ぐ(1回の分け方で選ぶより偏りが少ない)
        try:
            old_meta = json.load(open(M.META_FILE, encoding="utf-8"))
            if old_meta.get("select_from") == "walkforward":
                sels = old_meta["select"]
                if "nar" not in sels:
                    sels = {c: {"t2": sels["t2"], "gap": sels["gap"]} for c, _ in CENTERS}
                wf_check, select_from = old_meta.get("check_wf"), "walkforward"
                print("  → 月ごとの確率で選んだ基準を引き継ぐ:", json.dumps(sels, ensure_ascii=False))
        except (OSError, ValueError, KeyError):
            pass
    if a.walkforward:
        wf = {}
        months = sorted({d[:6] for d in dates})
        for mth in [m for m in months[1:] if m >= WF_FROM]:
            m_tr, m_pr = dates < mth + "01", np.array([d[:6] == mth for d in dates])
            b = lgb.train(params, lgb.Dataset(X[m_tr], y[m_tr], categorical_feature=cat), best_iter)
            pm = b.predict(X[m_pr])
            sub = [r for r, k in zip(rows, m_pr) if k]
            byr = {}
            for r, v in zip(sub, pm):
                byr.setdefault(r["rid"], {})[str(r["u"])] = float(v)
            for rid, d in byr.items():
                tot = sum(d.values())
                want = min(3, len(d))
                wf[rid] = {u: round(min(0.99, v * want / tot), 4) for u, v in d.items()}
            print("  {} 学習{}頭 → {}レース".format(mth, int(m_tr.sum()), len(byr)), flush=True)
        import gzip
        with gzip.open(os.path.join(K.OUT, "model_wf.json.gz"), "wt", encoding="utf-8") as f:
            json.dump({"note": "月ごとに、その月より前のレースだけで学んだモデルの3着以内の確率",
                       "made": datetime.now().strftime("%Y-%m-%d"), "probs": wf}, f, separators=(",", ":"))
        print("保存", os.path.join(K.OUT, "model_wf.json.gz"), len(wf), "レース")
        # 買うレースの基準を、月ごとの確率(先を知らない)の2月〜調整期間の終わりで選び直し、試験期間で確かめる
        pw = {(rid, int(u)): v for rid, d in wf.items() for u, v in d.items()}
        rs_sel = [x for x, d in zip(rows, dates) if d < TEST_FROM and (x["rid"], x["u"]) in pw]
        print("\n■ 月ごとの確率で、買うレースの基準を選び直す({}〜{}で選び、{}〜で確かめる)".format(WF_FROM, TEST_FROM[:6], TEST_FROM[:6]))
        g2 = []
        for t2w in (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            for gw in (0.0, 0.05, 0.10, 0.15, 0.20):
                r = evaluate(rs_sel, None, races_by_id, by_model(t2w, gw, True, pw), "")
                if r["races"] >= 600:
                    g2.append((r["roi"], t2w, gw, r))
        g2.sort(key=lambda g: -g[0])
        for roi, t2w, gw, r in g2[:6]:
            print("  t2={:.2f} gap={:.2f} → {}R 的中{:.0%} 回収率{:.0%}".format(t2w, gw, r["races"], r["hit"], r["roi"]))
        _, t2, gap, _ = max((g for g in g2 if g[0] >= g2[0][0] - 0.01), key=lambda g: g[3]["races"])
        sels = {c: {"t2": t2, "gap": gap} for c, _ in CENTERS}
        wf_check = check_centers(rows, races_by_id, pw, sels)
        select_from = "walkforward"
        print("  → 使う基準: t2={:.2f} gap={:.2f}".format(t2, gap))
        show_check(wf_check)
    # 利用者が決めた基準があれば、それを使う(中央・地方で別。成績は月ごとの確率で出し直す)
    if L.MODEL_SELECT_FIXED:
        sels = fixed_selects()
        select_from = "fixed"
        pw = pw if a.walkforward else load_wf()
        wf_check = check_centers(rows, races_by_id, pw, sels) if pw else None
        print("\n  → 利用者が決めた基準を使う:", json.dumps(sels, ensure_ascii=False))
        if wf_check:
            show_check(wf_check)

    if a.save:
        allm = np.ones(len(rows), dtype=bool)
        dall = lgb.Dataset(X[allm], y[allm], categorical_feature=cat)
        final = lgb.train(params, dall, int(best_iter * 1.15))
        final.save_model(M.MODEL_FILE)
        if sp:
            SP.save(sp)
        # 騎手・厩舎は全期間の成績を持たせる(当日の予想ではこれを使う)。乗った数が少ない人は省く
        jt_save = {k: {"base": v["base"], "by": {n: c for n, c in v["by"].items() if c[0] >= 5}}
                   for k, v in jt_all.items()}
        meta = {"features": feats, "l3f_norm": norm, "jt": jt_save, "trained": datetime.now().strftime("%Y-%m-%d"),
                "data_end": span[1],
                "data": "{}〜{} {}レース".format(*span),
                "iterations": int(best_iter * 1.15),
                "select": sels,
                "centers": list(K.CENTERS),
                "select_from": select_from,
                "check_wf": wf_check,
                "check": {k: v for k, v in out.items()},
                "note": "選ぶ基準は調整期間(7〜8月)で決め、試験期間(9月〜)は学習にも基準にも使っていない。"
                        "保存したモデルは全期間で学び直したもの"}
        json.dump(meta, open(M.META_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("保存", M.MODEL_FILE, M.META_FILE)


if __name__ == "__main__":
    main()

"""手元の材料すべて(モデルの155個 + 騎手・厩舎の動きなど)に、ワイドの市場がまだ値段に入れていない情報があるか(中央 2022〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl python tools/experiments/beyond_wide_all.py
  3着内に来たか を、まず市場だけ(ワイドの確定オッズから出した強さ と 頭数)で当てる。その残り(市場が外した分)を、
  材料すべてで LightGBM に学ばせる(2022〜2024年前半で学び、2024年後半で回数を決め、2025・2026年で確かめる)。
  確かめる年の対数尤度が市場だけより良くなれば、材料の中に市場が見落としている情報がある。単勝オッズを市場にした場合も並べる。
  足した材料: 乗り替わり・その日の騎乗数・同じ厩舎の出走数(多頭出し)・厩舎の東西と遠征・開催の何日目・騎手×厩舎の組の成績・
  3歳が古馬と走る・ハンデ戦・牝馬限定・1着賞金・レース番号"""
import gc, os, pickle, re, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
import keiba_final as F
import cross_pool as CP
import lightgbm as lgb
EAST = ("札幌", "函館", "福島", "新潟", "東京", "中山")
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
extra, wide, qwin = {}, {}, {}
last_jk, combo, same_jk = {}, defaultdict(lambda: [0, 0]), defaultdict(int)
cur, todays = None, []


def flush(todays):
    for rc in todays:
        top3 = set(rc["result"]["order"][:3])
        for h in rc["horses"]:
            if h.get("scratched"):
                continue
            c = combo[(h.get("jockey"), h.get("trainer"))]; c[0] += 1; c[1] += h["umaban"] in top3
            same_jk[(h.get("horse_id"), h.get("jockey"))] += 1
            last_jk[h.get("horse_id")] = h.get("jockey")


by_day = defaultdict(list)
for d, rc in races:
    by_day[d].append(rc)
for d in sorted(by_day):
    rides = defaultdict(int)
    for rc in by_day[d]:
        for h in rc["horses"]:
            if not h.get("scratched"):
                rides[(rc.get("venue"), h.get("jockey"))] += 1
    for rc in by_day[d]:
        rid = rc["id"]; cond = rc.get("cond") or ""
        m = re.search(r"本賞金:([\d,]+?),", cond)
        ntr = defaultdict(int)
        for h in rc["horses"]:
            if not h.get("scratched"):
                ntr[h.get("trainer")] += 1
        nk = {int(u): k for u, k in (rc["result"].get("ninki") or {}).items() if k}
        od = {int(u): o for u, o in (rc["result"].get("odds") or {}).items() if o}
        w = CP.wide_strength({"r": rc, "nk": nk}) if len(nk) >= 5 else None
        tot = sum(1 / o for o in od.values()) or 1
        for h in rc["horses"]:
            if h.get("scratched"):
                continue
            u = h["umaban"]; tr = h.get("trainer") or ""
            reg = 0 if tr.startswith("美浦") else 1 if tr.startswith("栗東") else 2
            east = rc.get("venue") in EAST
            c = combo.get((h.get("jockey"), tr), [0, 0])
            pj = last_jk.get(h.get("horse_id"))
            age = int(re.sub(r"\D", "", h.get("sex_age") or "0") or 0)
            extra[(rid, u)] = {
                "x_jk_change": np.nan if pj is None else float(pj != h.get("jockey")),
                "x_jk_rides_day": rides[(rc.get("venue"), h.get("jockey"))],
                "x_same_jk_n": same_jk.get((h.get("horse_id"), h.get("jockey")), 0),
                "x_tr_multi": ntr[tr], "x_tr_region": reg, "x_ensei": float((reg == 0 and not east) or (reg == 1 and east)),
                "x_kaisai_day": int(rid[8:10]), "x_kaisai_kai": int(rid[6:8]),
                "x_combo_n": c[0], "x_combo_top3": c[1] / c[0] if c[0] >= 5 else np.nan,
                "x_3yo_vs_old": float(age == 3 and "3歳以上" in cond), "x_hande": float("ハンデ" in cond), "x_hinba": float("牝" in cond.split("頭")[0] and "牝[" in cond or "牝 " in cond),
                "x_prize": float(m.group(1).replace(",", "")) if m else np.nan, "x_rno": int(rid[10:12]),
            }
            if w and u in w:
                wide[(rid, u)] = min(w[u], .98)
            if u in od:
                qwin[(rid, u)] = min(3 / od[u] / tot, .98)
    flush(by_day[d])
del races, by_day; gc.collect()

keep = [i for i, r in enumerate(rows) if (r["rid"], r["u"]) in wide and (r["rid"], r["u"]) in qwin]
rows = [rows[i] for i in keep]
feats = sorted({k for r in rows for k in r["f"]}); xf = sorted(next(iter(extra.values())))
X = np.array([[M._num(r["f"].get(k)) for k in feats] + [extra[(r["rid"], r["u"])][k] for k in xf] for r in rows], dtype=float)
names = feats + xf
y = np.array([r["y"] for r in rows], dtype=float); dts = np.array([r["date"] for r in rows])
nrun = np.array([M._num(r["f"].get("n")) for r in rows])
lgt = lambda p: np.log(p / (1 - p))
base_x = {"ワイド": lgt(np.array([wide[(r["rid"], r["u"])] for r in rows])), "単勝": lgt(np.array([qwin[(r["rid"], r["u"])] for r in rows]))}
tr, va, te = dts < "20240701", (dts >= "20240701") & (dts < "20250101"), dts >= "20250101"
print("馬 {}頭({}レース)・材料 {}個(うち足したもの {}個)  学ぶ {} / 回数を決める {} / 確かめる {}  ({:.0f}秒)".format(
    len(rows), len({r["rid"] for r in rows}), len(names), len(xf), tr.sum(), va.sum(), te.sum(), time.time() - t0), flush=True)
cat = [names.index(k) for k in ("venue", "style") if k in names]


def ll(yy, z):
    p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return float(np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


params = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.6,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1}
for mk in ("単勝", "ワイド"):
    B = np.column_stack([base_x[mk], np.log(nrun)])
    mu, sd = B[tr | va].mean(0), B[tr | va].std(0)
    w = np.array(F.fit_logit(((B[tr | va] - mu) / sd).tolist(), y[tr | va].tolist(), 0.01))
    z0 = np.column_stack([np.ones(len(B)), (B - mu) / sd]) @ w
    print("\n■ 市場 = {}のオッズ".format(mk))
    print("  市場だけ                     確かめる年の対数尤度 {:.5f}(2025年 {:.5f}・2026年 {:.5f})".format(
        ll(y[te], z0[te]), ll(y[te & (dts < "20260101")], z0[te & (dts < "20260101")]), ll(y[te & (dts >= "20260101")], z0[te & (dts >= "20260101")])))
    for label, cols in (("+ モデルの材料155個", list(range(len(feats)))), ("+ 足した材料{}個だけ".format(len(xf)), list(range(len(feats), len(names)))),
                        ("+ 全部", list(range(len(names))))):
        cc = [cols.index(c) for c in cat if c in cols]
        best = None
        for seed in (1, 2, 3):
            dtr = lgb.Dataset(X[tr][:, cols], y[tr], init_score=z0[tr], categorical_feature=cc)
            dva = lgb.Dataset(X[va][:, cols], y[va], init_score=z0[va], categorical_feature=cc, reference=dtr)
            b = lgb.train(dict(params, seed=seed), dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            z = z0[te] + b.predict(X[te][:, cols], raw_score=True, num_iteration=b.best_iteration)
            best = (best[0] + z / 3, best[1] + [b.best_iteration], b) if best else (z / 3, [b.best_iteration], b)
        z, its, b = best
        m25, m26 = dts[te] < "20260101", dts[te] >= "20260101"
        print("  {:22s} 確かめる年の対数尤度 {:.5f}(2025年 {:.5f}・2026年 {:.5f})  差 {:+.5f}  回数 {}".format(
            label, ll(y[te], z), ll(y[te][m25], z[m25]), ll(y[te][m26], z[m26]), ll(y[te], z) - ll(y[te], z0[te]), its), flush=True)
        if label == "+ 全部":
            imp = sorted(zip(b.feature_importance("gain"), [names[c] for c in cols]), reverse=True)[:12]
            print("    効いた材料: " + " / ".join("{} {:.0f}".format(k, v) for v, k in imp))
            d = z - z0[te]
            qs = np.quantile(d, [0, .1, .3, .7, .9, 1])
            print("    市場からのずれの帯ごと(確かめる年): 実際 / 市場の見込み / 直した見込み")
            for a, c in zip(qs[:-1], qs[1:]):
                m = (d >= a) & (d <= c)
                print("      ずれ {:+.3f}〜{:+.3f}  {:6d}頭  実際 {:5.1%}  市場 {:5.1%}  直した {:5.1%}".format(
                    a, c, m.sum(), y[te][m].mean(), (1 / (1 + np.exp(-z0[te][m]))).mean(), (1 / (1 + np.exp(-z[m]))).mean()))
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

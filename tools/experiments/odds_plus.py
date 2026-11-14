"""オッズに評価値を1つずつ足すと、的中率・回収率は上がるか(中央 2022〜2026年)
  KEIBA_CACHE=/path/to/cache.pkl PRED=/path/to/pred.npy PWIN=/path/to/pwin.npy OUTDIR=/path/to/dir python tools/experiments/odds_plus.py
  PRED … fresh_years.py(YEARS=2022,...,2026)の3着内の確率、PWIN … hit_rate.py が作る1着の確率(どちらもその年を除いて学んだもの)。

  やり方(どれも「4年で式を作り、残りの1年で確かめる」を5回):
   1着 … 土台は単勝オッズだけ(レースの中で比べる条件つきロジット: 点 = b0×log(オッズから出した確率))。
          これに評価値を1つ足す(点 = b0×log(確率) + b1×評価値)。評価値はレースの中で平均0・ばらつき1にそろえる。
   ワイド … 土台は3連複から見た組の確率(0.75×組の強さ)だけ。これに2頭の評価値の合計を足す(ロジスティック回帰)。
  見るもの(確かめる年を5年分合わせた数字):
   当たり方 … 対数尤度の伸び(1レースあたり×1000。+なら良くなった)と、良くなった年の数(5年のうち)
   的中率 … 全レースで1点ずつ買ったとき / 「堅い買い目」と同じ数だけ、自信のあるレースから買ったとき
   回収率 … 上と同じ買い方 / 評価値がいちばん押し上げる1割の馬(組)だけ買ったとき(= オッズより上と見た所が本当に得か)
  くらべる物差しとして、でたらめな数字(乱数)を足した場合も出す。最後に、評価値を全部まとめて足した場合"""
import gc, json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import train_model as T
import keiba_model as M
OUT = os.environ.get("OUTDIR", "/tmp/odds_plus"); os.makedirs(OUT, exist_ok=True)
YS = ("2022", "2023", "2024", "2025", "2026")
WIN_ODDS, WIDE_P, PT = 1.8, 0.45, 0.75            # 堅い買い目の基準(keiba_logic.SOLID)
t0 = time.time()
races = T.load_races()
norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
info = {rc["id"]: {"result": rc["result"], "tp": rc.get("trio_pair_final") or {}, "wf": rc.get("wide_final") or {}} for d, rc in races}
del races; gc.collect()
p3 = np.load(os.environ["PRED"]); pw1 = np.load(os.environ["PWIN"])

# ---- 評価値の一覧(レースの中で差が付くものだけ。_rank・_dmean などはレースの中でそろえ直した値なので元の値だけ使う)
SUF = ("_dhi", "_dlo", "_dmean", "_rank")
CATS = ("style", "sex")
SKIP = ("venue", "surface", "center", "course_kata")
allf = sorted({k for r in rows for k in r["f"]})
feats = [k for k in allf if not k.endswith(SUF) and k not in CATS and k not in SKIP]

by = defaultdict(list)
for i, r in enumerate(rows):
    if r["f"].get("center") == 0 and r["rid"] in info and not np.isnan(p3[i]) and not np.isnan(pw1[i]):
        by[r["rid"]].append(i)
ridx, hrow, sizes = [], [], []          # レースごと: rid / 馬ごと: rows の番号
H = defaultdict(list)                   # 馬ごとの列
for rid in sorted(by, key=lambda x: (rows[by[x][0]]["date"], x)):
    res = info[rid]["result"]; order = res.get("order") or []
    od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
    ix = [i for i in by[rid] if rows[i]["u"] in od]
    if len(order) < 3 or len(ix) < 5 or order[0] not in {rows[i]["u"] for i in ix}:
        continue
    q = np.array([1 / od[rows[i]["u"]] for i in ix]); q = q / q.sum()
    ridx.append(rid); sizes.append(len(ix))
    for j, i in enumerate(ix):
        u = rows[i]["u"]
        hrow.append(i); H["u"].append(u); H["logq"].append(np.log(q[j])); H["odds"].append(od[u])
        H["win"].append(int(u == order[0])); H["top3"].append(int(u in order[:3]))
sizes = np.array(sizes); NR = len(sizes); starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
hrow = np.array(hrow); NH = len(hrow)
H = {k: np.array(v, dtype=float) for k, v in H.items()}
race_of = np.repeat(np.arange(NR), sizes)
ryear = np.array([rows[by[rid][0]]["date"][:4] for rid in ridx]); hyear = ryear[race_of]
print("中央 {}R・{}頭 ({:.0f}分)".format(NR, NH, (time.time() - t0) / 60), flush=True)


def zrace(x):
    """レースの中で平均0・ばらつき1にそろえる(無い値は0 = 平均。±3で切る)"""
    ok = ~np.isnan(x); x0 = np.where(ok, x, 0.0)
    n = np.add.reduceat(ok.astype(float), starts); s = np.add.reduceat(x0, starts)
    mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    d = np.where(ok, x - np.repeat(mean, sizes), 0.0)
    sd = np.sqrt(np.add.reduceat(d * d, starts) / np.maximum(n, 1))
    z = d / np.repeat(np.where(sd > 1e-9, sd, np.inf), sizes)
    return np.clip(z, -3, 3)


raw = np.array([[M._num(rows[i]["f"].get(k)) for k in feats] for i in hrow])
Z, names = [], []
for j, k in enumerate(feats):
    z = zrace(raw[:, j])
    if np.abs(z).sum() > 0:
        Z.append(z); names.append(k)
for c in CATS:                           # 脚質・性別は、種類ごとの 0/1 にする
    v = np.array([M._num(rows[i]["f"].get(c)) for i in hrow])
    for val in sorted(set(v[~np.isnan(v)])):
        if (v == val).mean() > 0.02:
            Z.append(zrace((v == val).astype(float))); names.append("{}={:g}".format(c, val))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
Z.append(zrace(lg(p3[hrow]))); names.append("model_p3")
Z.append(zrace(np.log(np.clip(pw1[hrow], 1e-4, 1)))); names.append("model_p1")
rng = np.random.default_rng(7)
NRAND = 20
for j in range(NRAND):                   # 物差し: でたらめな数字
    Z.append(zrace(rng.normal(size=NH))); names.append("rand{}".format(j + 1))
Z = np.array(Z).T; NF = Z.shape[1]
del raw; gc.collect()
print("評価値 {}個(うち乱数 {}個)".format(NF, NRAND), flush=True)

JP = {"model_p3": "モデルの3着内の確率", "model_p1": "モデルの1着の確率", "rule_score": "30か条の評価(点)", "rule_ability": "能力(着差とクラス)",
      "rule_plus": "加点の数", "rule_minus": "減点の数", "rule_cut": "負け要因が重なる", "rule_axis_ng": "軸にできない",
      "spd_adj_mean3": "タイム指数(直し・3走平均)", "spd_adj_best": "タイム指数(直し・最高)", "spd_adj_last": "タイム指数(直し・前走)",
      "spd_mean3": "タイム指数(3走平均)", "spd_best": "タイム指数(最高)", "spd_last": "タイム指数(前走)", "spd_best_surf": "タイム指数(同じ馬場の最高)",
      "spd_best_dist": "タイム指数(近い距離の最高)", "l3fig_best": "上がり指数(最高)", "l3fig_mean": "上がり指数(平均)",
      "l3f_dev_best": "上がりの速さ(最高)", "l3f_dev_mean": "上がりの速さ(平均)", "margin_mean": "着差(平均)", "margin_min": "着差(いちばん良い)",
      "margin_last": "着差(前走)", "sc_mean": "クラスを入れた着差(平均)", "sc_best2": "クラスを入れた着差(上位2走)", "sc_last": "クラスを入れた着差(前走)",
      "sc_max": "クラスを入れた着差(最高)", "jockey_top3": "騎手の3着内率", "jockey_n": "騎手の騎乗数", "trainer_top3": "厩舎の3着内率",
      "trainer_n": "厩舎の出走数", "course_style_idx": "コース傾向(脚質)", "course_waku_idx": "コース傾向(枠)", "dist_change": "距離の変化",
      "dist_from_best": "得意な距離との差", "kinryo": "斤量", "kinryo_change": "斤量の変化", "weight": "馬体重", "wdiff": "馬体重の増減",
      "rest_days": "休みの長さ", "pop_last": "前走の人気", "pop_mean": "人気(平均)", "pop_min": "人気(いちばん上)", "beat_pop": "人気より上に来るか",
      "rank_ratio_mean": "着順(平均)", "rank_ratio_last": "着順(前走)", "top3_rate": "3着内率", "win_rate": "勝率", "n_runs": "出走数",
      "waku": "枠", "umaban_ratio": "馬番(内か外か)", "age": "年齢", "nige_rate": "逃げた割合", "pos_first": "最初の位置取り",
      "cls_diff_last": "クラスの上げ下げ(前走)", "cls_diff_max": "クラスの差(最高)", "same_venue_runs": "同じ競馬場の経験", "same_venue_last": "前走も同じ競馬場",
      "same_surface": "同じ馬場の経験", "n_venues": "走った競馬場の数", "from_jra": "中央から", "jra_share": "中央の割合"}
jp = lambda k: JP.get(k, k)


# ---------------------------------------------------------------- 1着: 条件つきロジット
def cl_fit(F, y, sz, l2=0.0):
    st = np.concatenate([[0], np.cumsum(sz)[:-1]])
    def f(b):
        s = F @ b
        s = s - np.repeat(np.maximum.reduceat(s, st), sz)
        e = np.exp(s); se = np.add.reduceat(e, st)
        p = e / np.repeat(se, sz)
        return -(s[y == 1].sum() - np.log(se).sum()) + l2 * (b[1:] @ b[1:]) / 2, -(F.T @ (y - p)) + l2 * np.r_[0, b[1:]]
    b0 = np.zeros(F.shape[1]); b0[0] = 1.0
    return minimize(f, b0, jac=True, method="L-BFGS-B").x


def cl_prob(F, b, sz):
    st = np.concatenate([[0], np.cumsum(sz)[:-1]])
    s = F @ b; s = s - np.repeat(np.maximum.reduceat(s, st), sz)
    e = np.exp(s)
    return e / np.repeat(np.add.reduceat(e, st), sz)


def first_top(p, sz):
    """レースごとに、p がいちばん大きい馬の位置(同じなら先の馬)"""
    st = np.concatenate([[0], np.cumsum(sz)[:-1]])
    mx = np.repeat(np.maximum.reduceat(p, st), sz)
    idx = np.flatnonzero(p >= mx)
    g = np.repeat(np.arange(len(sz)), sz)[idx]
    return idx[np.unique(g, return_index=True)[1]]


def win_eval(cols, l2=0.0):
    """cols = 足す評価値の列番号(空なら土台だけ)。確かめる年ごとの数字を合わせて返す"""
    o = dict(ll=[], hit=0, pay=0.0, n=0, sn=0, shit=0, spay=0.0, up_a=0.0, up_e=0.0, up_pay=0.0, up_n=0, dn_a=0.0, dn_e=0.0, b=[], yll=[], swn=0, sw_new=0, sw_fav=0)
    for yy in YS:
        te = hyear == yy; tr = ~te
        Ftr = np.column_stack([H["logq"][tr]] + [Z[tr, c] for c in cols]); Fte = np.column_stack([H["logq"][te]] + [Z[te, c] for c in cols])
        sz_tr, sz_te = sizes[ryear != yy], sizes[ryear == yy]
        b = cl_fit(Ftr, H["win"][tr], sz_tr, l2)
        p = cl_prob(Fte, b, sz_te); y = H["win"][te]; od = H["odds"][te]
        ll = np.log(p[y == 1]); o["ll"].append(ll.sum()); o["yll"].append(ll.mean()); o["b"].append(b)
        top = first_top(p, sz_te)
        o["n"] += len(top); o["hit"] += y[top].sum(); o["pay"] += (y[top] * od[top]).sum()
        fav = first_top(H["logq"][te], sz_te)
        sw = top != fav; o["swn"] += sw.sum(); o["sw_new"] += y[top][sw].sum(); o["sw_fav"] += y[fav][sw].sum()
        # 堅い単勝と同じ数だけ、自信のあるレースから
        k = int((od[fav] <= WIN_ODDS).sum())
        sel = top[np.argsort(-p[top], kind="stable")[:k]]
        o["sn"] += k; o["shit"] += y[sel].sum(); o["spay"] += (y[sel] * od[sel]).sum()
        # 評価値がいちばん押し上げる1割・押し下げる1割の馬(単勝20倍以下の中で)。土台(オッズだけ)の見込みとくらべる
        if cols:
            pb = cl_prob(Fte[:, :1], BASE_B[yy], sz_te)
            adj = Fte[:, 1:] @ b[1:]; m = od <= 20
            hi = m & (adj >= np.quantile(adj[m], 0.9)); lo = m & (adj <= np.quantile(adj[m], 0.1))
            o["up_a"] += y[hi].sum(); o["up_e"] += pb[hi].sum(); o["up_pay"] += (y[hi] * od[hi]).sum(); o["up_n"] += hi.sum()
            o["dn_a"] += y[lo].sum(); o["dn_e"] += pb[lo].sum()
    return o


BASE_B = {}
for yy in YS:
    tr = hyear != yy
    BASE_B[yy] = cl_fit(H["logq"][tr][:, None], H["win"][tr], sizes[ryear != yy])
W0 = win_eval([])
print("\n■ 1着(単勝)。土台 = 単勝オッズだけ: 全レースで1番人気 的中 {:.1%}・回収率 {:.1f}% / 堅い単勝と同じ数({}R) 的中 {:.1%}・回収率 {:.1f}%".format(
    W0["hit"] / W0["n"], 100 * W0["pay"] / W0["n"], W0["sn"], W0["shit"] / W0["sn"], 100 * W0["spay"] / W0["sn"]), flush=True)
WR = {}
for c in range(NF):
    WR[names[c]] = win_eval([c])
print("  1つずつ足した ({:.0f}分)".format((time.time() - t0) / 60), flush=True)


def wsum(o):
    d = [(a - b) * 1000 for a, b in zip(o["yll"], W0["yll"])]
    return dict(dll=(sum(o["ll"]) - sum(W0["ll"])) / NR * 1000, yrs=sum(x > 0 for x in d), hit=o["hit"] / o["n"], roi=100 * o["pay"] / o["n"],
                shit=o["shit"] / o["sn"], sroi=100 * o["spay"] / o["sn"], up_ae=o["up_a"] / o["up_e"] if o["up_e"] else float("nan"),
                up_roi=100 * o["up_pay"] / o["up_n"] if o["up_n"] else float("nan"), dn_ae=o["dn_a"] / o["dn_e"] if o["dn_e"] else float("nan"),
                b=float(np.mean([b[1] for b in o["b"]])) if len(o["b"][0]) == 2 else None, swn=int(o["swn"]),
                sw_new=o["sw_new"] / o["swn"] if o["swn"] else float("nan"), sw_fav=o["sw_fav"] / o["swn"] if o["swn"] else float("nan"))


WS = {k: wsum(v) for k, v in WR.items()}
rnd = [WS[k] for k in WS if k.startswith("rand")]
real = sorted((k for k in WS if not k.startswith("rand")), key=lambda k: -WS[k]["dll"])
print("  乱数 {}個: 当たり方の伸び {:+.2f}〜{:+.2f} / 全レースの的中 {:.1%}〜{:.1%} / 同じ数の的中 {:.1%}〜{:.1%}・回収率 {:.1f}〜{:.1f}% / 押し上げる1割の回収率 {:.0f}〜{:.0f}%".format(
    NRAND, min(x["dll"] for x in rnd), max(x["dll"] for x in rnd), min(x["hit"] for x in rnd), max(x["hit"] for x in rnd),
    min(x["shit"] for x in rnd), max(x["shit"] for x in rnd), min(x["sroi"] for x in rnd), max(x["sroi"] for x in rnd),
    min(x["up_roi"] for x in rnd), max(x["up_roi"] for x in rnd)))
hd = "  {:28s} {:>7s} {:>4s} {:>7s} {:>6s} | {:>7s} {:>6s} | {:>8s} {:>6s} {:>8s} | {:>6s} {:>7s} {:>7s}".format(
    "評価値", "伸び", "年", "的中", "回収", "同数的中", "回収", "上1割A/E", "回収", "下1割A/E", "替わるR", "新しい馬", "1番人気")
print(hd)
for k in real:
    x = WS[k]
    print("  {:28s} {:+7.2f} {:>3d}/5 {:6.1%} {:6.1f} | {:7.1%} {:6.1f} | {:8.2f} {:6.0f} {:8.2f} | {:6d} {:7.1%} {:7.1%}".format(
        jp(k)[:28], x["dll"], x["yrs"], x["hit"], x["roi"], x["shit"], x["sroi"], x["up_ae"], x["up_roi"], x["dn_ae"], x["swn"], x["sw_new"], x["sw_fav"]))

# ---------------------------------------------------------------- ワイド: 組ごとのロジスティック回帰
pos = {}
k0 = 0
for g, rid in enumerate(ridx):
    for j in range(sizes[g]):
        pos[(g, int(H["u"][k0 + j]))] = k0 + j
    k0 += sizes[g]
P = defaultdict(list); psz = []
for g, rid in enumerate(ridx):
    x = info[rid]; tp, wf = x["tp"], x["wf"]; res = x["result"]; top3 = set((res.get("order") or [])[:3])
    pay = (res.get("payout") or {}).get("ワイド") or {}
    n = 0
    for key, v in wf.items():
        s = tp.get(key)
        if not s or not v or v[0] <= 1:
            continue
        a, b = (int(t) for t in key.split("-"))
        if (g, a) not in pos or (g, b) not in pos:
            continue
        P["ia"].append(pos[(g, a)]); P["ib"].append(pos[(g, b)]); P["pt"].append(min(PT * s, 0.98)); P["y"].append(int(a in top3 and b in top3))
        P["pay"].append(pay.get(key, 0) / 100.0); P["mid"].append((v[0] + v[1]) / 2); n += 1
    psz.append(n)
psz = np.array(psz); has = psz > 0
P = {k: np.array(v, dtype=float if k not in ("ia", "ib") else int) for k, v in P.items()}
prace = np.repeat(np.arange(NR), psz); pyear = ryear[prace]
x0 = lg(P["pt"]); NP_ = len(x0)
print("\n■ ワイド。組 {}点・{}R ({:.0f}分)".format(NP_, int(has.sum()), (time.time() - t0) / 60), flush=True)


def lr_fit(F, y, l2=0.0):
    b = np.zeros(F.shape[1]); R = np.eye(F.shape[1]) * l2; R[0, 0] = R[1, 1] = 0
    for _ in range(40):
        p = 1 / (1 + np.exp(-(F @ b)))
        step = np.linalg.solve((F * (p * (1 - p))[:, None]).T @ F + R + 1e-9 * np.eye(F.shape[1]), F.T @ (y - p) - R @ b)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def pair_eval(cols, l2=0.0):
    o = dict(ll=[], yll=[], n=0, hit=0, pay=0.0, sn=0, shit=0, spay=0.0, up_a=0.0, up_e=0.0, up_pay=0.0, up_n=0, dn_a=0.0, dn_e=0.0)
    zs = [Z[P["ia"], c] + Z[P["ib"], c] for c in cols]
    for yy in YS:
        te = pyear == yy; tr = ~te
        Ftr = np.column_stack([np.ones(tr.sum()), x0[tr]] + [z[tr] for z in zs]); Fte = np.column_stack([np.ones(te.sum()), x0[te]] + [z[te] for z in zs])
        b = lr_fit(Ftr, P["y"][tr], l2)
        p = 1 / (1 + np.exp(-(Fte @ b))); y = P["y"][te]; pay = P["pay"][te]
        sz = psz[(ryear == yy) & has]
        ll = y * np.log(p) + (1 - y) * np.log(1 - p)
        o["ll"].append(ll.sum()); o["yll"].append(ll.sum() / len(sz))
        top = first_top(p, sz)
        o["n"] += len(top); o["hit"] += y[top].sum(); o["pay"] += pay[top].sum()
        # 堅いワイドと同じ数だけ、自信のあるレースから
        btop = first_top(P["pt"][te], sz)
        k = int((P["pt"][te][btop] >= WIDE_P - 1e-9).sum())
        sel = top[np.argsort(-p[top], kind="stable")[:k]]
        o["sn"] += k; o["shit"] += y[sel].sum(); o["spay"] += pay[sel].sum()
        if cols:
            pb = 1 / (1 + np.exp(-(Fte[:, :2] @ PBASE[yy])))
            adj = Fte[:, 2:] @ b[2:]; m = P["mid"][te] <= 15
            hi = m & (adj >= np.quantile(adj[m], 0.9)); lo = m & (adj <= np.quantile(adj[m], 0.1))
            o["up_a"] += y[hi].sum(); o["up_e"] += pb[hi].sum(); o["up_pay"] += pay[hi].sum(); o["up_n"] += hi.sum()
            o["dn_a"] += y[lo].sum(); o["dn_e"] += pb[lo].sum()
    return o


PBASE = {}
for yy in YS:
    tr = pyear != yy
    PBASE[yy] = lr_fit(np.column_stack([np.ones(tr.sum()), x0[tr]]), P["y"][tr])
P0 = pair_eval([])
NRW = int(has.sum())
print("  土台 = 3連複から見た組の確率だけ: 全レースでいちばん来そうな組 的中 {:.1%}・回収率 {:.1f}% / 堅いワイドと同じ数({}R) 的中 {:.1%}・回収率 {:.1f}%".format(
    P0["hit"] / P0["n"], 100 * P0["pay"] / P0["n"], P0["sn"], P0["shit"] / P0["sn"], 100 * P0["spay"] / P0["sn"]), flush=True)
PR = {}
for c in range(NF):
    PR[names[c]] = pair_eval([c])
print("  1つずつ足した ({:.0f}分)".format((time.time() - t0) / 60), flush=True)


def psum(o):
    d = [(a - b) * 1000 for a, b in zip(o["yll"], P0["yll"])]
    return dict(dll=(sum(o["ll"]) - sum(P0["ll"])) / NRW * 1000, yrs=sum(x > 0 for x in d), hit=o["hit"] / o["n"], roi=100 * o["pay"] / o["n"],
                shit=o["shit"] / o["sn"], sroi=100 * o["spay"] / o["sn"], up_ae=o["up_a"] / o["up_e"] if o["up_e"] else float("nan"),
                up_roi=100 * o["up_pay"] / o["up_n"] if o["up_n"] else float("nan"), dn_ae=o["dn_a"] / o["dn_e"] if o["dn_e"] else float("nan"))


PS = {k: psum(v) for k, v in PR.items()}
rnd = [PS[k] for k in PS if k.startswith("rand")]
realp = sorted((k for k in PS if not k.startswith("rand")), key=lambda k: -PS[k]["dll"])
print("  乱数 {}個: 当たり方の伸び {:+.2f}〜{:+.2f} / 全レースの的中 {:.1%}〜{:.1%} / 同じ数の的中 {:.1%}〜{:.1%}・回収率 {:.1f}〜{:.1f}% / 押し上げる1割の回収率 {:.0f}〜{:.0f}%".format(
    NRAND, min(x["dll"] for x in rnd), max(x["dll"] for x in rnd), min(x["hit"] for x in rnd), max(x["hit"] for x in rnd),
    min(x["shit"] for x in rnd), max(x["shit"] for x in rnd), min(x["sroi"] for x in rnd), max(x["sroi"] for x in rnd),
    min(x["up_roi"] for x in rnd), max(x["up_roi"] for x in rnd)))
print("  {:28s} {:>7s} {:>4s} {:>7s} {:>6s} | {:>7s} {:>6s} | {:>8s} {:>6s} {:>8s}".format("評価値", "伸び", "年", "的中", "回収", "同数的中", "回収", "上1割A/E", "回収", "下1割A/E"))
for k in realp:
    x = PS[k]
    print("  {:28s} {:+7.2f} {:>3d}/5 {:6.1%} {:6.1f} | {:7.1%} {:6.1f} | {:8.2f} {:6.0f} {:8.2f}".format(
        jp(k)[:28], x["dll"], x["yrs"], x["hit"], x["roi"], x["shit"], x["sroi"], x["up_ae"], x["up_roi"], x["dn_ae"]))

# ---------------------------------------------------------------- 全部まとめて足す(効きすぎないよう縮める)
allc = [c for c in range(NF) if not names[c].startswith("rand")]
print("\n■ 評価値を全部まとめて足す({}個。縮める強さを変えて)".format(len(allc)))
ALL = {}
for l2 in (10.0, 100.0, 1000.0):
    w = wsum(win_eval(allc, l2)); p = psum(pair_eval(allc, l2)); ALL[str(l2)] = {"win": w, "pair": p}
    print("  縮め {:6.0f}: 1着 伸び {:+.2f}({}/5年) 的中 {:.1%} 回収 {:.1f}% 同数 {:.1%}・{:.1f}% 上1割 A/E {:.2f}・回収 {:.0f}% | ワイド 伸び {:+.2f}({}/5年) 的中 {:.1%} 回収 {:.1f}% 同数 {:.1%}・{:.1f}% 上1割 A/E {:.2f}・回収 {:.0f}%".format(
        l2, w["dll"], w["yrs"], w["hit"], w["roi"], w["shit"], w["sroi"], w["up_ae"], w["up_roi"],
        p["dll"], p["yrs"], p["hit"], p["roi"], p["shit"], p["sroi"], p["up_ae"], p["up_roi"]), flush=True)
json.dump({"win0": {"hit": W0["hit"] / W0["n"], "roi": 100 * W0["pay"] / W0["n"], "shit": W0["shit"] / W0["sn"], "sroi": 100 * W0["spay"] / W0["sn"], "sn": int(W0["sn"])},
           "pair0": {"hit": P0["hit"] / P0["n"], "roi": 100 * P0["pay"] / P0["n"], "shit": P0["shit"] / P0["sn"], "sroi": 100 * P0["spay"] / P0["sn"], "sn": int(P0["sn"])},
           "win": WS, "pair": PS, "all": ALL}, open(os.path.join(OUT, "odds_plus.json"), "w"), ensure_ascii=False, indent=1, default=float)
print("かかった時間 {:.0f}分".format((time.time() - t0) / 60))

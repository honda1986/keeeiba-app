"""後押しの式(keiba/push_model.json)を作る。毎週の学び直しで、tools/train_model.py のあとに回す(中央だけ)。
  KEIBA_CACHE=/tmp/keiba_dataset.pkl python tools/train_push.py
  train_model.py が作った材料(KEIBA_CACHE)と、結果・確定オッズ(keiba/days)を使う。2022年〜の中央。

  式(keiba_push.py に説明):
   1着 … 条件つきロジット。土台 = 単勝オッズ(曲がりも)+ 3連複・ワイドから見た馬ごとの強さ。これに評価値(レースの中でそろえた値)を足す
   2頭とも3着内 … 組ごとのロジスティック回帰。土台 = 3連複から見た組の確率(曲がりも)+ ワイドのオッズ + 2頭の単勝オッズ。
                  これに 2頭の評価値の合計を足す
   評価値の重みは縮める(L2)。後押し = 評価値の分だけ(オッズの分は入れない)
  保存するもの: 全部の年で作った重み / 境目(単勝20倍以下の馬の後押しの上5%など・モデルの買い目を3つに分ける)/
               成績(その年を除いて作った式で、その年を確かめたもの。アプリに出す)/
               後押しの買い目(馬連: 後押しが上5%の馬 × 1番人気。keiba_logic.oshi_picks)の日ごとの成績(days)"""
import gc, json, os, pickle, sys, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import train_model as T
import keiba_logic as L
import keiba_push as PU

L2 = 100.0
PT = L.PRE["pt"]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def build(rows, races, log=print):
    """材料と結果から、式を作るための表にする"""
    info = {rc["id"]: rc for d, rc in races}
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if r["f"].get("center") == 0 and r["rid"] in info:
            by[r["rid"]].append(i)
    names = PU.candidates([rows[i]["f"] for ix in by.values() for i in ix])
    D = dict(rid=[], year=[], date=[], sizes=[], names=names)
    H = defaultdict(list); Z = []; P = defaultdict(list); psz = []
    k0 = 0
    for rid in sorted(by, key=lambda x: (rows[by[x][0]]["date"], x)):
        rc = info[rid]; res = rc["result"]; order = res.get("order") or []
        od = {int(u): o for u, o in (res.get("odds") or {}).items() if o and o > 1}
        ix = [i for i in by[rid] if rows[i]["u"] in od]
        if len(order) < 3 or len(ix) < 5 or order[0] not in {rows[i]["u"] for i in ix}:
            continue
        us = [rows[i]["u"] for i in ix]
        q = np.array([1 / od[u] for u in us]); q = q / q.sum()
        tp = rc.get("trio_pair_final") or {}; wf = rc.get("wide_final") or {}
        ts, ws = defaultdict(float), defaultdict(float)
        for key, v in tp.items():
            for t in key.split("-"):
                ts[int(t)] += v
        for key, v in wf.items():
            if v and v[0] > 1:
                for t in key.split("-"):
                    ws[int(t)] += 2 / (v[0] + v[1])
        fs = [rows[i]["f"] for i in ix]
        Z.append(np.array([PU.column(fs, nm) for nm in names]).T)
        zq = PU.zs([float(np.log(v)) for v in q])
        zt = PU.zs([float(np.log(ts[u])) if ts.get(u) else None for u in us]); zw = PU.zs([float(np.log(ws[u])) if ws.get(u) else None for u in us])
        pos = {}
        for j, u in enumerate(us):
            pos[u] = k0 + j
            H["u"].append(u); H["logq"].append(np.log(q[j])); H["zq"].append(zq[j]); H["zt"].append(zt[j]); H["zw"].append(zw[j])
            H["odds"].append(od[u]); H["win"].append(int(u == order[0]))
        top3 = set(order[:3]); pay = (res.get("payout") or {}).get("ワイド") or {}
        n = 0
        for key, v in wf.items():
            s = tp.get(key)
            if not s or not v or v[0] <= 1:
                continue
            a, b = (int(t) for t in key.split("-"))
            if a not in pos or b not in pos:
                continue
            P["ia"].append(pos[a]); P["ib"].append(pos[b]); P["pt"].append(min(PT * s, 0.98)); P["y"].append(int(a in top3 and b in top3))
            P["pay"].append(pay.get(key, 0) / 100.0); P["mid"].append((v[0] + v[1]) / 2); n += 1
        psz.append(n)
        D["rid"].append(rid); D["year"].append(rows[ix[0]]["date"][:4]); D["date"].append(rows[ix[0]]["date"]); D["sizes"].append(len(ix))
        k0 += len(ix)
    D["sizes"] = np.array(D["sizes"]); D["year"] = np.array(D["year"]); D["psz"] = np.array(psz)
    D["Z"] = np.vstack(Z); del Z
    keep = np.abs(D["Z"]).sum(axis=0) > 0
    D["Z"] = D["Z"][:, keep]; D["names"] = [nm for nm, k in zip(names, keep) if k]
    D["H"] = {k: np.array(v, dtype=float) for k, v in H.items()}
    D["P"] = {k: np.array(v, dtype=int if k in ("ia", "ib") else float) for k, v in P.items()}
    h = D["H"]
    D["BW"] = np.column_stack([h["logq"], h["logq"] ** 2, h["zq"], h["zt"], h["zw"]])
    p = D["P"]; x0 = lg(p["pt"]); xw = lg(np.minimum(0.775 / p["mid"], 0.98))
    D["BP"] = np.column_stack([np.ones(len(x0)), x0, x0 ** 2, xw, xw ** 2, h["zq"][p["ia"]] + h["zq"][p["ib"]], h["logq"][p["ia"]] + h["logq"][p["ib"]]])
    D["hyear"] = np.repeat(D["year"], D["sizes"]); D["pyear"] = np.repeat(D["year"], D["psz"])
    D["info"] = info
    log("中央 {}R・{}頭・組 {}点・評価値 {}個".format(len(D["rid"]), len(h["u"]), len(x0), len(D["names"])))
    return D


def fit_win(B, Z, y, sizes, l2=L2):
    """条件つきロジット(レースの中で1頭が勝つ)。土台 B は縮めず、評価値 Z の重みだけ縮める。-> 評価値の重み"""
    F = np.column_stack([B, Z]); kb = B.shape[1]; k = F.shape[1]
    st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    R = np.eye(k) * l2; R[:kb, :kb] = 0
    b = np.zeros(k); b[0] = 1.0
    for _ in range(50):
        s = F @ b; s -= np.repeat(np.maximum.reduceat(s, st), sizes)
        e = np.exp(s); p = e / np.repeat(np.add.reduceat(e, st), sizes)
        Fp = F * p[:, None]; M = np.add.reduceat(Fp, st, axis=0)
        step = np.linalg.solve(Fp.T @ F - M.T @ M + R + 1e-8 * np.eye(k), F.T @ (y - p) - R @ b)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b[kb:]


def fit_pair(B, ZP, y, l2=L2):
    """ロジスティック回帰(2頭とも3着内)。土台 B は縮めず、2頭の評価値の合計 ZP の重みだけ縮める。-> 評価値の重み"""
    F = np.column_stack([B, ZP]); kb = B.shape[1]; k = F.shape[1]
    R = np.eye(k) * l2; R[:kb, :kb] = 0
    b = np.zeros(k)
    for _ in range(40):
        p = 1 / (1 + np.exp(-(F @ b)))
        step = np.linalg.solve((F * (p * (1 - p))[:, None]).T @ F + R + 1e-8 * np.eye(k), F.T @ (y - p) - R @ b)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b[kb:]


def loyo(D, log=print):
    """年ごとに、その年を除いて作った式の後押し(馬ごと: 1着の式 w、2頭とも3着内の式 p)"""
    h, p = D["H"], D["P"]; Z = D["Z"]
    ZP = Z[p["ia"]] + Z[p["ib"]]
    w = np.zeros(len(Z)); q = np.zeros(len(Z))
    for yy in sorted(set(D["year"])):
        te = D["hyear"] == yy; tp_ = D["pyear"] == yy
        if (~te).sum() == 0:
            continue
        bw = fit_win(D["BW"][~te], Z[~te], h["win"][~te], D["sizes"][D["year"] != yy])
        bp = fit_pair(D["BP"][~tp_], ZP[~tp_], p["y"][~tp_])
        w[te] = Z[te] @ bw; q[te] = Z[te] @ bp
        log("  {}年を除いて作った".format(yy))
    bw = fit_win(D["BW"], Z, h["win"], D["sizes"]); bp = fit_pair(D["BP"], ZP, p["y"])
    del ZP; gc.collect()
    return w, q, bw, bp


def tab(lv, hit, pay, years):
    """強さ(0 弱い・1 ふつう・2 強い)ごとの 点数・的中・回収率(年ごとも)"""
    ys = sorted(set(years)); out = []
    for k in (0, 1, 2):
        m = lv == k
        out.append({"n": int(m.sum()), "hit": round(float(hit[m].mean()), 4) if m.any() else None,
                    "roi": round(float(100 * pay[m].mean()), 1) if m.any() else None,
                    "yr": [round(float(100 * pay[m & (years == y)].mean()), 0) if (m & (years == y)).any() else None for y in ys]})
    return out


def stats(D, w, q, log=print):
    """境目と成績(その年を除いて作った式の後押しで)"""
    h, p = D["H"], D["P"]; sizes = D["sizes"]; st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    race_of = np.repeat(np.arange(len(sizes)), sizes); hy = D["hyear"]
    cuts, out = {}, {"years": sorted(set(D["year"]))}
    # 単勝(20倍以下の馬を、後押しの強さで)
    m = h["odds"] <= 20
    cuts["win_top"] = {str(k): round(float(np.quantile(w[m], k)), 4) for k in (0.2, 0.8, 0.9, 0.95)}
    ys = out["years"]; wt = {}
    for nm, sel in (("top5", m & (w >= cuts["win_top"]["0.95"])), ("top10", m & (w >= cuts["win_top"]["0.9"])), ("top20", m & (w >= cuts["win_top"]["0.8"])),
                    ("bottom20", m & (w <= cuts["win_top"]["0.2"])), ("all", m)):
        wt[nm] = {"n": int(sel.sum()), "hit": round(float(h["win"][sel].mean()), 4), "roi": round(float(100 * (h["win"][sel] * h["odds"][sel]).mean()), 1),
                  "yr": [round(float(100 * (h["win"][sel & (hy == y)] * h["odds"][sel & (hy == y)]).mean()), 0) for y in ys]}
    out["win_top"] = wt
    # 後押しの買い目: 後押しが上5%(L.OSHI["q"])の馬 x(1番人気でない・単勝 max_odds 倍以下)× 1番人気 の馬連。日ごとの成績も残す
    cut = cuts["win_top"][L.OSHI["q"]]
    days = {}; oy = {y: [0, 0, 0.0] for y in ys}
    for g, (s0, n) in enumerate(zip(st, sizes)):
        us = h["u"][s0:s0 + n].astype(int); od = h["odds"][s0:s0 + n]
        push = {int(u): float(v) for u, v in zip(us, w[s0:s0 + n])}
        pay = (D["info"][D["rid"][g]]["result"].get("payout") or {}).get("馬連") or {}
        for pk in L.oshi_picks(push, {int(u): float(o) for u, o in zip(us, od)}, cut):
            v = pay.get("-".join(str(x) for x in sorted((pk["u"], pk["fav"]))), 0)
            d = days.setdefault(D["date"][g], [D["date"][g], 0, 0, 0])
            d[1] += 1; d[2] += int(v > 0); d[3] += int(v)
            t = oy[D["year"][g]]; t[0] += 1; t[1] += v > 0; t[2] += v
    tn, th, tp_ = (sum(t[k] for t in oy.values()) for k in (0, 1, 2))
    out["oshi"] = {"n": int(tn), "hit": round(th / max(tn, 1), 4), "roi": round(tp_ / max(tn, 1), 1),
                   "yn": [int(oy[y][0]) for y in ys], "yh": [int(oy[y][1]) for y in ys], "yr": [round(oy[y][2] / max(oy[y][0], 1), 1) for y in ys]}
    out["_days"] = [days[k] for k in sorted(days)]
    # モデルの買い目(記録に残っている予想。3連複◎-○-▲ は3頭、ワイド◎-○ は2頭の push2 の平均)。
    # アプリの収支と同じく HISTORY_FROM から(それより前の日の予想は、その日より前だけで学んだモデルのものではない)
    pos = {}
    k0 = 0
    for g, rid in enumerate(D["rid"]):
        for j in range(sizes[g]):
            pos[(rid, int(h["u"][k0 + j]))] = k0 + j
        k0 += sizes[g]
    tv, th, tpay, ty = [], [], [], []
    for g, rid in enumerate(D["rid"]):
        rc = D["info"][rid]; pr = rc.get("pred") or {}; stl = rc.get("settle")
        if D["date"][g] < T.K.HISTORY_FROM or pr.get("verdict") not in ("A", "B") or not stl or not pr.get("bets") or not stl.get("stake"):
            continue
        us = sorted({u for bt in pr["bets"] for c in bt.get("combos") or [] for u in c})
        if not us or any((rid, u) not in pos for u in us):
            continue
        tv.append(sum(q[pos[(rid, u)]] for u in us) / len(us)); th.append(int(stl.get("ret", 0) > 0)); tpay.append(stl.get("ret", 0) / stl["stake"]); ty.append(D["year"][g])
    if len(tv) >= 60:
        tv, th, tpay, ty = np.array(tv), np.array(th), np.array(tpay), np.array(ty)
        cuts["model"] = [round(float(v), 4) for v in np.quantile(tv, [1 / 3, 2 / 3])]
        out["model"] = tab(np.digitize(tv, cuts["model"]), th, tpay, ty)
        out["model_years"] = sorted(set(ty))
    for k in ("model",):
        if k in out:
            log("  {}: {}".format(k, " / ".join("{}点 的中{:.1%} 回収率{:.1f}% {}".format(x["n"], x["hit"] or 0, x["roi"] or 0, x["yr"]) for x in out[k])))
    log("  後押しの買い目(馬連): {}点 的中{:.1%} 回収率{:.1f}% 年ごとの点数 {} 回収率 {}".format(out["oshi"]["n"], out["oshi"]["hit"], out["oshi"]["roi"], out["oshi"]["yn"], out["oshi"]["yr"]))
    log("  単勝(20倍以下): " + " / ".join("{} {}頭 的中{:.1%} 回収率{:.1f}% {}".format(k, v["n"], v["hit"], v["roi"], v["yr"]) for k, v in wt.items()))
    return cuts, out


def main():
    t0 = time.time()
    races = T.load_races()
    norm, rows, jt_all, sp = pickle.load(open(T.CACHE, "rb"))
    D = build(rows, races)
    del rows, races; gc.collect()
    w, q, bw, bp = loyo(D)
    cuts, st = stats(D, w, q)
    days = st.pop("_days")
    out = {"updated": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"), "races": len(D["rid"]), "l2": L2,
           "features": D["names"], "win": [round(float(v), 5) for v in bw], "pair": [round(float(v), 5) for v in bp], "cuts": cuts, "stats": st, "days": days}
    json.dump(out, open(PU.FILE, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print("保存:", PU.FILE, "({:.0f}分)".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()

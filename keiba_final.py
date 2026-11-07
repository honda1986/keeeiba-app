"""高知の最終レース(ファイナルレース)用の確率。荒れるレースなので、今のモデルより単勝オッズの方が当たる
(2024〜2026年の最終レース: AUC 今のモデル 0.645 / 単勝オッズ 0.695 / これ 0.699)。
単勝オッズから出した3着内の確率(Harville)に、今のモデルの確率と前走などを足したロジスティック回帰。
式は keiba/kochi/final_model.json(tools/kochi_final_fit.py が作る)。印(◎○▲△)はこの確率の順に付け直す。
オッズがまだ無いとき(朝・昼)は今のモデルのまま"""
import json
import math
import os

import numpy as np

FEATS = ["lp3", "lmt3", "last_rank", "last_nk", "last_margin", "best3", "pos", "cls_diff", "rest", "nruns", "wdiff", "dist_diff", "waku"]


def pl_top3(w):
    """Plackett-Luce(Harville)で各馬が3着内に入る確率"""
    w = np.asarray(w, float)
    S = w.sum(); p1 = w / S; A = p1[:, None] * w[None, :] / (S - w)[:, None]; np.fill_diagonal(A, 0)
    B = S - w[:, None] - w[None, :]; C = np.where(B > 1e-12, A / np.maximum(B, 1e-12), 0); np.fill_diagonal(C, 0)
    return p1 + A.sum(0) + w * (C.sum() - C.sum(1) - C.sum(0))


def lg(v):
    v = min(max(v, 1e-4), 1 - 1e-4)
    return math.log(v / (1 - v))


def features(race, pred_horses, odds):
    """馬番 -> 特徴(dict)。odds は 馬番 -> 単勝オッズ。オッズが5頭に無ければ {}"""
    ph = {h["umaban"]: h for h in pred_horses if h.get("p3") is not None}
    us = sorted(u for u in ph if (odds.get(u) or 0) > 1)
    if len(us) < 5:
        return {}
    inv = sum(1 / odds[u] for u in us)
    mt3 = pl_top3([1 / odds[u] / inv for u in us])
    card = {h["umaban"]: h for h in race.get("horses") or []}

    def rel(x, n):
        return (x - 1) / max(n - 1, 1) if x and n else 0.5
    out = {}
    for i, u in enumerate(us):
        h, c = ph[u], card.get(u, {})
        ps = c.get("past") or []
        l0 = ps[0] if ps else {}
        out[u] = dict(lp3=lg(h["p3"]), lmt3=lg(mt3[i]),
                      last_rank=rel(l0.get("rank"), l0.get("n")), last_nk=rel(l0.get("ninki"), l0.get("n")),
                      last_margin=min(max(l0.get("margin") or 0, 0), 5) if l0 else 2.0,
                      cls_diff=(l0.get("cls") or 0) - (race.get("cls") or 0) if l0 else 0,
                      best3=min([rel(x.get("rank"), x.get("n")) for x in ps[:3]] or [0.5]),
                      pos=rel((l0.get("pass") or [0])[0], l0.get("n")) if l0.get("pass") else 0.5,
                      rest=min((h.get("rest_days") or 60), 365) / 30, nruns=min(h.get("nruns") or 0, 10),
                      wdiff=abs(c.get("wdiff") or 0),
                      dist_diff=abs((race.get("dist") or 0) - (l0.get("dist") or 0)) / 200 if l0 else 0,
                      waku=rel(u, len(us)))
    return out


def chaos_x(odds):
    """荒れ度の材料: [log(1番人気のオッズ), オッズのばらけ方(0〜1。1に近いほど混戦)]。5頭に無ければ None"""
    od = [v for v in odds.values() if v and v > 1]
    if len(od) < 5:
        return None
    q = np.array([1 / v for v in od]); q = q / q.sum()
    return [math.log(min(od)), float(-(q * np.log(q)).sum() / math.log(len(q)))]


def chaos_level(p):
    """荒れ度の段階(高 55%以上・中 40%以上・低)。アプリ(index.html の chaosLevel)と同じ"""
    return "高" if p >= .55 else "中" if p >= .4 else "低"


def fit_logit(X, y, l2=1.0):
    """L2 付きのロジスティック回帰(ニュートン法)。[切片, 係数...]"""
    Z = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Z.shape[1]); R = l2 * np.eye(len(w)); R[0, 0] = 0
    for _ in range(60):
        p = 1 / (1 + np.exp(-Z @ w)); H = (Z * (p * (1 - p))[:, None]).T @ Z + R + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, Z.T @ (y - p) - R @ w)
    return w


def chaos(model, odds):
    """1番人気が3着に来ない確率(荒れ度)。式は model["chaos"](高知の最終レースは final_model.json、中央は chaos_model.json)"""
    c, x = (model or {}).get("chaos"), chaos_x(odds)
    if not c or not x:
        return None
    z = c["w"][0] + sum(w * (v - m) / s for w, v, m, s in zip(c["w"][1:], x, c["mu"], c["sd"]))
    return 1 / (1 + math.exp(-z))


def probs(model, feats):
    """馬番 -> 3着内の確率(レースで合計3にそろえる)"""
    raw = {}
    for u, f in feats.items():
        z = model["w"][0] + sum(w * (f[k] - m) / s for k, w, m, s in zip(model["feats"], model["w"][1:], model["mu"], model["sd"]))
        raw[u] = 1 / (1 + math.exp(-z))
    tot = sum(raw.values())
    return {u: min(0.97, v * 3 / tot) for u, v in raw.items()}


def load(out_dir):
    try:
        return json.load(open(os.path.join(out_dir, "final_model.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None


def apply(race, pred, model, odds):
    """最終レースの予想を、最終レース用の確率と印に置き換える(元の確率は p3_model に残す)。オッズが無ければ何もしない"""
    if not model or not pred.get("horses"):
        return False
    fe = features(race, pred["horses"], odds)
    if not fe:
        return False
    pf = probs(model, fe)
    for h in pred["horses"]:
        if h["umaban"] in pf:
            h["p3_model"] = h.get("p3_model", h["p3"])
            h["p3"] = round(pf[h["umaban"]], 3)
        if h.get("mark") in ("◎", "○", "▲", "△"):
            h["mark"] = ""
    hs = sorted((h for h in pred["horses"] if h["umaban"] in pf), key=lambda h: -h["p3"])
    for h, m in zip(hs[:4], "◎○▲△"):
        h["mark"] = m
    pred["horses"].sort(key=lambda h: -(h.get("p3") or 0))
    if len(hs) >= 4:
        pred["honmei"], pred["taikou"] = hs[0]["umaban"], hs[1]["umaban"]
        pred["aite"] = [hs[2]["umaban"], hs[3]["umaban"]]
    pred["final_model"] = True
    pred.setdefault("notes", []).append("最終レース用の確率(単勝オッズと前走から。荒れるレースなので今のモデルよりオッズの方が当たる)")
    pc = chaos(model, odds)
    if pc is not None:
        fav = min((u for u in odds if odds[u] and odds[u] > 1), key=lambda u: odds[u])
        lv = chaos_level(pc)
        pred["chaos"] = {"p": round(pc, 3), "fav": fav, "level": lv}
        pred["reason"] = "荒れ度 {}: 1番人気({}番)が3着に来ない確率 {:.0%}(予想だけ。買わない)".format(lv, fav, pc)
    return True

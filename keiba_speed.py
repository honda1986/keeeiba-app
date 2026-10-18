#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keiba_speed.py -- 走破タイム・上がり3Fを馬場差で補正した指数

■ 考え方
  1km あたりの走破タイム(秒)を、次の足し算で表す。
      タイム = コースの基準(競馬場・芝ダ・距離) + クラスの差 + その日のその馬場の速さ(馬場差) + 馬の力
  馬柱に載っている過去走を全部集め、基準・クラスの差・馬場差を交互に求める(中央値に近い、外れに強い平均)。
  指数 = (基準 + 馬場差 − タイム) × 10。0 がそのコースの普通、+10 で 1km あたり 1 秒速い。
  クラスの差は指数から引かない(強いクラスで速く走れるのは力のうち)。
  上がり3F も同じように、コースの基準 + クラスの差 + その日の上がりの出やすさ で補正する。

■ 先の結果を使わないか
  馬場差はその日に走ったレースのタイムだけで決まる(レース後すぐ分かる)。
  基準とクラスの差は全期間から求めるが、コースの速さはほとんど変わらないので影響は小さい。

■ ファイル
  keiba_speed.json … 基準・クラスの差・馬場差(tools/train_model.py が作る。当日の予想では、
                     まだ表に無い日の馬場差を、その日の出馬表の馬柱から足す)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPEED_FILE = os.path.join(HERE, "keiba_speed.json")

SHRINK_DAY = 4         # 馬場差は、走った数が少ない日は 0 に寄せる
SHRINK_COURSE = 10     # 走った数が少ないコースは、芝ダ・距離帯の基準に寄せる
MAX_MARGIN = 2.5       # 大きく負けた走り(バテた・止めた)は基準づくりに使わない


def km(r):
    return (r.get("dist") or 0) / 1000.0


def course_key(r):
    return "{}|{}|{}".format(r.get("venue") or "?", r.get("surface") or "?", r.get("dist") or 0)


def band_key(r):
    d = r.get("dist") or 0
    b = 0 if d <= 1300 else 1 if d <= 1600 else 2 if d <= 2000 else 3
    return "{}|{}".format(r.get("surface") or "?", b)


def day_key(r):
    return "{}|{}|{}".format((r.get("date") or "").replace("-", ""), r.get("venue") or "?", r.get("surface") or "?")


def cls_key(r, center):
    return "{}|{}".format(center or "?", r.get("cls") if r.get("cls") is not None else "?")


def usable(r):
    return (r.get("surface") in ("芝", "ダ") and r.get("time") and (r.get("dist") or 0) >= 800
            and r.get("date") and r.get("venue"))


def collect(races, center_of):
    """[(日付, レース)] の馬柱から、同じ走りを1回ずつにして集める。center_of(競馬場) -> "jra"/"nar"/None"""
    seen = {}
    for _, rc in races:
        for h in rc.get("horses") or []:
            for r in h.get("past") or []:
                if not usable(r):
                    continue
                k = (r["date"], r["venue"], r.get("race_name"), r.get("dist"), r.get("umaban"))
                if k in seen:
                    continue
                x = {f: r.get(f) for f in ("date", "venue", "surface", "dist", "time", "last3f", "cls",
                                           "margin", "rank", "n")}
                x["center"] = center_of(r["venue"])
                seen[k] = x
    return list(seen.values())


def _mean(vals, shrink=0.0, prior=0.0):
    return (sum(vals) + shrink * prior) / (len(vals) + shrink) if vals else prior


def _clip(v, lim):
    return max(-lim, min(lim, v))


def fit(recs, rounds=4):
    """基準・クラスの差・馬場差を求める。戻り値はそのまま JSON にできる表"""
    rows = []
    for r in recs:
        if r.get("margin") is not None and r["margin"] > MAX_MARGIN:
            continue
        rows.append({"c": course_key(r), "b": band_key(r), "d": day_key(r), "k": cls_key(r, r.get("center")),
                     "t": r["time"] / km(r), "l": r.get("last3f")})
    out = {}
    for field in ("t", "l"):
        rs = [x for x in rows if x[field]]
        S, B, C, V = {}, {}, {}, {}

        def base(x):
            return S.get(x["c"], B.get(x["b"], 0.0))
        for _ in range(rounds):
            # コースの基準(走った数が少ないコースは距離帯の基準へ寄せる)
            accb, accc = {}, {}
            for x in rs:
                v = x[field] - C.get(x["k"], 0.0) - V.get(x["d"], 0.0)
                accb.setdefault(x["b"], []).append(v)
                accc.setdefault(x["c"], []).append(v)
            B = {k: _mean(_trim(v)) for k, v in accb.items()}
            S = {k: _mean(_trim(v), SHRINK_COURSE, B.get(band_of(k), 0.0)) for k, v in accc.items()}
            # クラスの差
            acck = {}
            for x in rs:
                acck.setdefault(x["k"], []).append(_clip(x[field] - base(x) - V.get(x["d"], 0.0), 3.0))
            C = {k: _mean(v, 20, 0.0) for k, v in acck.items()}
            # 馬場差(その日・その競馬場・芝ダ)
            accd = {}
            for x in rs:
                accd.setdefault(x["d"], []).append(_clip(x[field] - base(x) - C.get(x["k"], 0.0), 3.0))
            V = {k: _mean(v, SHRINK_DAY, 0.0) for k, v in accd.items()}
        res = [x[field] - base(x) - C.get(x["k"], 0.0) - V.get(x["d"], 0.0) for x in rs]
        sd = (sum(v * v for v in res) / len(res)) ** 0.5 if res else 0
        out[field] = {"S": _round(S), "B": _round(B), "C": _round(C), "V": _round(V),
                      "n_day": {k: len(v) for k, v in accd.items()} if rs else {}, "sd": round(sd, 3), "n": len(rs)}
    return out


def band_of(course):
    """コースのキー -> 距離帯のキー"""
    venue, surf, dist = course.split("|")
    return band_key({"surface": surf, "dist": int(dist)})


def _trim(vals, frac=0.1):
    """外れを両端から1割ずつ落とす"""
    if len(vals) < 10:
        return vals
    v = sorted(vals)
    k = int(len(v) * frac)
    return v[k:len(v) - k]


def _round(d):
    return {k: round(v, 4) for k, v in d.items()}


def figure(r, table, field="t"):
    """1走の指数(+ほど速い)。表に無いコース・馬場差は距離帯の基準・0 で代える"""
    if not table or not usable(r):
        return None
    tb = table.get(field)
    if not tb:
        return None
    if field == "t":
        v = r["time"] / km(r)
    else:
        if not r.get("last3f"):
            return None
        v = r["last3f"]
    s = tb["S"].get(course_key(r))
    if s is None:
        s = tb["B"].get(band_key(r))
        if s is None:
            return None
    return round((s + tb["V"].get(day_key(r), 0.0) - v) * 10, 2)


def extend(table, recs, center_of=None):
    """表に無い日の馬場差を、集めた走り(当日の出馬表の馬柱など)から足す。基準・クラスの差はそのまま"""
    if not table:
        return table
    for field in ("t", "l"):
        tb = table.get(field)
        if not tb:
            continue
        accd = {}
        for r in recs:
            if not usable(r):
                continue
            dk = day_key(r)
            if dk in tb["V"]:
                continue
            if r.get("margin") is not None and r["margin"] > MAX_MARGIN:
                continue
            v = r["time"] / km(r) if field == "t" else r.get("last3f")
            if not v:
                continue
            s = tb["S"].get(course_key(r), tb["B"].get(band_key(r)))
            if s is None:
                continue
            c = (center_of(r["venue"]) if center_of else r.get("center"))
            accd.setdefault(dk, []).append(_clip(v - s - tb["C"].get(cls_key(r, c), 0.0), 3.0))
        for k, v in accd.items():
            tb["V"][k] = round(_mean(v, SHRINK_DAY, 0.0), 4)
    return table


_TABLE = {"t": None, "tried": False}


def load():
    if not _TABLE["tried"]:
        _TABLE["tried"] = True
        try:
            _TABLE["t"] = json.load(open(SPEED_FILE, encoding="utf-8"))
        except Exception:
            _TABLE["t"] = None
    return _TABLE["t"]


def save(table, path=SPEED_FILE):
    slim = {f: {k: v for k, v in tb.items() if k != "n_day"} for f, tb in table.items()}
    json.dump(slim, open(path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

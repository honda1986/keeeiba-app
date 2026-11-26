"""後押し: オッズに足す評価値の式(中央。2026-11-14)。

オッズ(単勝・ワイド・3連複)だけの式に、モデルの材料(評価値)を足すと、的中率は上がらないが当たり方は少し良くなる
(tools/experiments/odds_plus.py。過去の成績が良く見える馬は買われすぎ、見劣りする馬は売れなさすぎ、という向き)。
その「足した分」だけを取り出した数字が後押し:
  後押し = Σ 重み × (評価値をレースの中で平均0・ばらつき1にそろえた値)
オッズは式を作るときの土台にだけ使い、後押しの計算には使わない(朝の予想の時点で出せる)。
  push  … 1着の式の後押し(単勝に使う)
  push2 … 2頭とも3着内の式の後押し(馬ごと。ワイドは2頭の合計、3連複は3頭の合計で見る)
式(重み・境目・過去の成績)は keiba/push_model.json。毎週の学び直しで tools/train_push.py が作り直す。
ここは keiba.py の予想(numpy 無しでも動く)と tools/train_push.py の両方から使う。そろえ方を1か所にするため"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "keiba", "push_model.json")
# 過去のレース(予想に push が入る前の日)の後押しの買い目と、モデルの買い目の後押しの強さ。アプリが過去の日を開いたときに出す
PAST_FILE = os.path.join(HERE, "keiba", "oshi_past.json")
SUF = ("_dhi", "_dlo", "_dmean", "_rank")       # レースの中でそろえ直した値(元の値だけ使う)
CATS = ("style", "sex")                          # 種類ごとの 0/1 にするもの
# レースの中で同じ値のもの(場・馬場など)と、コース傾向の2つ(表が 2021〜2025年の集計で、その年までの成績には先の結果が混ざる)は使わない
SKIP = ("venue", "surface", "center", "course_kata", "course_waku_idx", "course_style_idx",
        # 自前のコース傾向(2026-11-18 から確率モデルの材料)は後押しには入れない: 入れると後押しの買い目が 5年で 102% → 98.5% に下がる
        # (差はぶれの中だが、後押しの買い目は入れる前の式で確かめて決めたので、そのままにする)
        "cown_style", "cown_waku", "cown_gate", "cown_n")
CLIP = 3.0


def num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def zs(vals):
    """レースの中で平均0・ばらつき1にそろえる。無い値(None)は0 = 平均。±3で切る"""
    xs = [v for v in vals if v is not None]
    if not xs:
        return [0.0] * len(vals)
    mean = sum(xs) / len(xs)
    sd = (sum((v - mean) ** 2 for v in xs) / len(xs)) ** 0.5
    if sd <= 1e-9:
        return [0.0] * len(vals)
    return [0.0 if v is None else max(-CLIP, min(CLIP, (v - mean) / sd)) for v in vals]


def column(fs, name):
    """1レース分の評価値(fs = 馬ごとの材料の dict)から、name の列をそろえて返す。"style=2" の形は 0/1 にしてから"""
    if "=" in name:
        c, v = name.split("=")
        v = float(v)
        return zs([1.0 if num(f.get(c)) == v else 0.0 for f in fs])
    return zs([num(f.get(name)) for f in fs])


def candidates(all_fs):
    """式に入れる候補の名前。all_fs = 全部の馬の材料(名前と、種類の値を集めるのに使う)"""
    names = sorted({k for f in all_fs for k in f})
    out = [k for k in names if not k.endswith(SUF) and k not in CATS and k not in SKIP]
    n = len(all_fs)
    for c in CATS:
        cnt = {}
        for f in all_fs:
            v = num(f.get(c))
            if v is not None:
                cnt[v] = cnt.get(v, 0) + 1
        out += ["{}={:g}".format(c, v) for v in sorted(cnt) if cnt[v] > 0.02 * n]
    return out


_M = {}


def load():
    """式(無ければ None)。予想のたびに読み直さない"""
    if "m" not in _M:
        try:
            _M["m"] = json.load(open(FILE, encoding="utf-8"))
        except (OSError, ValueError):
            _M["m"] = None
    return _M["m"]


def score(rows, model=None):
    """rows = [(馬番, 材料の dict)](keiba_model.race_rows の返り)-> {馬番: (push, push2)}。式が無ければ {}"""
    m = model or load()
    if not m or not rows:
        return {}
    fs = [f for _, f in rows]
    w = [0.0] * len(rows)
    p = [0.0] * len(rows)
    for name, bw, bp in zip(m["features"], m["win"], m["pair"]):
        if not bw and not bp:
            continue
        for i, z in enumerate(column(fs, name)):
            if z:
                w[i] += bw * z
                p[i] += bp * z
    return {u: (round(w[i], 4), round(p[i], 4)) for i, (u, _) in enumerate(rows)}


def _lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def pair_probs(horses, tp, wide, push2, model=None):
    """後押しの式の組の確率(2頭とも3着内。2026-11-27)= オッズの土台(push_model.json の pair_base)+ 2頭の後押し(push2)。
    horses = [(馬番, 単勝オッズ)](取消を除く)、tp = 3連複から見た組の強さ、wide = ワイドのオッズ {"a-b": [下限, 上限]}、
    push2 = 馬番 -> 後押し(2頭とも3着内の式)。土台は tools/train_push.py の build の BP と同じ:
    [1, logit(pt), logit(pt)^2, logit(0.775/中), その2乗, 2頭の zq の和, 2頭の log q の和](pt = keiba_logic.PRE の pt × 組の強さ、
    中 = (下限 + 上限)/2、q = 単勝オッズから出した売れ方、zq = q の log をレースの中でそろえたもの)。アプリの pushPairProbs と同じ。
    -> {"a-b": 確率}。式・オッズが無ければ {}"""
    import keiba_logic as L
    m = model if model is not None else load()
    pb = (m or {}).get("pair_base")
    od = {int(u): float(o) for u, o in horses if o and float(o) > 1}
    if not pb or len(od) < 5 or not tp or not wide:
        return {}
    us = sorted(od)
    inv = {u: 1 / od[u] for u in us}; tot = sum(inv.values())
    lq = {u: math.log(inv[u] / tot) for u in us}
    zq = dict(zip(us, zs([lq[u] for u in us])))
    out = {}
    for k, v in wide.items():
        s = tp.get(k)
        lo, hi = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v, v)
        if not s or not lo or lo <= 1:
            continue
        a, b = (int(t) for t in k.split("-"))
        if a not in od or b not in od or push2.get(a) is None or push2.get(b) is None:
            continue
        x0 = _lg(min(L.PRE["pt"] * s, 0.98)); xw = _lg(min(0.775 / ((lo + hi) / 2), 0.98))
        f = [1.0, x0, x0 * x0, xw, xw * xw, zq[a] + zq[b], lq[a] + lq[b]]
        z = sum(w * x for w, x in zip(pb, f)) + push2[a] + push2[b]
        out[k] = 1 / (1 + math.exp(-z))
    return out


def race_pair_probs(rc, wide=None, model=None):
    """日別ファイルのレースから pair_probs(単勝オッズ・3連複の組の強さ・ワイドのオッズは発走前に取ったもの、後押しは予想のもの)"""
    hs = [(h["umaban"], h.get("odds")) for h in rc.get("horses") or [] if not h.get("scratched")]
    p2 = {h["umaban"]: h.get("push2") for h in ((rc.get("pred") or {}).get("horses") or [])}
    return pair_probs(hs, rc.get("trio_pair") or {}, wide if wide is not None else (rc.get("wide_odds") or {}), p2, model)


def level(v, cut):
    """後押しの強さ: 2 = 強い(上 1/3)、1 = ふつう、0 = 弱い(下 1/3)。cut = [下の境目, 上の境目]"""
    if v is None or not cut:
        return None
    return 0 if v <= cut[0] else 2 if v >= cut[1] else 1


def add(out, rows, center="jra"):
    """予想(keiba_logic.predict の返り)の馬に push・push2 を足す(中央だけ)"""
    if center != "jra" or not rows:
        return
    sc = score(rows)
    for h in out.get("horses") or []:
        v = sc.get(h.get("umaban"))
        if v:
            h["push"], h["push2"] = v

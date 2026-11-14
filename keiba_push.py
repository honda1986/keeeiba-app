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
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "keiba", "push_model.json")
SUF = ("_dhi", "_dlo", "_dmean", "_rank")       # レースの中でそろえ直した値(元の値だけ使う)
CATS = ("style", "sex")                          # 種類ごとの 0/1 にするもの
# レースの中で同じ値のもの(場・馬場など)と、コース傾向の2つ(表が 2021〜2025年の集計で、その年までの成績には先の結果が混ざる)は使わない
SKIP = ("venue", "surface", "center", "course_kata", "course_waku_idx", "course_style_idx")
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

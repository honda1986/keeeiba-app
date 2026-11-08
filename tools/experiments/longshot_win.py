"""単勝5倍以上の馬が勝つときの特徴(中央。2022〜2026年。障害は除く)
  python tools/experiments/longshot_win.py            表を出す
  python tools/experiments/longshot_win.py --pkl P    馬ごとの表を P に保存(続きの確かめ用)
  見方: 5倍以上の馬を条件で分けて、勝った率(実際)と単勝オッズから見込む勝つ率(見込み。レースの中で 1/オッズ を合計1にそろえる)をくらべる。
  実/見 が 1 より大きいほど、オッズの見込みより勝っている(人気より走る)。年ごとの 実/見 と単勝の回収率も出す。
  1つ目は「どんなレースで5倍以上が勝つか」(レースごと)、2つ目は「5倍以上のどんな馬が勝つか」(馬ごと)"""
import glob, gzip, json, os, re, sys
import numpy as np
import pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, R)
import keiba_weather as W
YS = ["2022", "2023", "2024", "2025", "2026"]
EAST = {"東京", "中山", "新潟", "福島", "札幌", "函館"}
WX = W.load()
KIND = {0: "晴", 1: "曇", 2: "雨", 3: "雪"}


def past_of(d):
    f = R + "/keiba/past/%s.json.gz" % d
    try:
        return json.load(gzip.open(f))
    except (OSError, ValueError):
        return {}


rows = []
for f in sorted(glob.glob(R + "/keiba/days/202[2-6]*.json")):
    d = os.path.basename(f)[:8]
    races = json.load(open(f)).get("races") or {}
    if not any(r.get("center") == "jra" for r in races.values()):
        continue
    past = past_of(d)
    for rid, r in races.items():
        res = r.get("result") or {}
        if r.get("center") != "jra" or r.get("surface") not in ("芝", "ダ") or len(res.get("order") or []) < 3:
            continue
        odds = {int(u): v for u, v in (res.get("odds") or {}).items() if v and v > 1}
        if len(odds) < 5:
            continue
        inv = {u: 1 / v for u, v in odds.items()}; tot = sum(inv.values())
        fav = min(odds.values())
        cond = r.get("cond") or ""; name = r.get("name") or ""
        wx = W.at(WX, r.get("venue"), d, r.get("start") or "12:00")
        pred = {h["umaban"]: h for h in (r.get("pred") or {}).get("horses") or []}
        rank = {u: i + 1 for i, u in enumerate(sorted(pred, key=lambda u: -(pred[u].get("p3") or 0)))}
        tan = res["payout"].get("単勝") or {}
        nn = len(odds)
        kin = [h.get("kinryo") for h in r.get("horses") or [] if h.get("kinryo") and h["umaban"] in odds]
        base = dict(rid=rid, y=d[:4], m=int(d[4:6]), venue=r.get("venue"), surface=r["surface"], dist=r.get("dist") or 0,
                    direction=r.get("direction"), baba=r.get("baba") or "?", cls=r.get("cls") or 0, shinba="新馬" in name, n=nn,
                    rno=r.get("rno") or 0, handicap="ハンデ" in cond, filly=bool(re.search(r"牝", cond.split("馬齢")[0] if "馬齢" in cond else cond)),
                    age2="2歳" in cond, age3="3歳" in cond and "以上" not in cond, fav=fav,
                    temp=wx[0] if wx else np.nan, wkind=KIND.get(wx[2], "?") if wx else "?", rain=wx[3] if wx else np.nan)
        pr = past.get(rid) or {}
        for h in r.get("horses") or []:
            u = h["umaban"]
            if u not in odds:
                continue
            ph = pred.get(u) or {}
            runs = pr.get(str(u)) or []
            last = runs[0] if runs else {}
            jk = (ph.get("jk") or {}).get("all") or [0, 0]
            sa = h.get("sex_age") or ""
            row = dict(base, u=u, odds=odds[u], q=inv[u] / tot, win=int(res["order"][0] == u), top3=int(u in res["order"][:3]),
                       pay=tan.get(str(u), 0), ninki=int((res.get("ninki") or {}).get(str(u)) or 0), waku=h.get("waku") or 0,
                       sex=sa[:1], age=int(sa[1:]) if sa[1:].isdigit() else 0,
                       kin_diff=(h.get("kinryo") or 0) - (np.mean(kin) if kin else 0),
                       jk_rate=jk[1] / jk[0] if jk[0] >= 50 else np.nan, east=(h.get("trainer") or "").startswith("美浦"),
                       style=ph.get("style") or "?", rest=ph.get("rest_days"), p3=ph.get("p3"), mrank=rank.get(u),
                       nruns=len(runs), last_rank=last.get("rank"), last_ninki=last.get("ninki"), last_margin=last.get("margin"),
                       last_dist=last.get("dist"), last_cls=last.get("cls"), last_surface=last.get("surface"), last_venue=last.get("venue"),
                       last_n=last.get("n"), last_pass=(last.get("pass") or [None])[-1], last3f=last.get("last3f"),
                       best3=min([x.get("rank") or 99 for x in runs[:3]] or [99]),
                       mud_n=sum(1 for x in runs if x.get("baba") in ("重", "不")),
                       mud_top3=sum(1 for x in runs if x.get("baba") in ("重", "不") and (x.get("rank") or 99) <= 3),
                       plus=" ".join(x["t"] for x in ph.get("plus") or []), minus=" ".join(x["t"] for x in ph.get("minus") or []))
            rows.append(row)
df = pd.DataFrame(rows)
L = df[df.odds >= 5].copy()
print("中央 {:,}R・{:,}頭。そのうち5倍以上 {:,}頭。5倍以上の馬が勝ったレース {:.0%}(オッズの見込み {:.0%})".format(
    df.rid.nunique(), len(df), len(L), L.groupby("rid").win.sum().mean(), L.groupby("rid").q.sum().mean()))
print("5倍以上の全体: 勝率 {:.1%} / 見込み {:.1%}(実/見 {:.2f})・単勝回収率 {:.0f}%".format(L.win.mean(), L.q.mean(), L.win.sum() / L.q.sum(), L.pay.sum() / len(L)))


def table(title, frame, key, order=None, minn=300):
    """frame を key で分けて、実/見(年ごと)と回収率"""
    print("\n■ " + title)
    print("  {:18s} {:>7s} {:>6s} {:>6s} {:>5s} {:>6s}   実/見 {}".format("", "頭", "勝率", "見込み", "実/見", "回収率", " ".join(y[2:] for y in YS)))
    g = frame.groupby(key)
    keys = order if order is not None else sorted(g.groups)
    for k in keys:
        if k not in g.groups:
            continue
        s = g.get_group(k)
        if len(s) < minn:
            continue
        ae = s.win.sum() / s.q.sum()
        ys = " ".join("{:4.2f}".format(s[s.y == y].win.sum() / s[s.y == y].q.sum()) if (s.y == y).sum() >= 100 else "  - " for y in YS)
        print("  {:18s} {:7,d} {:5.1%} {:5.1%} {:5.2f} {:5.0f}%   {}".format(str(k)[:18], len(s), s.win.mean(), s.q.mean(), ae, s.pay.sum() / len(s), ys))


def band(v, edges, labels):
    for e, lab in zip(edges, labels):
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and v < e:
            return lab
    return labels[-1] if v is not None and not (isinstance(v, float) and np.isnan(v)) else "不明"


# ---- 1. どんなレースで5倍以上が勝つか(レースごと: 勝ち馬が5倍以上だった率と、オッズの見込み)
RC = L.groupby("rid").agg(win=("win", "sum"), q=("q", "sum"), y=("y", "first")).join(
    df.groupby("rid").first()[["venue", "surface", "dist", "baba", "cls", "shinba", "n", "rno", "handicap", "filly", "age2", "age3", "m", "fav", "temp", "wkind", "rain", "direction"]])
RC["pay"] = 0


def rtable(title, key, order=None):
    print("\n■ " + title + "(レースごと: 5倍以上の馬が勝った率 / オッズの見込み)")
    g = RC.groupby(key)
    for k in (order if order is not None else sorted(g.groups)):
        if k not in g.groups:
            continue
        s = g.get_group(k)
        if len(s) < 150:
            continue
        ys = " ".join("{:4.2f}".format(s[s.y == y].win.sum() / s[s.y == y].q.sum()) if (s.y == y).sum() >= 60 else "  - " for y in YS)
        print("  {:14s} {:6,d}R 実際 {:4.0%} / 見込み {:4.0%}  実/見 {:4.2f}  [{}]".format(str(k)[:14], len(s), s.win.mean(), s.q.mean(), s.win.sum() / s.q.sum(), ys))


print("\n================ 1. どんなレースで5倍以上の馬が勝つか ================")
RC["距離"] = RC.dist.map(lambda v: band(v, [1401, 1801, 2201], ["〜1400m", "1500〜1800m", "1900〜2200m", "2300m〜"]))
RC["クラス"] = RC.apply(lambda r: "新馬" if r.shinba else {1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝", 5: "OP・L", 6: "G3", 7: "G2", 8: "G1"}.get(r.cls, "?"), axis=1)
RC["頭数"] = RC.n.map(lambda v: band(v, [11, 15, 17], ["〜10頭", "11〜14頭", "15〜16頭", "17〜18頭"]))
RC["1番人気"] = RC.fav.map(lambda v: band(v, [2.0, 3.0, 4.0], ["〜1.9倍", "2.0〜2.9倍", "3.0〜3.9倍", "4.0倍〜"]))
RC["季節"] = RC.m.map(lambda m: {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏", 7: "夏", 8: "夏"}.get(m, "秋"))
RC["気温"] = RC.temp.map(lambda v: band(v, [10, 20, 28], ["〜9℃", "10〜19℃", "20〜27℃", "28℃〜"]))
RC["雨量"] = RC.rain.map(lambda v: band(v, [0.1, 3, 10], ["0", "0.1〜2.9mm", "3〜9.9mm", "10mm〜"]))
RC["条件"] = RC.apply(lambda r: "ハンデ" if r.handicap else "牝馬限定" if r.filly else "2歳" if r.age2 else "3歳" if r.age3 else "古馬(3歳以上など)", axis=1)
RC["R"] = RC.rno.map(lambda v: band(v, [5, 9, 11, 12], ["1〜4R", "5〜8R", "9・10R", "11R", "12R"]))
rtable("競馬場", "venue")
rtable("芝・ダート", "surface")
rtable("距離", "距離", ["〜1400m", "1500〜1800m", "1900〜2200m", "2300m〜"])
rtable("馬場", "baba", ["良", "稍", "重", "不"])
rtable("天気(発走のころ)", "wkind", ["晴", "曇", "雨", "雪"])
rtable("朝からの雨量", "雨量", ["0", "0.1〜2.9mm", "3〜9.9mm", "10mm〜"])
rtable("気温", "気温", ["〜9℃", "10〜19℃", "20〜27℃", "28℃〜"])
rtable("季節", "季節", ["春", "夏", "秋", "冬"])
rtable("クラス", "クラス", ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP・L", "G3", "G2", "G1"])
rtable("レースの条件", "条件")
rtable("頭数", "頭数", ["〜10頭", "11〜14頭", "15〜16頭", "17〜18頭"])
rtable("レース番号", "R", ["1〜4R", "5〜8R", "9・10R", "11R", "12R"])
rtable("1番人気のオッズ", "1番人気", ["〜1.9倍", "2.0〜2.9倍", "3.0〜3.9倍", "4.0倍〜"])

# ---- 2. 5倍以上のどんな馬が勝つか(馬ごと)
print("\n================ 2. 5倍以上のどんな馬が勝つか(馬ごと) ================")
L["オッズ"] = L.odds.map(lambda v: band(v, [10, 20, 50], ["5〜9.9倍", "10〜19.9倍", "20〜49.9倍", "50倍〜"]))
L["枠"] = L.waku
L["性"] = L.sex.map({"牡": "牡", "牝": "牝", "セ": "セン"})
L["年齢"] = L.age.map(lambda a: "{}歳".format(a) if a <= 6 else "7歳〜")
L["斤量"] = L.kin_diff.map(lambda v: band(v, [-1.5, -0.4, 0.5, 1.5], ["平均-1.5kg以上軽い", "少し軽い", "平均くらい", "少し重い", "平均+1.5kg以上重い"]))
L["騎手の3着内率"] = L.jk_rate.map(lambda v: band(v, [0.2, 0.3, 0.4], ["20%未満", "20〜29%", "30〜39%", "40%以上"]))
L["東西"] = L.apply(lambda r: "地元(関東馬が関東・関西馬が関西)" if r.east == (r.venue in EAST) else "遠征", axis=1)
L["脚質"] = L["style"]
L["休み"] = L.rest.map(lambda v: band(v, [15, 36, 91, 181], ["連闘・中1週", "中2〜4週", "中5〜12週", "3〜6か月", "半年以上"]))
L["キャリア"] = L.nruns.map(lambda v: band(v, [1, 2, 4], ["初出走", "1走", "2〜3走", "4走以上"]))
L["前走着順"] = L.last_rank.map(lambda v: band(v, [2, 4, 6, 10], ["1着", "2〜3着", "4〜5着", "6〜9着", "10着以下"]))
L["前走人気"] = L.last_ninki.map(lambda v: band(v, [2, 4, 7, 10], ["1番人気", "2〜3番人気", "4〜6番人気", "7〜9番人気", "10番人気以下"]))
L["前走 着順-人気"] = (L.last_rank - L.last_ninki).map(lambda v: band(v, [-3, 0, 1, 4], ["人気より3つ以上上", "人気より上", "人気どおり", "人気より下", "人気より4つ以上下"]))
L["前走着差"] = L.last_margin.map(lambda v: band(v, [0.01, 0.4, 0.9, 1.6], ["勝ち(0秒)", "0.1〜0.3秒", "0.4〜0.8秒", "0.9〜1.5秒", "1.6秒以上"]))
L["前走4角"] = (L.last_pass / L.last_n).map(lambda v: band(v, [0.2, 0.45, 0.7], ["前(〜2割)", "前めの中", "後ろめの中", "後ろ(7割〜)"]))
L["前走上がり3F"] = L.apply(lambda r: band(r.last3f - (34.0 if r.last_surface == "芝" else 37.0) if r.last3f else None, [-1, 0, 1], ["速い(平均-1秒以上)", "少し速い", "少し遅い", "遅い(+1秒以上)"]), axis=1)
L["距離の変化"] = (L.dist - L.last_dist).map(lambda v: band(v, [-199, 200], ["短縮(200m以上)", "同じくらい", "延長(200m以上)"]))
L["クラスの変化"] = L.apply(lambda r: "新馬・初出走" if not r.last_cls else ("昇級" if r.cls > r.last_cls else "降級" if r.cls < r.last_cls else "同じクラス"), axis=1)
L["芝ダ替わり"] = L.apply(lambda r: "初出走" if not r.last_surface else ("替わり" if r.last_surface != r.surface else "同じ"), axis=1)
L["前走と同じ場"] = L.apply(lambda r: "初出走" if not r.last_venue else ("同じ場" if r.last_venue == r.venue else "違う場"), axis=1)
L["近3走の最高着順"] = L.best3.map(lambda v: band(v, [2, 4, 6, 99.5], ["1着あり", "2〜3着あり", "4〜5着まで", "6着以下だけ", "初出走"]))
L["モデルの確率の順位"] = L.mrank.map(lambda v: band(v, [2, 4, 7], ["1位", "2〜3位", "4〜6位", "7位以下"]))
L["モデル÷市場(3着内)"] = L.apply(lambda r: band((r.p3 or 0) / min(0.95, 3 * r.q) if r.p3 is not None else None, [0.8, 1.2, 1.6], ["市場より低い", "同じくらい", "1.2〜1.6倍", "1.6倍以上"]), axis=1)
for col, order in (("オッズ", ["5〜9.9倍", "10〜19.9倍", "20〜49.9倍", "50倍〜"]), ("枠", list(range(1, 9))), ("性", ["牡", "牝", "セン"]),
                   ("年齢", ["2歳", "3歳", "4歳", "5歳", "6歳", "7歳〜"]), ("斤量", None), ("騎手の3着内率", ["20%未満", "20〜29%", "30〜39%", "40%以上", "不明"]),
                   ("東西", None), ("脚質", ["逃げ", "先行", "差し", "追込", "?"]),
                   ("休み", ["連闘・中1週", "中2〜4週", "中5〜12週", "3〜6か月", "半年以上", "不明"]), ("キャリア", ["初出走", "1走", "2〜3走", "4走以上"]),
                   ("前走着順", ["1着", "2〜3着", "4〜5着", "6〜9着", "10着以下", "不明"]), ("前走人気", ["1番人気", "2〜3番人気", "4〜6番人気", "7〜9番人気", "10番人気以下", "不明"]),
                   ("前走 着順-人気", ["人気より3つ以上上", "人気より上", "人気どおり", "人気より下", "人気より4つ以上下", "不明"]),
                   ("前走着差", ["勝ち(0秒)", "0.1〜0.3秒", "0.4〜0.8秒", "0.9〜1.5秒", "1.6秒以上", "不明"]),
                   ("前走4角", ["前(〜2割)", "前めの中", "後ろめの中", "後ろ(7割〜)", "不明"]),
                   ("前走上がり3F", ["速い(平均-1秒以上)", "少し速い", "少し遅い", "遅い(+1秒以上)", "不明"]),
                   ("距離の変化", ["短縮(200m以上)", "同じくらい", "延長(200m以上)", "不明"]), ("クラスの変化", ["降級", "同じクラス", "昇級", "新馬・初出走"]),
                   ("芝ダ替わり", None), ("前走と同じ場", None), ("近3走の最高着順", ["1着あり", "2〜3着あり", "4〜5着まで", "6着以下だけ", "初出走"]),
                   ("モデルの確率の順位", ["1位", "2〜3位", "4〜6位", "7位以下", "不明"]), ("モデル÷市場(3着内)", ["市場より低い", "同じくらい", "1.2〜1.6倍", "1.6倍以上", "不明"])):
    table(col, L, col, order)


# ---- 3. 言われがちな組み合わせ
print("\n================ 3. 組み合わせ ================")
L["夏の牝馬"] = L.apply(lambda r: ("夏" if r.m in (6, 7, 8) else "冬" if r.m in (12, 1, 2) else "春秋") + "・" + ("牝" if r.sex == "牝" else "牡セン"), axis=1)
table("季節 × 性", L, "夏の牝馬", ["夏・牝", "夏・牡セン", "冬・牝", "冬・牡セン", "春秋・牝", "春秋・牡セン"])
L["道悪"] = L.apply(lambda r: "良・稍" if r.baba not in ("重", "不") else ("重・不良 × 道悪で3着内あり" if r.mud_top3 else "重・不良 × 道悪を走って3着内なし" if r.mud_n else "重・不良 × 道悪を走ったことがない"), axis=1)
table("道悪 × 道悪の実績", L, "道悪", ["重・不良 × 道悪で3着内あり", "重・不良 × 道悪を走って3着内なし", "重・不良 × 道悪を走ったことがない", "良・稍"])
SMALL = {"福島", "小倉", "函館", "札幌", "中山"}
L["コース×脚質"] = L.apply(lambda r: ("小回り" if r.venue in SMALL else "大きいコース") + "・" + r["style"], axis=1)
table("コース × 脚質(小回り=福島・小倉・函館・札幌・中山)", L, "コース×脚質", [a + "・" + b for a in ("小回り", "大きいコース") for b in ("逃げ", "先行", "差し", "追込")])
L["短距離×枠"] = L.apply(lambda r: "{}1400m以下・{}".format(r.surface, "内1〜3枠" if r.waku <= 3 else "外7・8枠" if r.waku >= 7 else "中4〜6枠") if r.dist <= 1400 else "1500m以上", axis=1)
table("1400m以下 × 枠", L, "短距離×枠", ["芝1400m以下・内1〜3枠", "芝1400m以下・中4〜6枠", "芝1400m以下・外7・8枠", "ダ1400m以下・内1〜3枠", "ダ1400m以下・中4〜6枠", "ダ1400m以下・外7・8枠"])
L["巻き返し"] = L.apply(lambda r: "前走1〜3番人気で10着以下" if (r.last_ninki or 99) <= 3 and (r.last_rank or 0) >= 10 else
                        "前走1〜3番人気で6〜9着" if (r.last_ninki or 99) <= 3 and (r.last_rank or 0) >= 6 else "ほか", axis=1)
table("巻き返し(前走で人気だったのに負けた)", L, "巻き返し", ["前走1〜3番人気で10着以下", "前走1〜3番人気で6〜9着", "ほか"])

# ---- 4. 2022〜2024年だけで特徴を選び、まとめた式で2025〜2026年の5倍以上の勝ち馬を当てられるか
print("\n================ 4. 特徴をまとめて、5倍以上の勝ち馬を当てられるか ================")
COLS = ["オッズ", "枠", "性", "年齢", "斤量", "東西", "脚質", "休み", "キャリア", "前走着順", "前走人気", "前走 着順-人気", "前走着差", "前走4角",
        "前走上がり3F", "距離の変化", "クラスの変化", "芝ダ替わり", "前走と同じ場", "近3走の最高着順", "モデルの確率の順位", "夏の牝馬", "道悪",
        "コース×脚質", "短距離×枠", "巻き返し"]
RCOLS = {"venue": "競馬場", "surface": "芝ダ", "baba": "馬場"}
TR = L.y <= "2024"
cand = {}
for c in COLS + list(RCOLS):
    for lv in L[c].dropna().unique():
        m = L[c] == lv
        if (m & TR).sum() < 1500 or str(lv) in ("不明", "?", "ほか"):
            continue
        aes = [L[m & (L.y == y)].win.sum() / L[m & (L.y == y)].q.sum() for y in ("2022", "2023", "2024")]
        if (all(a > 1 for a in aes) or all(a < 1 for a in aes)) and abs(np.mean(aes) - 1) >= 0.04:
            cand["{}={}".format(RCOLS.get(c, c), lv)] = m.astype(float).values
print("2022〜2024年の3年とも同じ向きに4%以上ずれた条件 {}個: {}".format(len(cand), " / ".join(cand)))
import keiba_final as F
X = np.column_stack([np.log(L.q.values)] + list(cand.values()))
y = L.win.values.astype(float); tr = TR.values; te = ~tr
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
Z = (X - mu) / sd
w = F.fit_logit(Z[tr], y[tr], 5.0); w0 = F.fit_logit(Z[tr][:, :1], y[tr], 5.0)
p = 1 / (1 + np.exp(-(w[0] + Z @ w[1:]))); p0 = 1 / (1 + np.exp(-(w0[0] + Z[:, :1] @ w0[1:])))
ll = lambda pp, m: float(np.mean(y[m] * np.log(pp[m]) + (1 - y[m]) * np.log(1 - pp[m])))
print("対数尤度(2025〜2026年。大きいほど当たる): オッズだけ {:.5f} / 条件も足す {:.5f}".format(ll(p0, te), ll(p, te)))
print("係数(+はオッズの見込みより勝つ): " + " / ".join("{} {:+.3f}".format(k, v) for k, v in sorted(zip(cand, w[2:]), key=lambda t: -abs(t[1]))))
L["ratio"] = p / p0
for th in (1.1, 1.2, 1.3):
    for name, m in (("2022〜2024(学んだ)", tr), ("2025", (L.y == "2025").values), ("2026", (L.y == "2026").values)):
        s = L[m & (L.ratio >= th).values]
        big = sorted(s.pay, reverse=True)
        print("  見込みの{}倍以上を単勝で {:14s} {:6,d}頭 勝率 {:4.1%} 回収率 {:4.0f}%(大きい3つを除くと {:3.0f}%)".format(
            th, name, len(s), s.win.mean() if len(s) else 0, s.pay.sum() / max(1, len(s)), sum(big[3:]) / max(1, len(s))))

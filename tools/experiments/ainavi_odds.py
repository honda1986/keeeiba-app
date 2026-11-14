"""AI競馬ナビ(horse-racing-ai-navi.com)の追切評価を、オッズに足すと的中率・回収率が上がるか。
読むだけで何も保存しない(結果の数字だけをログに出す)。
  python tools/experiments/ainavi_odds.py [最初の日 YYYYMMDD]
  (Actions の「記事を読む」を mode=ainavi・url=odds でまわす。ainavi_check.py がこのファイルを呼ぶ)
  ページ: https://horse-racing-ai-navi.com/jra/<日付>/<場コード>/ に埋め込まれた JSON(script#jra-index-report-data)。過去のページは 2026-07-18 から
  見るもの:
   1. 追切評価(S・A・B…)ごとの成績。「見込み」はオッズだけの式の確率(1着 = 単勝オッズ、3着内 = 3連複のオッズ)。
      実際が見込みより高ければ、オッズに入っていない分がある
   2. 1番人気・堅い単勝(1番人気が1.8倍以下)を追切評価で分けたときの的中率・回収率
   3. 堅いワイド(3連複から見ていちばん来そうな組が45%以上)を、2頭の追切評価で分けたとき
   4. 単勝オッズの式に追切を足したとき(週ごとに5つに分け、ほかの週で作った式で確かめる)
  4か月分(1,100レースほど)しか無いので、数字の幅(±)も出す"""
import glob, json, os, sys, time
from collections import defaultdict
import numpy as np
import requests
from bs4 import BeautifulSoup

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
START = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "20260718"
WIN_ODDS, WIDE_P, PT = 1.8, 0.45, 0.75            # 堅い買い目の基準(keiba_logic.SOLID)
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/124.0.0.0 Safari/537.36", "Accept-Language": "ja,en;q=0.8"})


def page(d, v):
    """その日・その場の JSON。無いページは None、つながらないときは False"""
    for _ in range(3):
        try:
            r = S.get("https://horse-racing-ai-navi.com/jra/{}/{}/".format(d, v), timeout=20)
            time.sleep(1.0)
            if r.status_code == 404:
                return None
            if r.ok:
                el = BeautifulSoup(r.text, "html.parser").find("script", id="jra-index-report-data")
                return json.loads(el.string) if el and el.string else None
        except requests.RequestException as e:
            print("ERR", d, v, type(e).__name__, flush=True)
            time.sleep(3)
    return False


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


RACES, pages, fails = [], {}, 0          # RACES = [{rid, d, hs: [馬], tp, wf, wpay}]
for f in sorted(glob.glob(R + "/keiba/days/2026*.json")):
    d = os.path.basename(f)[:8]
    if d < START:
        continue
    day = json.load(open(f))
    ours = {}
    for rid, r in (day.get("races") or {}).items():
        res = r.get("result") or {}
        if r.get("center") == "jra" and len(res.get("order") or []) >= 3 and res.get("odds"):
            ours.setdefault(rid[4:6], {})[int(rid[10:12])] = r
    for v, byno in sorted(ours.items()):
        j = page(d, v)
        pages[d + v] = "ok" if j else "なし" if j is None else "×"
        if j is False:
            fails += 1
            if fails >= 4 and not RACES:
                sys.exit("このマシンからはつながらない(もう一度まわす)")
            continue
        if not j:
            continue
        for race in (j.get("venue") or {}).get("races") or []:
            r = byno.get(int(race.get("no") or 0))
            if not r:
                continue
            res = r["result"]; order = res["order"]; pay = res.get("payout") or {}
            od = {int(u): o for u, o in res["odds"].items() if o and o > 1}
            if len(od) < 5 or order[0] not in od:
                continue
            inv = sum(1 / o for o in od.values())
            tp = r.get("trio_pair_final") or {}
            ts = defaultdict(float)
            for k, s_ in tp.items():
                for t in k.split("-"):
                    ts[int(t)] += s_
            hs = []
            for h in race.get("horses") or []:
                u = int(h.get("umaban") or 0)
                if u not in od:
                    continue
                fin = (h.get("training_data") or {}).get("final") or {}
                hs.append(dict(u=u, od=od[u], q=1 / od[u] / inv, win=int(u == order[0]), top3=int(u in order[:3]),
                               p3t=min(0.98, PT * ts[u] / 2) if ts.get(u) else None,
                               tb=(h.get("training_bunrui") or "-"), td=num(h.get("training_deviation")),
                               fi=num(h.get("fukusho_index")), best=bool(num(fin.get("rank_4f")) == 1 and (num(fin.get("bunbo")) or 0) >= 2)))
            if len(hs) >= 5 and any(h["win"] for h in hs):
                RACES.append(dict(rid=r["id"], d=d, hs=hs, tp=tp, wf=r.get("wide_final") or {}, wpay=pay.get("ワイド") or {}, top3=set(order[:3])))
ok = sorted(k for k, s in pages.items() if s == "ok")
print("ページ: 読めた {} / 無い {} / つながらない {}".format(len(ok), sum(1 for s in pages.values() if s == "なし"), sum(1 for s in pages.values() if s == "×")))
print("読めた日: {} 〜 {}".format(ok[0][:8] if ok else "-", ok[-1][:8] if ok else "-"))
NH = sum(len(r["hs"]) for r in RACES)
print("レース {} / 馬 {} 頭".format(len(RACES), NH))
if len(RACES) < 300:
    sys.exit("数が少なすぎる")

lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
pm = lambda p, n: 196 * (p * (1 - p) / max(n, 1)) ** 0.5       # 的中率の95%の幅(±ポイント)


def cl_fit(groups):
    """条件つきロジット。groups = [(X(n×k), y(n))]。レースの中で1頭だけ y=1"""
    k = groups[0][0].shape[1]; b = np.zeros(k); b[0] = 1.0
    for _ in range(40):
        g = np.zeros(k); H = np.zeros((k, k))
        for X, y in groups:
            s = X @ b; e = np.exp(s - s.max()); p = e / e.sum()
            g += X.T @ (y - p); m = X.T @ p; H += (X * p[:, None]).T @ X - np.outer(m, m)
        step = np.linalg.solve(H + 1e-6 * np.eye(k), g); b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def cl_p(X, b):
    s = X @ b; e = np.exp(s - s.max()); return e / e.sum()


def lr_fit(X, y):
    Z = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Z.shape[1])
    for _ in range(40):
        p = 1 / (1 + np.exp(-Z @ w))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + 1e-6 * np.eye(len(w)), Z.T @ (y - p)); w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


# ---- 追切の数字をレースの中でそろえる(偏差値: 平均0・ばらつき1。無い馬は0)。オッズだけの式の見込みも付ける
for r in RACES:
    td = np.array([h["td"] if h["td"] is not None else np.nan for h in r["hs"]]); okm = ~np.isnan(td)
    z = np.zeros(len(td))
    if okm.sum() >= 3 and np.nanstd(td) > 1e-9:
        z[okm] = np.clip((td[okm] - np.nanmean(td)) / np.nanstd(td), -3, 3)
    fi = np.array([h["fi"] if h["fi"] is not None else np.nan for h in r["hs"]]); okf = ~np.isnan(fi)
    zf = np.zeros(len(fi))
    if okf.sum() >= 3 and np.nanstd(fi) > 1e-9:
        zf[okf] = np.clip((fi[okf] - np.nanmean(fi)) / np.nanstd(fi), -3, 3)
    rk = (-np.where(okm, td, -1e9)).argsort().argsort() + 1
    for i, h in enumerate(r["hs"]):
        h["tz"] = z[i]; h["fz"] = zf[i]; h["has"] = bool(okm[i]); h["trank"] = int(rk[i]) if okm[i] else None
        h["sa"] = h["tb"] in ("S", "A")
b0 = cl_fit([(np.array([[np.log(h["q"])] for h in r["hs"]]), np.array([h["win"] for h in r["hs"]], float)) for r in RACES])
for r in RACES:
    p = cl_p(np.array([[np.log(h["q"])] for h in r["hs"]]), b0)
    for i, h in enumerate(r["hs"]):
        h["pb"] = p[i]
HS = [h for r in RACES for h in r["hs"]]
m3 = [h for h in HS if h["p3t"] is not None]
w3 = lr_fit(lg(np.array([h["p3t"] for h in m3]))[:, None], np.array([h["top3"] for h in m3], float))
for h in HS:
    h["p3b"] = 1 / (1 + np.exp(-(w3[0] + w3[1] * lg(h["p3t"])))) if h["p3t"] is not None else None
print("追切評価の種類:", {b: sum(1 for h in HS if h["tb"] == b) for b in sorted({h["tb"] for h in HS})}, "/ 偏差値がある馬 {:.0%}".format(np.mean([h["has"] for h in HS])))


def row(name, hs):
    """馬の集まりの成績: 1着(実際/単勝オッズの見込み/単勝の回収率)、3着内(実際/3連複の見込み)。複勝の払戻は 10/24 からしか記録が無いので出さない"""
    n = len(hs)
    if n < 20:
        print("  {:22s} {:6d}頭 (少ない)".format(name, n)); return
    w = np.mean([h["win"] for h in hs]); e = np.mean([h["pb"] for h in hs]); roi = 100 * np.mean([h["win"] * h["od"] for h in hs])
    h3 = [h for h in hs if h["p3b"] is not None]
    t = np.mean([h["top3"] for h in h3]) if h3 else 0; e3 = np.mean([h["p3b"] for h in h3]) if h3 else 0
    print("  {:22s} {:6d}頭 1着 {:5.1%}(見込み {:5.1%} ±{:.1f}) 単勝 {:5.1f}% | 3着内 {:5.1%}(見込み {:5.1%} ±{:.1f})".format(
        name, n, w, e, pm(e, n), roi, t, e3, pm(e3, len(h3))))


GR = sorted({h["tb"] for h in HS})
print("\n■ 1. 追切評価ごとの成績(全部の馬)。見込み = オッズだけの式の確率。±は見込みの95%の幅(ポイント)")
for b in GR:
    row("追切 " + b, [h for h in HS if h["tb"] == b])
for lo, hi in ((0, 45), (45, 50), (50, 55), (55, 60), (60, 999)):
    row("偏差値 {}〜{}".format(lo, hi if hi < 999 else ""), [h for h in HS if h["has"] and lo <= h["td"] < hi])
row("レースの中で偏差値1位", [h for h in HS if h["trank"] == 1])
row("自己ベスト", [h for h in HS if h["best"]])
print("  -- 人気ごと(単勝オッズ)")
for nm, lo, hi in (("単勝 〜3倍", 0, 3), ("単勝 3〜10倍", 3, 10), ("単勝 10〜30倍", 10, 30)):
    for g, f in (("S・A", lambda h: h["sa"]), ("B", lambda h: h["tb"] == "B"), ("それより下・無し", lambda h: not h["sa"] and h["tb"] != "B")):
        row("{} 追切{}".format(nm, g), [h for h in HS if lo <= h["od"] < hi and f(h)])

print("\n■ 2. 1番人気を追切評価で分ける")
FAV = [min(r["hs"], key=lambda h: (h["od"], h["u"])) for r in RACES]
for nm, sel in (("1番人気(全部)", FAV), ("堅い単勝(1.8倍以下)", [h for h in FAV if h["od"] <= WIN_ODDS])):
    row(nm, sel)
    for g, f in (("追切 S・A", lambda h: h["sa"]), ("追切 B", lambda h: h["tb"] == "B"), ("追切 それより下・無し", lambda h: not h["sa"] and h["tb"] != "B"),
                 ("偏差値がレースで1〜2位", lambda h: h["trank"] in (1, 2)), ("偏差値がレースで3位より下", lambda h: h["trank"] is not None and h["trank"] >= 3),
                 ("偏差値 55以上", lambda h: h["has"] and h["td"] >= 55), ("偏差値 50未満", lambda h: h["has"] and h["td"] < 50)):
        row("  " + g, [h for h in sel if f(h)])

print("\n■ 3. 堅いワイド(3連複から見ていちばん来そうな組が45%以上)を、2頭の追切評価で分ける")
PW = []
for r in RACES:
    byu = {h["u"]: h for h in r["hs"]}
    best = None
    for k, v in r["wf"].items():
        s_ = r["tp"].get(k)
        if not s_ or not v or v[0] <= 1:
            continue
        if best is None or (s_, k) > (best[0], best[1]):
            best = (s_, k)
    if not best:
        continue
    a, b = (int(t) for t in best[1].split("-"))
    if a not in byu or b not in byu:
        continue
    PW.append(dict(p=PT * best[0], hit=int(a in r["top3"] and b in r["top3"]), pay=r["wpay"].get(best[1], 0) / 100.0,
                   nsa=int(byu[a]["sa"]) + int(byu[b]["sa"]), z=byu[a]["tz"] + byu[b]["tz"]))
wa = lr_fit(lg(np.array([x["p"] for x in PW]))[:, None], np.array([x["hit"] for x in PW], float))
for x in PW:
    x["pb"] = 1 / (1 + np.exp(-(wa[0] + wa[1] * lg(x["p"]))))


def prow(name, xs):
    n = len(xs)
    if n < 15:
        print("  {:26s} {:5d}R (少ない)".format(name, n)); return
    e = np.mean([x["pb"] for x in xs])
    print("  {:26s} {:5d}R 的中 {:5.1%}(見込み {:5.1%} ±{:.1f}) 回収率 {:5.1f}%".format(name, n, np.mean([x["hit"] for x in xs]), e, pm(e, n), 100 * np.mean([x["pay"] for x in xs])))


for nm, sel in (("いちばん来そうな組(全レース)", PW), ("堅いワイド(45%以上)", [x for x in PW if x["p"] >= WIDE_P - 1e-9])):
    prow(nm, sel)
    for g, f in (("2頭とも追切 S・A", lambda x: x["nsa"] == 2), ("1頭だけ S・A", lambda x: x["nsa"] == 1), ("どちらも S・A でない", lambda x: x["nsa"] == 0),
                 ("2頭の偏差値(そろえた値)の合計が+", lambda x: x["z"] > 0), ("合計が−", lambda x: x["z"] <= 0)):
        prow("  " + g, [x for x in sel if f(x)])

print("\n■ 4. 単勝オッズの式に足す(週ごとに5つに分け、ほかの週で作った式で確かめる)")
weeks = sorted({int(r["d"]) // 7 for r in RACES}); fold = {w: i % 5 for i, w in enumerate(weeks)}
SETS = [("単勝オッズだけ", lambda h: [np.log(h["q"])]),
        ("+追切の偏差値", lambda h: [np.log(h["q"]), h["tz"]]),
        ("+追切の偏差値・S/A・自己ベスト", lambda h: [np.log(h["q"]), h["tz"], float(h["sa"]), float(h["best"])]),
        ("+複勝能力指数", lambda h: [np.log(h["q"]), h["fz"]]),
        ("+追切の偏差値+複勝能力指数", lambda h: [np.log(h["q"]), h["tz"], h["fz"]])]
for name, fx in SETS:
    G = [(np.array([fx(h) for h in r["hs"]], float), np.array([h["win"] for h in r["hs"]], float), fold[int(r["d"]) // 7], r) for r in RACES]
    ll = hit = pay = 0.0; coef = []; ups = []; sel = []
    for k in range(5):
        b = cl_fit([(X, y) for X, y, f_, _ in G if f_ != k]); coef.append(b)
        for X, y, f_, r in G:
            if f_ != k:
                continue
            p = cl_p(X, b); i = int(np.argmax(p)); ll += np.log(p[y == 1][0]); hit += y[i]; pay += y[i] * r["hs"][i]["od"]
            sel.append((p[i], y[i], y[i] * r["hs"][i]["od"]))
            if X.shape[1] > 1:
                adj = X[:, 1:] @ b[1:]
                for jx, h in enumerate(r["hs"]):
                    if h["od"] <= 20:
                        ups.append((adj[jx], h["win"], h["pb"], h["win"] * h["od"]))
    n = len(G); sel.sort(key=lambda t: -t[0]); k = sum(1 for h in FAV if h["od"] <= WIN_ODDS); top = sel[:k]
    line = "  {:30s} 対数尤度 {:.5f}/R 全レースの1位 的中 {:5.1%}(±{:.1f}) 回収率 {:5.1f}% | 堅い単勝と同じ数({}R) 的中 {:5.1%} 回収率 {:5.1f}%".format(
        name, ll / n, hit / n, pm(hit / n, n), 100 * pay / n, k, np.mean([t[1] for t in top]), 100 * np.mean([t[2] for t in top]))
    if ups:
        ups.sort(key=lambda t: -t[0]); u10 = ups[:len(ups) // 10]
        line += " | 押し上げる上1割 {}頭 的中 {:.1%}(見込み {:.1%}) 単勝 {:.0f}%".format(len(u10), np.mean([t[1] for t in u10]), np.mean([t[2] for t in u10]), 100 * np.mean([t[3] for t in u10]))
        line += " 係数 " + " ".join("{:+.3f}".format(v) for v in np.mean(coef, axis=0)[1:])
    print(line)
print("\n(このサイトのデータは保存していない)")

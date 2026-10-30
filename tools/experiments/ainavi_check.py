"""AI競馬ナビ(horse-racing-ai-navi.com)の過去のページから追切評価と複勝能力指数を読み、うちの月ごとの3着内の確率に
足すと当たり方が良くなるかを確かめる。読むだけで何も保存しない(結果の数字だけをログに出す)。
  python tools/experiments/ainavi_check.py [最初の日 YYYYMMDD]
  ページ: https://horse-racing-ai-navi.com/jra/<日付>/<場コード>/ に埋め込まれた JSON(script#jra-index-report-data)
  比べ方: 日付の前半で混ぜ方(ロジスティック回帰)を決め、後半の対数尤度・AUC で比べる"""
import glob, gzip, json, math, os, sys, time
import numpy as np
import requests
from bs4 import BeautifulSoup

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
START = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "20260401"
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
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


rows, pages, fails = [], {}, 0
for f in sorted(glob.glob(R + "/keiba/days/2026*.json")):
    d = os.path.basename(f)[:8]
    if d < START:
        continue
    day = json.load(open(f))
    ours = {}
    for rid, r in (day.get("races") or {}).items():
        res = r.get("result") or {}
        if r.get("center") == "jra" and rid in wf and len(res.get("order") or []) >= 3 and res.get("odds"):
            ours.setdefault(rid[4:6], {})[int(rid[10:12])] = r
    for v, byno in sorted(ours.items()):
        j = page(d, v)
        pages[d + v] = "ok" if j else "なし" if j is None else "×"
        if j is False:
            fails += 1
            if fails >= 4 and not rows:
                sys.exit("このマシンからはつながらない(もう一度まわす)")
            continue
        if not j:
            continue
        for race in (j.get("venue") or {}).get("races") or []:
            r = byno.get(int(race.get("no") or 0))
            if not r:
                continue
            rid = r["id"]
            top3 = set(r["result"]["order"][:3])
            od = r["result"]["odds"]
            inv = sum(1 / o for o in od.values() if o and o > 1)
            for h in race.get("horses") or []:
                u = int(h.get("umaban") or 0)
                p3 = wf[rid].get(str(u))
                if p3 is None or not od.get(str(u)):
                    continue
                fin = (h.get("training_data") or {}).get("final") or {}
                f4 = num(fin.get("wood_4f_dev")) if fin.get("wood_4f_dev") is not None else num(fin.get("sakamichi_4f_dev"))
                f1 = num(fin.get("wood_1f_dev")) if fin.get("wood_1f_dev") is not None else num(fin.get("sakamichi_1f_dev"))
                rows.append(dict(d=d, rid=rid, y=int(u in top3), p3=p3, q=1 / od[str(u)] / inv,
                                 fi=num(h.get("fukusho_index")), ki=num(h.get("kitaichi_index")),
                                 td=num(h.get("training_deviation")), tb=h.get("training_bunrui") or "-",
                                 f4=f4, f1=f1, r4=num(fin.get("rank_4f")), bb=num(fin.get("bunbo")),
                                 tc=(fin.get("course") or "-")))
ok = sorted(k for k, s in pages.items() if s == "ok")
print("ページ: 読めた {} / 無い {} / つながらない {}".format(len(ok), sum(1 for s in pages.values() if s == "なし"),
                                                    sum(1 for s in pages.values() if s == "×")))
print("読めた日: {} 〜 {}".format(ok[0][:8] if ok else "-", ok[-1][:8] if ok else "-"))
print("無かった日(最初の10):", [k for k, s in sorted(pages.items()) if s == "なし"][:10])
print("馬 {} 頭 / レース {}".format(len(rows), len({r["rid"] for r in rows})))
if len(rows) < 2000:
    sys.exit("数が少なすぎる")


def auc(s, y):
    o = np.argsort(s, kind="mergesort"); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    pos = y.sum(); return (rk[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
y = np.array([r["y"] for r in rows]); p3 = np.array([r["p3"] for r in rows]); q = np.array([r["q"] for r in rows])
fi = np.array([(r["fi"] if r["fi"] is not None else 20.0) / 100 for r in rows])
tdv = np.array([r["td"] if r["td"] is not None else np.nan for r in rows])
has_td = ~np.isnan(tdv); tdz = np.where(has_td, (np.nan_to_num(tdv, nan=50) - 50) / 10, 0.0)
f1 = np.array([r["f1"] if r["f1"] is not None else np.nan for r in rows]); has_f1 = ~np.isnan(f1)
f1z = np.where(has_f1, (np.nan_to_num(f1, nan=50) - 50) / 10, 0.0)
best = np.array([1.0 if r["r4"] == 1 and (r["bb"] or 0) >= 2 else 0.0 for r in rows])
print("\n■ 単独の当たり方(AUC)  うちの確率 {:.4f} / 市場(単勝) {:.4f} / 複勝能力指数 {:.4f} / 追切偏差値(ある馬だけ) {:.4f}".format(
    auc(p3, y), auc(q, y), auc(fi, y), auc(tdv[has_td], y[has_td])))
print("  追切偏差値がある馬 {:.0%}".format(has_td.mean()))
print("\n■ 追切評価ごとの3着内率(実際 / うちの確率の見込み)")
for b in sorted({r["tb"] for r in rows}):
    m = np.array([r["tb"] == b for r in rows])
    print("  {:3s} {:6d}頭  実際 {:5.1%} / 見込み {:5.1%}".format(b, int(m.sum()), y[m].mean(), p3[m].mean()))
for lo, hi in ((0, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 999)):
    m = has_td & (tdv >= lo) & (tdv < hi)
    if m.sum():
        print("  偏差値 {:3d}〜{:<3d} {:6d}頭  実際 {:5.1%} / 見込み {:5.1%}".format(lo, hi if hi < 999 else 99, int(m.sum()), y[m].mean(), p3[m].mean()))
m = best == 1
print("  自己ベスト(4F 1位・2回以上) {:6d}頭  実際 {:5.1%} / 見込み {:5.1%}".format(int(m.sum()), y[m].mean(), p3[m].mean()))


def fit(X, yy):
    Z = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Z.shape[1])
    for _ in range(40):
        p = 1 / (1 + np.exp(-Z @ w)); H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, Z.T @ (yy - p))
    return w


def pred(X, w):
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))


dates = np.array([r["d"] for r in rows]); cut = np.quantile(dates.astype(int), 0.6)
tr, te = dates.astype(int) <= cut, dates.astype(int) > cut
print("\n■ 混ぜ方を {}〜{} で決め、{}〜{} で確かめる(後半 {}頭)".format(dates[tr].min(), dates[tr].max(), dates[te].min(),
                                                            dates[te].max(), int(te.sum())))
SETS = [("うちの確率だけ", [lg(p3)]), ("+追切(偏差値・1F・自己ベスト)", [lg(p3), tdz, has_td * 1.0, f1z, has_f1 * 1.0, best]),
        ("+複勝能力指数", [lg(p3), lg(fi)]), ("+複勝能力指数+追切", [lg(p3), lg(fi), tdz, has_td * 1.0, f1z, has_f1 * 1.0, best]),
        ("市場だけ", [lg(q)]), ("市場+追切", [lg(q), tdz, has_td * 1.0, f1z, has_f1 * 1.0, best]),
        ("市場+うちの確率+複勝能力指数+追切", [lg(q), lg(p3), lg(fi), tdz, has_td * 1.0, f1z, has_f1 * 1.0, best])]
for name, cols in SETS:
    X = np.column_stack(cols); w = fit(X[tr], y[tr]); p = pred(X[te], w)
    ll = np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    print("  {:32s} 対数尤度 {:.5f}  AUC {:.4f}  係数 {}".format(name, ll, auc(p, y[te]), " ".join("%.3f" % v for v in w[1:])))
print("\n(このサイトのデータは保存していない)")

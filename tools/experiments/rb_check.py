"""RACING BOOK(racing-book.net/app/)の逃げ馬・差し馬・決め手メーターと追いきりマスターの過去のページを読み、
うちの月ごとの3着内の確率に足すと当たり方が良くなるかを確かめる。読むだけで何も保存しない(結果の数字だけをログに出す)。
  python tools/experiments/rb_check.py
  対象は各ページに載っている特別レース(だいたい9R以降)。ページのレースは、馬名がいちばん重なるうちのレースに結びつける。
  比べ方: 週ごとに5つに分け、ほかの週で混ぜ方(ロジスティック回帰)を決めた確率の対数尤度を、うちの確率だけ・市場だけと比べる"""
import datetime, glob, gzip, json, math, os, re, sys, time
import numpy as np
import requests
from bs4 import BeautifulSoup

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/124.0.0.0 Safari/537.36", "Accept-Language": "ja,en;q=0.8"})
BASE = "https://racing-book.net/app/"
# 過去のページがある日(read_site.py probe で確かめた。2026-10-30)
DAYS = {
    "nige": "20260628 20260704 20260705 20260711 20260712 20260718 20260719 20260725 20260726 20260801 20260802 20260808 "
            "20260809 20260815 20260816 20260822 20260823 20260829 20260830 20260905 20260906 20260912 20260913 20260919 "
            "20260920 20260921 20260926 20260927 20261003 20261004 20261010 20261011 20261012 20261017 20261018 20261024 20261025",
    "sashi": "20260816 20260829 20260830 20260905 20260906 20260912 20260913 20260919 20260920 20260921 20260926 20260927 "
             "20261003 20261004 20261010 20261011 20261012 20261017 20261018 20261024 20261025",
    "kimete": "20260906 20260912 20260913 20260919 20260920 20260921 20260926 20260927 20261003 20261004 20261010 20261011 "
              "20261012 20261017 20261018 20261024 20261025",
    "oikiri": "20260726 20260809 20260816 20260823 20260830 20260906 20260913 20260920 20260921 20260927 20261004 20261012 "
              "20261018 20261025",
}
URL = {"nige": "nigeuma_meter_{}.html", "sashi": "sashiuma_meter_{}.html", "kimete": "kimete_meter_{}.html", "oikiri": "oikiri_{}.html"}

# うちのレース(中央・結果と月ごとの確率があるもの)を日付ごとに
ours = {}
for f in sorted(glob.glob(R + "/keiba/days/2026*.json")):
    d = os.path.basename(f)[:8]
    if d < "20260620":
        continue
    for rid, r in (json.load(open(f)).get("races") or {}).items():
        res = r.get("result") or {}
        if r.get("center") == "jra" and rid in wf and len(res.get("order") or []) >= 3 and res.get("odds") and r.get("horses"):
            ours.setdefault(d, []).append(r)
            r["_d"] = int(d)


def get(u):
    for _ in range(3):
        try:
            r = S.get(u, timeout=40)
            time.sleep(1.0)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException as e:
            print("ERR", u, type(e).__name__, flush=True)
            time.sleep(3)
    return None


def own(el, name, of):
    return [x for x in el.find_all(name) if x.find_parent(of.name) is of]


def tables(sec):
    """その入れ物の中の、馬番の列がある表(入れ子の表は除く)ごとに [{列名: 文字}]"""
    out = []
    for tb in sec.find_all("table"):
        if tb.find_parent("table"):
            continue
        trs = [tr for tr in tb.find_all("tr") if tr.find_parent("table") is tb]
        if not trs:
            continue
        head = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in own(trs[0], "th", trs[0]) or own(trs[0], "td", trs[0])]
        if "馬番" not in head:
            continue
        rows = []
        for tr in trs[1:]:
            cs = [c for c in tr.find_all(["td", "th"]) if c.find_parent("tr") is tr]
            if len(cs) < len(head) - 1:
                continue
            rows.append({h: re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for h, c in zip(head, cs)})
        out.append(rows)
    return out


def match(rows, cands):
    """表の馬名がいちばん重なるうちのレース"""
    txt = " ".join(r.get("馬名", "") for r in rows)
    best, bn = None, 0
    for rc in cands:
        n = sum(1 for h in rc["horses"] if h.get("name") and h["name"] in txt)
        if n > bn:
            best, bn = rc, n
    return best if best and bn >= max(3, 0.5 * len(rows)) else None


def num(s):
    m = re.search(r"-?\d+(\.\d+)?", s or "")
    return float(m.group()) if m else None


BADGES = ["BEST+加速", "BEST更新", "時計良化", "パターン変更", "型注意", "型好転", "連闘注意", "手前◎", "手前△", "いつも通り", "判定対象外", "滞在"]
feat = {}          # (rid, 馬番) -> {名前: 値}
info = {}
for src, ds in DAYS.items():
    nr = npg = 0
    for d in ds.split():
        html = get(BASE + URL[src].format(d))
        if not html:
            print("読めない", src, d, flush=True)
            continue
        npg += 1
        soup = BeautifulSoup(html, "html.parser")
        dd = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
        cands = [rc for k in range(0, 3 if src == "oikiri" else 1) for rc in ours.get((dd - datetime.timedelta(days=k)).strftime("%Y%m%d"), [])]
        secs = soup.select("section.race") or soup.select(".race")
        for sec in secs:
            for rows in tables(sec):
                rc = match(rows, cands)
                if not rc:
                    continue
                nr += 1
                full = len(rows) >= 0.8 * len(rc["horses"])
                for h in rc["horses"]:
                    row = next((x for x in rows if h.get("name") and h["name"] in x.get("馬名", "")), None)
                    k = (rc["id"], str(h["umaban"]))
                    v = feat.setdefault(k, {})
                    if row is None:
                        if src == "oikiri" and full:
                            v["oi_seen"] = 1
                        continue
                    if src == "nige":
                        v["nige"] = num(row.get("逃げ馬率")); v["nige_sp"] = num(row.get("先行度"))
                    elif src == "sashi":
                        v["sashi"] = num(row.get("差し率")); v["sashi_pw"] = num(row.get("差し力"))
                    elif src == "kimete":
                        v["kime"] = num(row.get("決め手率")); v["kime_chu"] = 1.0 if row.get("馬名", "").startswith("注") else 0.0
                    else:
                        t = row.get("調教の見立て", "")
                        v["oi_seen"] = 1
                        for b in BADGES:
                            if b in t:
                                v["oi_" + b] = 1
                        v["oi_core"] = 1 if any(b in row.get("調教の見立て", "") for b in BADGES[:4]) else 0
                    info.setdefault(src, set()).add(rc["id"])
    print("{:7s} ページ {:3d} / 結びついたレース {:4d}".format(src, npg, len(info.get(src, ()))), flush=True)

# 行: うちの確率・市場・結果
rows = []
for rid in sorted({k[0] for k in feat}):
    rc = next(r for rs in ours.values() for r in rs if r["id"] == rid)
    res = rc["result"]; od = res["odds"]; top3 = set(res["order"][:3])
    inv = sum(1 / o for o in od.values() if o and o > 1)
    for h in rc["horses"]:
        u = str(h["umaban"])
        if u not in wf[rid] or not od.get(u):
            continue
        v = feat.get((rid, u), {})
        rows.append(dict(rid=rid, d=rc["_d"], u=u,
                         y=int(int(u) in top3), p3=wf[rid][u], q=1 / od[u] / inv, v=v))
print("馬 {} 頭 / レース {}".format(len(rows), len({r["rid"] for r in rows})))
y = np.array([r["y"] for r in rows]); p3 = np.array([r["p3"] for r in rows]); q = np.array([r["q"] for r in rows])
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def auc(s, yy):
    o = np.argsort(s, kind="mergesort"); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    pos = yy.sum(); return (rk[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(yy) - pos)) if 0 < pos < len(yy) else float("nan")


def fit(X, yy):
    Z = np.column_stack([np.ones(len(X)), X]); w = np.zeros(Z.shape[1])
    for _ in range(40):
        p = 1 / (1 + np.exp(-Z @ w)); H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-3 * np.eye(len(w))
        w += np.linalg.solve(H, Z.T @ (yy - p))
    return w


def pred(X, w):
    return 1 / (1 + np.exp(-(w[0] + X @ w[1:])))


weeks = np.array([r["d"] // 7 for r in rows])


def oof_ll(cols, m):
    """週ごとに5つに分け、ほかの週で決めた混ぜ方での対数尤度(m の行だけ)"""
    X = np.column_stack(cols)[m]; yy = y[m]; wk = weeks[m]; uw = sorted(set(wk)); fold = np.array([uw.index(w) % 5 for w in wk])
    p = np.zeros(len(yy))
    for k in range(5):
        if (fold == k).any() and (fold != k).any():
            w = fit(X[fold != k], yy[fold != k]); p[fold == k] = pred(X[fold == k], w)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)), fit(X, yy)


def val(name, default=np.nan):
    return np.array([r["v"].get(name, default) if r["v"].get(name) is not None else default for r in rows], dtype=float)


for src, name, lab in (("nige", "nige", "逃げ馬率"), ("sashi", "sashi", "差し率"), ("kimete", "kime", "決め手率")):
    x = val(name); m = ~np.isnan(x) & np.isin([r["rid"] for r in rows], list(info.get(src, ())))
    inr = np.isin([r["rid"] for r in rows], list(info.get(src, ())))
    if m.sum() < 200:
        print("\n■", lab, "数が少ない", int(m.sum())); continue
    print("\n■ {}({}頭に値 / 載っているレースの馬 {}頭)".format(lab, int(m.sum()), int(inr.sum())))
    print("  AUC(3着内): {} {:.4f} / うちの確率 {:.4f} / 市場 {:.4f}(値のある馬)".format(lab, auc(x[m], y[m]), auc(p3[m], y[m]), auc(q[m], y[m])))
    qs = np.nanquantile(x[m], [0, .5, .75, .9, 1])
    for lo, hi in zip(qs[:-1], qs[1:]):
        b = m & (x >= lo) & (x <= hi)
        if b.sum():
            print("  {} {:5.1f}〜{:5.1f} {:5d}頭  3着内 実際 {:5.1%} / うちの見込み {:5.1%} / 市場の見込み(1着の確率×2.6) {:5.1%}".format(
                lab, lo, hi, int(b.sum()), y[b].mean(), p3[b].mean(), min(1, 2.6 * q[b].mean())))
    has = (~np.isnan(x)).astype(float); xz = np.where(np.isnan(x), 0, np.log1p(np.nan_to_num(x)))
    for base, bl in (([lg(p3)], "うちの確率"), ([lg(q)], "市場"), ([lg(p3), lg(q)], "うち+市場")):
        l0, _ = oof_ll(base, inr); l1, w = oof_ll(base + [xz, has], inr)
        print("  {:10s} 対数尤度 {:.5f} → +{} {:.5f}(差 {:+.5f}) 係数 {}".format(bl, l0, lab, l1, l1 - l0, " ".join("%.2f" % v for v in w[1:])))

# 追いきりマスター
inr = np.isin([r["rid"] for r in rows], list(info.get("oikiri", ())))
seen = val("oi_seen", 0) == 1
print("\n■ 追いきりマスター(載っているレースの馬 {}頭 / うち表に出ている馬 {}頭)".format(int(inr.sum()), int((inr & seen).sum())))
for b in BADGES + ["core"]:
    x = val("oi_" + b, 0) == 1
    mm = inr & x
    if mm.sum() >= 10:
        print("  {:10s} {:5d}頭  3着内 実際 {:5.1%} / うちの見込み {:5.1%} / 市場の見込み(1着の確率×2.6) {:5.1%}".format(
            b, int(mm.sum()), y[mm].mean(), p3[mm].mean(), min(1, 2.6 * q[mm].mean())))
rest = inr & seen & (val("oi_core", 0) == 0)
print("  {:10s} {:5d}頭  3着内 実際 {:5.1%} / うちの見込み {:5.1%}".format("注目でない", int(rest.sum()), y[rest].mean() if rest.sum() else 0, p3[rest].mean() if rest.sum() else 0))
cols = [val("oi_" + b, 0) for b in ("BEST+加速", "BEST更新", "時計良化", "パターン変更", "型注意", "型好転", "連闘注意")]
for base, bl in (([lg(p3)], "うちの確率"), ([lg(q)], "市場"), ([lg(p3), lg(q)], "うち+市場")):
    l0, _ = oof_ll(base, inr); l1, w = oof_ll(base + cols, inr)
    print("  {:10s} 対数尤度 {:.5f} → +追いきり {:.5f}(差 {:+.5f}) 係数 {}".format(bl, l0, l1, l1 - l0, " ".join("%.2f" % v for v in w[1:])))

# 全部まとめて(どれかに載っているレース)
allr = np.isin([r["rid"] for r in rows], list(set().union(*info.values())))
X = [lg(p3)]
for name in ("nige", "sashi", "kime"):
    x = val(name); X += [np.where(np.isnan(x), 0, np.log1p(np.nan_to_num(x))), (~np.isnan(x)).astype(float)]
X += cols
for base, bl in (([lg(p3)], "うちの確率"), ([lg(q)], "市場")):
    l0, _ = oof_ll(base, allr); l1, w = oof_ll(base + X[1:], allr)
    print("\n■ 全部({}頭) {:8s} 対数尤度 {:.5f} → +全部 {:.5f}(差 {:+.5f})".format(int(allr.sum()), bl, l0, l1, l1 - l0))
print("\n(このサイトのデータは保存していない)")

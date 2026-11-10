"""モデルの3着内の確率に、市場(単勝・ワイドの確定オッズ)がまだ値段に入れていない情報があるか(中央 2024〜2026年)
  python tools/experiments/beyond_market.py
  3着内に来たか を、ワイドのオッズから出した強さ・単勝オッズから出した強さ・モデルの確率 のロジスティック回帰で当てる。
  2024・2025年で係数を決め、2026年の対数尤度で比べる(年を入れ替えても出す)。モデルを足して良くなれば、モデルには市場に無い情報がある。
  良くなるなら、どんな馬で市場とずれるか(モデルが市場より高く見る馬・低く見る馬の実際)も出す"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import wide_info as WI
import keiba_final as F
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def rows():
    out = []
    for rc in WI.load():
        us = [u for u in rc["w"] if u in rc["od"] and u in rc["pm"]]
        if len(us) < 5:
            continue
        q = np.array([1 / rc["od"][u] for u in us]); q = q / q.sum()
        top3 = set(rc["order"][:3])
        for u, qq in zip(us, q):
            out.append((rc["y"], rc["rid"], u, int(u in top3), rc["w"][u] if rc["w"][u] < .98 else .98, qq, rc["pm"][u], len(us),
                        rc["nk"].get(u, 99), int(u == rc["order"][0])))
    return out


def ll(y, p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


if __name__ == "__main__":
    D = rows()
    Y = np.array([r[0] for r in D]); y = np.array([r[3] for r in D], dtype=float)
    pw, q, pm, n = (np.array([r[i] for r in D], dtype=float) for i in (4, 5, 6, 7))
    X_all = {"ワイド": lg(pw), "単勝": lg(np.clip(q * 3, 0, .98)), "モデル": lg(pm), "頭数": np.log(n)}
    print("馬 {}頭({}レース)".format(len(D), len({r[1] for r in D})))
    sets = (("単勝だけ", ("単勝", "頭数")), ("ワイドだけ", ("ワイド", "頭数")), ("単勝+ワイド", ("単勝", "ワイド", "頭数")),
            ("モデルだけ", ("モデル", "頭数")), ("単勝+モデル", ("単勝", "モデル", "頭数")), ("ワイド+モデル", ("ワイド", "モデル", "頭数")),
            ("単勝+ワイド+モデル", ("単勝", "ワイド", "モデル", "頭数")))
    for test in ("2026", "2025", "2024"):
        tr, te = Y != test, Y == test
        print("\n■ {}年を確かめる(ほかの2年で係数を決める)  対数尤度 / 係数".format(test))
        for name, ks in sets:
            X = np.column_stack([X_all[k] for k in ks])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            w = F.fit_logit(((X[tr] - mu) / sd).tolist(), y[tr].tolist(), 0.01)
            w = np.array(w)
            z = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ w
            print("  {:14s} {:.5f}  {}".format(name, ll(y[te], 1 / (1 + np.exp(-z))), " ".join("{} {:+.3f}".format(k, c / s) for k, c, s in zip(ks, w[1:], sd))))

    # モデルが市場(ワイド)より高く見る馬・低く見る馬
    print("\n■ モデルの確率 ÷ ワイドから出した確率 で分ける: 実際に3着内に来た割合(ワイドの見込み / モデルの見込み)")
    ratio = lg(pm) - lg(pw)
    qs = np.quantile(ratio, [0, .05, .2, .4, .6, .8, .95, 1])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (ratio >= a) & (ratio <= b)
        print("  モデル−ワイド {:+.2f}〜{:+.2f}  {:6d}頭  実際 {:5.1%}  ワイド {:5.1%}  モデル {:5.1%}   年ごとの実際/ワイド [{}]".format(
            a, b, m.sum(), y[m].mean(), pw[m].mean(), pm[m].mean(),
            " ".join("{:.2f}".format(y[m & (Y == t)].mean() / pw[m & (Y == t)].mean()) for t in ("2024", "2025", "2026"))))
    print("\n■ ワイドの見込みの帯ごとに、モデルが高く見る馬(上1/4)・低く見る馬(下1/4)の実際")
    for a, b in ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, .7), (.7, 1)):
        m = (pw >= a) & (pw < b)
        r_ = ratio[m]; hi, lo = np.quantile(r_, .75), np.quantile(r_, .25)
        mh, ml = m & (ratio >= hi), m & (ratio <= lo)
        print("  ワイド {:3.0%}〜{:3.0%}  高く見る {:5.1%}(ワイド {:5.1%}・{}頭)  低く見る {:5.1%}(ワイド {:5.1%}・{}頭)".format(
            a, b, y[mh].mean(), pw[mh].mean(), mh.sum(), y[ml].mean(), pw[ml].mean(), ml.sum()))

"""軸不可(◎○にしない馬)の条件を1つずつ見直す。ワイド◎-○の回収率を年ごとに(中央、2024年1月〜、月ごとの確率)
  python tools/experiments/axis_review.py [--past]
  予想に残っている札(距離延長の幅・距離実績の有無は点 -1.5/-3・休み明けの日数)から、条件を変えたときの軸不可を作り直す。
  --past … 保存してある馬柱から前走との距離差を出し、+100m・+150m の延長も軸不可にしたときを比べる(札は +200m 以上だけなので)"""
import glob, gzip, json, os, re, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
wf = json.load(gzip.open(R + "/keiba/model_wf.json.gz"))["probs"]
races = []
for f in sorted(glob.glob(R + "/keiba/days/202[456]*.json")):
    d = json.load(open(f))
    for rid, r in (d.get("races") or {}).items():
        res = r.get("result") or {}
        pay = (res.get("payout") or {}).get("ワイド")
        if r.get("center") != "jra" or rid not in wf or not pay or len(res.get("order") or []) < 3:
            continue
        info = {}
        for h in ((r.get("pred") or {}).get("horses") or []):
            x = {"ng": set(h.get("axis_ng") or []), "rest": h.get("rest_days"), "ext": 0, "proven": False, "first": None}
            for m in h.get("minus", []):
                t = m["t"]
                if t.startswith("距離延長"):
                    x["ext"] = int(re.search(r"\+(\d+)", t).group(1))
                    x["proven"] = m["p"] > -2
                elif t in ("初芝", "初ダート"):
                    x["first"] = t
            info[str(h["umaban"])] = x
        races.append((os.path.basename(f)[:4], wf[rid], info, pay, rid))


def now_rule(x):
    return bool(x["ng"])


def make(no_debut=True, first=("初芝", "初ダート"), ext=200, ext_proven=True, rest=180):
    """条件の組み合わせから軸不可の判定を作る。ext=None なら延長は見ない / ext_proven=False なら距離実績のある延長は軸にしてよい"""
    def ng(x):
        if no_debut and "能力不明" in x["ng"]:
            return True
        if x["first"] in first:
            return True
        if ext is not None and x["ext"] >= ext and (ext_proven or not x["proven"]):
            return True
        if rest is not None and (x["rest"] or 0) >= rest:
            return True
        return False
    return ng


EMPTY = {"ng": set(), "rest": None, "ext": 0, "proven": False, "first": None}


def run(ng, t1=0.0, who="both"):
    """who … both=◎○とも軸不可を外す / hon=◎だけ外す(○は確率どおり) / tai=○だけ外す(◎は確率どおり)"""
    out = []
    for y in ("2024", "2025", "2026"):
        n = ret = hit = 0
        for yy, pr, info, pay, _ in races:
            if yy != y:
                continue
            hs = sorted(pr.items(), key=lambda kv: -kv[1])
            ok = [kv for kv in hs if not ng(info.get(kv[0]) or EMPTY)]
            if len(hs) < 5 or len(ok) < 2:
                continue
            if who == "both":
                h, t = ok[0], ok[1]
            elif who == "hon":
                h = ok[0]
                t = next(kv for kv in hs if kv[0] != h[0])
            else:
                h = hs[0]
                t = next((kv for kv in ok if kv[0] != h[0]), None)
                if not t:
                    continue
            if h[1] < t1:
                continue
            v = pay.get("-".join(sorted((h[0], t[0]), key=int)), 0)
            n += 1; ret += v; hit += v > 0
        out.append((n, ret, hit))
    tn, tr, th = (sum(x[i] for x in out) for i in range(3))
    return " ".join("%4.0f%%" % (r / max(n, 1)) for n, r, _ in out) + " | %5dR 的中%3.0f%% 計%4.0f%%" % (tn, th / max(tn, 1) * 100, tr / max(tn, 1))


if __name__ == "__main__":
    base = make()
    chk = sum(1 for _, pr, info, _, _ in races for u, x in info.items() if now_rule(x) != base(x))
    print("今の決まりと作り直した決まりが違う馬:", chk, "頭(0なら作り直しは正しい)")
    variants = [("今の決まり", base), ("軸不可なし", lambda x: False),
                ("- 能力不明を外す", make(no_debut=False)),
                ("- 初芝/初ダートを外す", make(first=())), ("- 初芝だけ外す", make(first=("初ダート",))),
                ("- 初ダートだけ外す", make(first=("初芝",))),
                ("- 距離延長を外す", make(ext=None)), ("  延長 +400m以上だけ", make(ext=400)),
                ("  延長 +600m以上だけ", make(ext=600)), ("  距離実績のある延長はよい", make(ext_proven=False)),
                ("- 長期休み明けを外す", make(rest=None)), ("  休み 90日以上", make(rest=90)), ("  休み 120日以上", make(rest=120)),
                ("  休み 240日以上", make(rest=240)), ("  休み 365日以上", make(rest=365))]
    for label, t1 in (("全レース", 0.0), ("今の基準 ◎80%以上", 0.8)):
        print("\n■ ワイド◎-○ " + label + "   2024  2025  2026 | 計")
        for name, ng in variants:
            print("  {:22s} {}".format(name, run(ng, t1)))
        print("  {:22s} {}".format("◎だけに効かせる", run(base, t1, "hon")))
        print("  {:22s} {}".format("○だけに効かせる", run(base, t1, "tai")))
    if "--past" in sys.argv:
        sys.path.insert(0, R)
        import keiba as K
        dd = {}
        for f in sorted(glob.glob(K.OUT + "/days/202[456]*.json")):
            day = K.load_day(os.path.basename(f)[:8])
            for rid, rc in day["races"].items():
                for h in rc.get("horses") or []:
                    past = [x for x in (h.get("past") or []) if x.get("surface") != "障"]
                    if rc.get("center") == "jra" and past and past[0].get("dist") and rc.get("dist"):
                        dd[(rid, str(h["umaban"]))] = rc["dist"] - past[0]["dist"]
        for _, _, info, _, rid in races:
            for u, x in info.items():
                x["dd"] = dd.get((rid, u), 0)
        for lim in (100, 150):
            ng = lambda x, lim=lim: base(x) or x.get("dd", 0) >= lim
            for label, t1 in (("全レース", 0.0), ("◎80%以上", 0.8)):
                print("延長 +{}m以上も軸不可 {:8s} {}".format(lim, label, run(ng, t1)))

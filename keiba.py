#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keiba.py -- 中央・地方の当日の全レースから「買うレース」だけを選び、買い目を出す

■ 取得(netkeiba)
  レース一覧 → 出馬表(枠・馬番・斤量・馬体重・馬場) → 馬柱(過去5走) → 単勝オッズ(人気)
  → 結果(着順・払戻)
■ 予想
  keiba_logic.py(規則 1〜30)。能力はオッズを見ずに作り、人気は最後に照らすだけ。
  発走を過ぎたレースの予想は書き換えない(成績をごまかさない)。
■ 出力
  keiba/data.json           … いまの日(アプリが読む。days/ の最新と同じ中身)
  keiba/days/YYYYMMDD.json  … 日別の予想と結果
  keiba/history.json        … 収支・条件別の回収率・負けパターン(規則 30)

使い方:
  python keiba.py                 # 本日(JST)。前日の結果が欠けていれば埋める
  python keiba.py --date 20261018
  python keiba.py --history-only  # days/ から history.json だけ作り直す
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

import keiba_logic as L

JST = timezone(timedelta(hours=9))
OUT = "keiba"
SITE = {"jra": "https://race.netkeiba.com", "nar": "https://nar.netkeiba.com"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "ja"}
WAIT = 0.4                 # 1リクエストごとの間隔(秒)
CARD_REFRESH = 50          # 出馬表(馬場・取消)を取り直す間隔(分)
ODDS_REFRESH = 20          # オッズを取り直す間隔(分)
RESULT_AFTER = 8           # 発走から何分後に結果を見に行くか

SESS = requests.Session()
SESS.headers.update(UA)


# ---------------------------------------------------------------- 取得
def get(url, params=None, as_json=False):
    for attempt in (1, 2, 3):
        try:
            r = SESS.get(url, params=params, timeout=30)
            time.sleep(WAIT)
            if r.status_code == 200:
                if as_json:
                    return r.json()
                return decode(r.content)
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * attempt)
    return None


def decode(b):
    m = re.search(rb'charset=["\']?([A-Za-z0-9_\-]+)', b[:3000])
    if not m:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            pass
    enc = m.group(1).decode() if m else "euc-jp"
    if enc.lower() in ("euc-jp", "eucjp", "x-euc-jp"):
        enc = "euc_jis_2004"
    try:
        return b.decode(enc, errors="replace")
    except LookupError:
        return b.decode("utf-8", errors="replace")


def center_of(rid):
    return "nar" if int(rid[4:6]) >= 30 else "jra"


def venue_of(rid):
    code = rid[4:6]
    return (L.NAR_CODE if center_of(rid) == "nar" else L.JRA_CODE).get(code)


def race_ids(d):
    """その日の中央・地方のレースID"""
    ids = set()
    t = get(SITE["jra"] + "/top/race_list_sub.html", {"kaisai_date": d}) or ""
    ids |= set(re.findall(r"race_id=(\d{12})", t))
    # 地方は race_list_sub.html にレースIDと開催ID(kaisai_id)が載る(race_list.html には無い)
    t = get(SITE["nar"] + "/top/race_list_sub.html", {"kaisai_date": d}) or ""
    ids |= set(re.findall(r"race_id=(\d{12})", t))
    for kid in sorted(set(re.findall(r"kaisai_id=(\d{10})", t))):
        s = get(SITE["nar"] + "/top/race_list_sub.html",
                {"kaisai_id": kid, "kaisai_date": d}) or ""
        ids |= set(re.findall(r"race_id=(\d{12})", s))
    # 地方のIDは日付(MMDD)を含む。別の日が混ざらないようにする
    out = []
    for rid in ids:
        if not venue_of(rid):
            continue
        if center_of(rid) == "nar" and (rid[:4] != d[:4] or rid[6:10] != d[4:8]):
            continue
        if center_of(rid) == "jra" and rid[:4] != d[:4]:
            continue
        out.append(rid)
    return sorted(out)


# ---------------------------------------------------------------- パース
def txt(el):
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)) if el else ""


def num(s, typ=int):
    try:
        return typ(s)
    except (TypeError, ValueError):
        return None


def parse_header(soup, rid):
    center = center_of(rid)
    name = txt(soup.select_one(".RaceName"))
    d1 = L.nfkc(txt(soup.select_one(".RaceData01")))
    d2 = L.nfkc(txt(soup.select_one(".RaceData02")))
    race = {"id": rid, "center": center, "venue": venue_of(rid), "rno": int(rid[-2:]),
            "name": name, "cond": d2}
    m = re.search(r"(\d{1,2}:\d{2})\s*発走", d1)
    race["start"] = m.group(1).zfill(5) if m else None
    m = re.search(r"(芝|ダ|障)\D{0,4}?(\d{3,4})\s*m", d1)
    race["surface"], race["dist"] = (m.group(1), int(m.group(2))) if m else (None, None)
    m = re.search(r"\((右|左|直)", d1)
    vi = L.VENUES.get(race["venue"])
    race["direction"] = m.group(1) if m else (vi[1] if vi else None)
    race["slope"] = vi[2] if vi else None
    m = re.search(r"馬場\s*:\s*(良|稍重|稍|重|不良|不)", d1)
    race["baba"] = m.group(1)[0] if m else None
    grade = None
    for g, lv in (("Icon_GradeType1", 8), ("Icon_GradeType2", 7), ("Icon_GradeType3", 6)):
        if soup.select_one(".RaceName ." + g) or soup.select_one(".RaceName." + g):
            grade = lv
    race["cls"] = grade or L.class_level(name + " " + d2, center)
    return race


def parse_card(html, rid):
    soup = BeautifulSoup(html, "html.parser")
    race = parse_header(soup, rid)
    horses = []
    for tr in soup.select("tr.HorseList"):
        tds = tr.find_all("td")
        h = {}
        for td in tds:
            cl = " ".join(td.get("class") or [])
            t = txt(td)
            if re.search(r"\bWaku\d", cl) and "waku" not in h:
                h["waku"] = num(t)
            elif "Umaban" in cl:
                h["umaban"] = num(t)
            elif "Barei" in cl:
                h["sex_age"] = t
            elif "Jockey" in cl:
                h["jockey"] = t
            elif "Trainer" in cl:
                h["trainer"] = t
            elif "Weight" in cl:
                m = re.search(r"(\d{3})\s*\(([+-]?\d+)\)", t)
                if m:
                    h["weight"], h["wdiff"] = int(m.group(1)), int(m.group(2))
        # 斤量は性齢の次の列
        for i, td in enumerate(tds):
            if "Barei" in " ".join(td.get("class") or []) and i + 1 < len(tds):
                h["kinryo"] = num(txt(tds[i + 1]), float)
        a = tr.select_one(".HorseName a") or tr.select_one("a[href*='/horse/']")
        h["name"] = txt(a) or txt(tr.select_one(".HorseName"))
        m = re.search(r"/horse/(\w+)", a.get("href", "")) if a else None
        h["horse_id"] = m.group(1) if m else None
        rowtxt = txt(tr)
        h["scratched"] = bool(re.search(r"取消|除外|Cancel", rowtxt + " ".join(tr.get("class") or [])))
        if h.get("umaban"):
            horses.append(h)
    race["horses"] = horses
    race["n"] = sum(1 for h in horses if not h["scratched"])
    return race


def parse_past_cell(td):
    t = L.nfkc(txt(td))
    if not re.search(r"\d{4}\.\d{1,2}\.\d{1,2}", t):
        return None
    r = {}
    m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\s*([^\s\d]+)?", t)
    r["date"] = "{}-{:02d}-{:02d}".format(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    r["venue"] = L.venue_info(t[m.end(3):m.end(3) + 8])[0] or (m.group(4) or "")
    numel = td.select_one(".Num")
    rk = txt(numel) if numel else ""
    r["rank"] = num(L.nfkc(rk)) if rk else None
    r["race_name"] = txt(td.select_one(".Data02"))
    d5 = L.nfkc(txt(td.select_one(".Data05"))) or t
    m = re.search(r"(芝|ダ|障)\D{0,2}?(\d{3,4})", d5)
    if m:
        r["surface"], r["dist"] = m.group(1), int(m.group(2))
    m = re.search(r"(良|稍重|稍|重|不良|不)(?!\S*頭)", d5)
    r["baba"] = m.group(1)[0] if m else None
    d3 = L.nfkc(txt(td.select_one(".Data03"))) or t
    m = re.search(r"(\d+)頭", d3)
    r["n"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+)番", d3)
    r["umaban"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+)人", d3)
    r["ninki"] = int(m.group(1)) if m else None
    ks = re.findall(r"(\d{2}\.\d)", d3)
    r["kinryo"] = float(ks[-1]) if ks else None
    d6 = L.nfkc(txt(td.select_one(".Data06"))) or t
    m = re.search(r"(\d{1,2}(?:-\d{1,2})+)", d6)
    r["pass"] = [int(x) for x in m.group(1).split("-")] if m else []
    m = re.search(r"\((\d{2}\.\d)\)", d6)
    r["last3f"] = float(m.group(1)) if m else None
    d7 = L.nfkc(txt(td.select_one(".Data07")))
    m = re.search(r"\((-?\d+\.\d)\)", d7)
    r["margin"] = float(m.group(1)) if m else None
    if r["rank"] is None:
        m = re.search(r"(\d{1,2})着", t)
        r["rank"] = int(m.group(1)) if m else None
    r["cls"] = L.class_level(r["race_name"], (L.VENUES.get(r["venue"]) or ("jra",))[0])
    return r


def parse_past(html):
    """馬番 -> 過去走(新しい順)"""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for tr in soup.select("tr.HorseList"):
        # 追い切りの行(OikiriDataHead*)にも馬番があるが、過去走は無い
        if any(c.startswith("Oikiri") for c in tr.get("class") or []):
            continue
        um = None
        for td in tr.find_all("td"):
            cl = td.get("class") or []
            # 馬番の列は "Umaban" か、数字の付かない "Waku"(枠の列は "Waku1" など)
            if any(c.startswith("Umaban") or c == "Waku" for c in cl):
                um = num(txt(td))
                break
        if um is None:
            ints = [num(txt(td)) for td in tr.find_all("td")[:3]]
            ints = [x for x in ints if x]
            um = ints[1] if len(ints) >= 2 else (ints[0] if ints else None)
        runs = []
        for td in tr.find_all("td"):
            cl = " ".join(td.get("class") or [])
            if "Past" in cl or "Rest" in cl:
                r = parse_past_cell(td)
                if r:
                    runs.append(r)
        if um and (runs or um not in out):
            out[um] = runs
    return out


def fetch_odds(rid):
    """馬番 -> (単勝オッズ, 人気)"""
    c = center_of(rid)
    api = "/api/api_get_jra_odds.php" if c == "jra" else "/api/api_get_nar_odds.php"
    j = get(SITE[c] + api, {"pid": api.split("/")[-1][:-4], "input": "UTF-8", "output": "json",
                            "race_id": rid, "type": "1", "action": "update"}, as_json=True)
    out = {}
    try:
        odds = j["data"]["odds"]["1"]
        for k, v in odds.items():
            o, nk = num(v[0], float), num(v[2])
            if o:
                out[int(k)] = (o, nk)
    except (TypeError, KeyError, IndexError, AttributeError):
        pass
    if out and not any(nk for _, nk in out.values()):
        ranked = sorted(out, key=lambda u: out[u][0])
        out = {u: (out[u][0], i + 1) for i, u in enumerate(ranked)}
    return out


PAY = {"Tansho": ("単勝", 1), "Umaren": ("馬連", 2), "Wide": ("ワイド", 2),
       "Fuku3": ("3連複", 3), "Tan3": ("3連単", 3)}


def parse_result(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table.RaceTable01 tr.HorseList, #All_Result_Table tr.HorseList"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        rk, um = num(L.nfkc(txt(tds[0]))), num(txt(tds[2]))
        if rk and um:
            rows.append((rk, um))
    order = [u for _, u in sorted(set(rows))]
    pay = {}
    for tr in soup.select("table.Payout_Detail_Table tr"):
        cl = " ".join(tr.get("class") or [])
        key = next((k for k in PAY if k in cl.split()), None)
        if not key:
            continue
        typ, k = PAY[key]
        res = tr.select_one("td.Result")
        nums = [int(x) for x in (txt(s) for s in res.select("span")) if x.isdigit()] if res else []
        if not nums and res:
            nums = [int(x) for x in re.findall(r"\d+", txt(res))]
        yen = [int(x.replace(",", "")) for x in re.findall(r"([\d,]+)円", txt(tr.select_one("td.Payout")))]
        combos = [nums[i:i + k] for i in range(0, len(nums) - k + 1, k)]
        d = {}
        for c, y in zip(combos, yen):
            c = c if typ == "3連単" else sorted(c)
            d["-".join(str(x) for x in c)] = y
        pay[typ] = d
    return {"order": order, "payout": pay}


# ---------------------------------------------------------------- 1日分
def load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def start_dt(d, hhmm):
    if not hhmm:
        return None
    return datetime.strptime(d + hhmm, "%Y%m%d%H:%M").replace(tzinfo=JST)


def minutes_since(iso, now):
    if not iso:
        return 1e9
    return (now - datetime.fromisoformat(iso)).total_seconds() / 60


def run_day(d, now, fetch_cards=True):
    path = "{}/days/{}.json".format(OUT, d)
    day = load(path, {"date": d, "races": {}})
    races = day["races"]
    if fetch_cards and (not races or minutes_since(day.get("list_at"), now) > 180):
        ids = race_ids(d)
        print("{} レース一覧 {}件".format(d, len(ids)))
        if ids:
            day["list_at"] = now.isoformat(timespec="seconds")
        for rid in ids:
            races.setdefault(rid, {"id": rid})
    today = datetime.strptime(d, "%Y%m%d").date()
    for rid in sorted(races):
        rc = races[rid]
        st = start_dt(d, rc.get("start"))
        started = st is not None and now >= st
        # 出馬表(馬場・取消は朝から変わるので発走前は取り直す)
        if fetch_cards and not started and (not rc.get("horses") or
                                            minutes_since(rc.get("card_at"), now) > CARD_REFRESH):
            html = get(SITE[center_of(rid)] + "/race/shutuba.html", {"race_id": rid})
            if html:
                new = parse_card(html, rid)
                if new["horses"]:
                    keep = {h["umaban"]: h for h in rc.get("horses") or []}
                    for h in new["horses"]:
                        old = keep.get(h["umaban"]) or {}
                        for k in ("past", "odds", "ninki"):
                            if k in old:
                                h[k] = old[k]
                    rc.update(new)
                    rc["card_at"] = now.isoformat(timespec="seconds")
        if not rc.get("horses"):
            continue
        # 馬柱(過去5走)は1回だけ
        if fetch_cards and not rc.get("past_at") and not started:
            html = get(SITE[center_of(rid)] + "/race/shutuba_past.html", {"race_id": rid})
            if html:
                past = parse_past(html)
                for h in rc["horses"]:
                    h["past"] = past.get(h["umaban"], [])
                if past:
                    rc["past_at"] = now.isoformat(timespec="seconds")
        # オッズ(人気)。予想には使わず、最後に照らすだけ
        if fetch_cards and not started and minutes_since(rc.get("odds_at"), now) > ODDS_REFRESH:
            od = fetch_odds(rid)
            if od:
                for h in rc["horses"]:
                    if h["umaban"] in od:
                        h["odds"], h["ninki"] = od[h["umaban"]]
                rc["odds_at"] = now.isoformat(timespec="seconds")
        # 予想。発走後は固定
        if not started or not rc.get("pred"):
            if rc.get("past_at") or not started:
                rc["pred"] = L.predict(rc, today)
                rc["pred_at"] = now.isoformat(timespec="seconds")
        # 結果
        if st and now >= st + timedelta(minutes=RESULT_AFTER) and not rc.get("result"):
            html = get(SITE[center_of(rid)] + "/race/result.html", {"race_id": rid})
            res = parse_result(html) if html else None
            if res and len(res["order"]) >= 3:
                rc["result"] = res
        if rc.get("result") and rc.get("pred"):
            rc["settle"] = L.settle(rc["pred"], rc["result"])
    day["updated"] = now.isoformat(timespec="seconds")
    save(path, day)
    return day


# ---------------------------------------------------------------- 収支と分析(規則 30)
def dist_band(dd):
    if not dd:
        return "?"
    return "〜1400m" if dd <= 1400 else ("1500〜1800m" if dd <= 1800 else "1900m〜")


def field_band(n):
    return "〜10頭" if (n or 0) <= 10 else ("11〜14頭" if n <= 14 else "15頭〜")


def gap_band(g):
    return "差6〜" if g >= 6 else ("差4.5〜6" if g >= 4.5 else "差3.5〜4.5")


def build_history():
    files = sorted(glob.glob(OUT + "/days/*.json"))
    days, bets_list = [], []
    groups = {}
    by_type = {}
    marks = {}
    tags = {}
    base = [0, 0]

    def add(g, key, s):
        x = groups.setdefault(g, {}).setdefault(key, {"n": 0, "hit": 0, "stake": 0, "ret": 0})
        x["n"] += 1
        x["hit"] += 1 if s["hit"] else 0
        x["stake"] += s["stake"]
        x["ret"] += s["ret"]

    for f in files:
        day = load(f, None)
        if not day:
            continue
        d = day["date"]
        row = {"date": d, "races": 0, "buy": 0, "settled": 0, "stake": 0, "ret": 0, "hit": 0}
        for rid, rc in sorted(day.get("races", {}).items()):
            p = rc.get("pred")
            if not p:
                continue
            row["races"] += 1
            res = rc.get("result")
            order = (res or {}).get("order") or []
            # 印と消しの成績(全レース。見送りも含めて精度を見る)
            if len(order) >= 3:
                for h in p.get("horses", []):
                    rk = L.rank_of(order, h["umaban"])
                    if rk is None:
                        continue
                    mk = h.get("mark") or "無"
                    m = marks.setdefault(mk, {"n": 0, "win": 0, "top3": 0})
                    m["n"] += 1
                    m["win"] += rk == 1
                    m["top3"] += rk <= 3
                    base[0] += 1
                    base[1] += rk <= 3
                    for t in {re.sub(r"\(.*?\)", "", x["t"]) for x in h["minus"] if x["p"] <= -2}:
                        tg = tags.setdefault(t, [0, 0])
                        tg[0] += 1
                        tg[1] += rk <= 3
            if p.get("verdict") == "見送り":
                continue
            row["buy"] += 1
            s = rc.get("settle")
            if not s:
                continue
            row["settled"] += 1
            row["stake"] += s["stake"]
            row["ret"] += s["ret"]
            row["hit"] += 1 if s["hit"] else 0
            add("中央/地方", "中央" if rc["center"] == "jra" else "地方", s)
            add("判定", p["verdict"], s)
            add("芝/ダート", {"芝": "芝", "ダ": "ダート"}.get(rc.get("surface"), "?"), s)
            add("距離", dist_band(rc.get("dist")), s)
            add("頭数", field_band(rc.get("n")), s)
            add("馬場", "道悪" if (rc.get("baba") or "良") in L.HEAVY else "良・稍", s)
            add("2強の差", gap_band(p.get("gap") or 0), s)
            add("展開", (p.get("pace") or {}).get("pace", "?"), s)
            add("競馬場", rc.get("venue") or "?", s)
            for b in p.get("bets", []):
                k = b["type"] + ("(◎飛び保険)" if "保険" in b["role"] and "飛" in b["role"] else "")
                x = by_type.setdefault(k, {"n": 0, "hit": 0, "stake": 0, "ret": 0})
                x["n"] += 1
                x["hit"] += 1 if b.get("hit") else 0
                x["stake"] += len(b["combos"]) * b["unit"]
                x["ret"] += b.get("ret", 0)
            bets_list.append({"date": d, "id": rid, "venue": rc.get("venue"), "rno": rc.get("rno"),
                              "verdict": p["verdict"], "stake": s["stake"], "ret": s["ret"],
                              "hit": s["hit"], "hr": s.get("honmei_rank"),
                              "tr": s.get("taikou_rank")})
        days.append(row)
    tag_rows = [{"tag": t, "n": v[0], "top3": v[1]} for t, v in tags.items() if v[0] >= 1]
    tag_rows.sort(key=lambda x: -x["n"])
    hist = {"updated": datetime.now(JST).isoformat(timespec="seconds"), "unit": L.UNIT,
            "days": days, "groups": groups, "by_type": by_type, "marks": marks,
            "tags": tag_rows, "base_top3": base, "bets": bets_list[-400:]}
    save(OUT + "/history.json", hist)
    return hist


# ---------------------------------------------------------------- 診断(取得がおかしいとき)
def diag(d):
    """netkeiba の応答の形をログに出す。何も保存しない。"""
    def raw(url, params=None):
        try:
            r = SESS.get(url, params=params, timeout=30)
            time.sleep(WAIT)
            return r.status_code, r.content
        except requests.RequestException as e:
            return type(e).__name__, b""

    def rows(html, sel="tr.HorseList", k=4):
        soup = BeautifulSoup(html, "html.parser")
        trs = soup.select(sel)
        print("   {} 行 / table {}".format(len(trs), [(" ".join(t.get("class") or []), len(t.select("tr")))
                                                    for t in soup.select("table")][:6]))
        for tr in trs[:k] + trs[-1:]:
            print("   tr.class={}".format(tr.get("class")))
            for td in tr.find_all("td")[:7]:
                print("     td.class={} | {}".format(td.get("class"), txt(td)[:70]))

    print("==== 地方の一覧", d)
    for path, prm in (("/top/race_list.html", {"kaisai_date": d}),
                      ("/top/race_list_sub.html", {"kaisai_date": d})):
        st, b = raw(SITE["nar"] + path, prm)
        t = decode(b)
        print(" {} {} {}B race_id={} kaisai_id={}".format(path, st, len(b),
              len(set(re.findall(r"race_id=(\d{12})", t))), sorted(set(re.findall(r"kaisai_id=(\d+)", t)))[:8]))
        print("  links:", sorted(set(re.findall(r'href="([^"]*race[^"]*)"', t)))[:12])
    print("==== 中央の一覧", d)
    ids = race_ids(d)
    print(" ", len(ids), ids[:3], ids[-3:])
    jra = [x for x in ids if center_of(x) == "jra"]
    nar = [x for x in ids if center_of(x) == "nar"]
    nar = [x for x in nar if venue_of(x) != "帯広"]
    for rid in (jra[:1] + nar[:1]):
        c = center_of(rid)
        print("==== 馬柱", rid)
        st, b = raw(SITE[c] + "/race/shutuba_past.html", {"race_id": rid})
        print("  status", st, len(b))
        rows(decode(b))
        print("==== オッズ", rid)
        api = "/api/api_get_jra_odds.php" if c == "jra" else "/api/api_get_nar_odds.php"
        for prm in ({"pid": api.split("/")[-1][:-4], "input": "UTF-8", "output": "json",
                     "race_id": rid, "type": "1", "action": "update"},
                    {"pid": api.split("/")[-1][:-4], "input": "UTF-8", "output": "json",
                     "race_id": rid, "type": "1", "action": "init", "compress": "0"}):
            st, b = raw(SITE[c] + api, prm)
            print("  {} {} {}B {}".format(prm["action"], st, len(b), decode(b)[:300].replace("\n", " ")))
        for path in ("/odds/index.html", "/race/shutuba.html"):
            st, b = raw(SITE[c] + path, {"type": "b1", "race_id": rid})
            t = decode(b)
            print("  {} {} {}B".format(path, st, len(b)))
            print("   api:", sorted(set(re.findall(r'[\w/.:-]*(?:api|odds_get|jsonp?)[\w/.:-]*\.(?:php|html|json)', t)))[:15])
            for m in list(re.finditer(r"(api_get_[a-z_]+|odds_get_form|getOdds|OddsData)", t))[:4]:
                print("   ctx:", t[max(0, m.start() - 160):m.end() + 160].replace("\n", " "))
            soup = BeautifulSoup(t, "html.parser")
            for el in soup.select("[id^=odds-], td.Popular, td.Odds, span.Odds")[:4]:
                print("   cell:", el.name, el.get("id"), el.get("class"), txt(el)[:40])
        for path in ("/odds/odds_get_form.html",):
            st, b = raw(SITE[c] + path, {"type": "b1", "race_id": rid})
            soup = BeautifulSoup(decode(b), "html.parser")
            print("  {} {} {}B".format(path, st, len(b)))
            for tr in soup.select("tr")[:3]:
                print("   tr:", [(td.get("class"), txt(td)[:20]) for td in tr.find_all("td")][:8])
    # 前日(過去のレース)が取れるか
    y = (datetime.strptime(d, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    yid = race_ids(y)
    print("==== 前日", y, len(yid), "レース")
    for rid in [x for x in yid if center_of(x) == "jra"][-1:] + [x for x in yid if center_of(x) == "nar"][-1:]:
        c = center_of(rid)
        print("==== 結果", rid)
        st, b = raw(SITE[c] + "/race/result.html", {"race_id": rid})
        html = decode(b)
        print("  status", st, len(b))
        rows(html, "table.RaceTable01 tr.HorseList, #All_Result_Table tr.HorseList", 2)
        tr0 = BeautifulSoup(html, "html.parser").select_one("table.RaceTable01 tr.HorseList")
        if tr0:
            print("   全列:", [(td.get("class"), txt(td)[:16]) for td in tr0.find_all("td")])
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.select("table.Payout_Detail_Table tr")[:9]:
            print("   pay", tr.get("class"), "|", txt(tr)[:90])
        print("  parse_result:", json.dumps(parse_result(html), ensure_ascii=False)[:400])
        st, b = raw(SITE[c] + "/race/shutuba.html", {"race_id": rid})
        card = parse_card(decode(b), rid)
        print("  過去レースの出馬表:", st, len(card["horses"]), "頭", card.get("start"), card.get("baba"))
        st, b = raw(SITE[c] + "/race/shutuba_past.html", {"race_id": rid})
        past = parse_past(decode(b))
        print("  過去レースの馬柱:", st, len(past), "頭", {u: [r["date"] for r in v[:2]] for u, v in list(past.items())[:4]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--history-only", action="store_true")
    ap.add_argument("--diag", action="store_true", help="取得の診断をログに出す(保存しない)")
    a = ap.parse_args()
    if a.diag:
        diag(a.date or datetime.now(JST).strftime("%Y%m%d"))
        return
    now = datetime.now(JST)
    if not a.history_only:
        d = a.date or now.strftime("%Y%m%d")
        # 前日の夜(地方のナイター)の結果が欠けていれば先に埋める
        y = (datetime.strptime(d, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        if os.path.exists("{}/days/{}.json".format(OUT, y)):
            run_day(y, now, fetch_cards=False)
        day = run_day(d, now)
        save(OUT + "/data.json", day)
        n = len(day["races"])
        buy = sum(1 for r in day["races"].values() if (r.get("pred") or {}).get("verdict") in ("A", "B"))
        print("{} 全{}レース / 買い {}レース".format(d, n, buy))
    build_history()


if __name__ == "__main__":
    main()

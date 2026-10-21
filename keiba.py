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
  keiba/past/YYYYMMDD.json.gz … 馬柱(過去5走)。アプリは読まない。予想の作り直し用
  keiba/history.json        … 収支・条件別の回収率・負けパターン(規則 30)

使い方:
  python keiba.py                 # 本日(JST)。前日の結果が欠けていれば埋める
  python keiba.py --date 20261018
  python keiba.py --history-only  # days/ から history.json だけ作り直す
  python keiba.py --backfill-date 20261001
                                  # 過去の日を再現する(その日の馬柱で予想し、結果と照合)
"""

import argparse
import glob
import gzip
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

import keiba_logic as L
import keiba_model as KM
import keiba_speed as SP

JST = timezone(timedelta(hours=9))
OUT = "keiba"
SITE = {"jra": "https://race.netkeiba.com", "nar": "https://nar.netkeiba.com"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "ja"}
WAIT = 0.4                 # 1リクエストごとの間隔(秒。スレッドごと)
WORKERS = 4                # 1日のレースを同時に取る数
CARD_REFRESH = 50          # 出馬表(馬場・取消)を取り直す間隔(分)
ODDS_REFRESH = 20          # オッズを取り直す間隔(分)
RESULT_AFTER = 8           # 発走から何分後に結果を見に行くか
# 予想する中央・地方(2026-10-21: 中央だけに絞る。地方は取らない・予想しない・収支に出さない)
CENTERS = tuple(c for c in os.environ.get("KEIBA_CENTERS", "jra").split(",") if c)
AHEAD_DAYS = 2             # 当日の更新で、何日先の開催まで出馬表を取って予想しておくか
PAST_VER = 3               # 馬柱の読み方を直したら上げる(発走前のレースは取り直す)。3: 走破タイムを足した

_TL = threading.local()


def sess():
    """スレッドごとのセッション(requests.Session はスレッド間で共有しない)"""
    s = getattr(_TL, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(UA)
        _TL.s = s
    return s


# ---------------------------------------------------------------- 取得
def get(url, params=None, as_json=False):
    """取れなければ None。混んでいる(429/403/5xx)ときは待ってやり直す"""
    for attempt in range(5):
        try:
            r = sess().get(url, params=params, timeout=30)
            time.sleep(WAIT)
            if r.status_code == 200:
                if as_json:
                    return r.json()
                return decode(r.content)
            if r.status_code == 404:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(min(60, 2 ** attempt * 2) + random.random())
    print("取れない", url, params)
    return None


def pmap(fn, items, workers=None):
    """items を並列に処理する(順番は保つ)"""
    items = list(items)
    n = min(workers or WORKERS, len(items))
    if n <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


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
    """その日の対象(CENTERS)のレースID。一覧が取れなかったら None(開催が無い日は [])"""
    ids, ok = set(), True
    if "jra" in CENTERS:
        t = get(SITE["jra"] + "/top/race_list_sub.html", {"kaisai_date": d})
        ok = ok and t is not None
        ids |= set(re.findall(r"race_id=(\d{12})", t or ""))
    if "nar" in CENTERS:
        # 地方は race_list_sub.html にレースIDと開催ID(kaisai_id)が載る(race_list.html には無い)
        t = get(SITE["nar"] + "/top/race_list_sub.html", {"kaisai_date": d})
        ok = ok and t is not None
        t = t or ""
        ids |= set(re.findall(r"race_id=(\d{12})", t))
        kids = sorted(set(re.findall(r"kaisai_id=(\d{10})", t)))
        for s in pmap(lambda kid: get(SITE["nar"] + "/top/race_list_sub.html",
                                      {"kaisai_id": kid, "kaisai_date": d}) or "", kids):
            ids |= set(re.findall(r"race_id=(\d{12})", s))
    # 地方のIDは日付(MMDD)を含む。別の日が混ざらないようにする
    out = []
    for rid in ids:
        if not venue_of(rid) or venue_of(rid) == "帯広" or center_of(rid) not in CENTERS:   # ばんえいは対象外
            continue
        if center_of(rid) == "nar" and (rid[:4] != d[:4] or rid[6:10] != d[4:8]):
            continue
        if center_of(rid) == "jra" and rid[:4] != d[:4]:
            continue
        out.append(rid)
    return sorted(out) if ok else None


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
    m = re.search(r"天候\s*:\s*(晴|曇|小雨|雨|小雪|雪)", d1)
    race["tenko"] = m.group(1) if m else None
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
    # 走破タイム(秒)。「1:34.8」、1分を切るときは「57.9」のこともある
    m = re.search(r"(\d{1,2}):(\d{2}\.\d)", d5)
    if m:
        r["time"] = round(int(m.group(1)) * 60 + float(m.group(2)), 1)
    else:
        m = re.search(r"(?<![\d.:])(\d{2}\.\d)(?![\d])", d5.replace(r.get("surface", "") + str(r.get("dist", "")), ""))
        r["time"] = float(m.group(1)) if m else None
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


def parse_odds_form(html):
    """odds_get_form.html(単勝)。馬番は「馬名の前にある最後の数字の列」(中央は 枠・馬番、地方は 人気・馬番)"""
    out = {}
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        um = nk = o = None
        for td in tds:
            cl = td.get("class") or []
            t = L.nfkc(txt(td))
            if "Horse_Name" in cl:
                break
            if "Ninki" in cl:
                nk = num(t)
            elif re.fullmatch(r"\d{1,2}", t):
                um = int(t)
        for td in tds:
            cl = td.get("class") or []
            t = L.nfkc(txt(td))
            if "Odds" in cl and re.fullmatch(r"\d+\.\d", t):
                o = float(t)
            elif "Ninki" in cl or "Popular_Ninki" in cl:
                nk = num(t) or nk
        if um and o:
            out[um] = (o, nk)
    return out


def fetch_odds(rid):
    """馬番 -> (単勝オッズ, 人気)。発売前は空"""
    c = center_of(rid)
    html = get(SITE[c] + "/odds/odds_get_form.html", {"type": "b1", "race_id": rid})
    out = parse_odds_form(html) if html else {}
    if not out and c == "jra":
        j = get(SITE[c] + "/api/api_get_jra_odds.html",
                {"pid": "api_get_jra_odds", "input": "UTF-8", "output": "json",
                 "race_id": rid, "type": "1", "action": "init", "compress": "0"}, as_json=True)
        try:
            for k, v in j["data"]["odds"]["1"].items():
                o, nk = num(v[0], float), num(v[2])
                if o:
                    out[int(k)] = (o, nk)
        except (TypeError, KeyError, IndexError, AttributeError):
            pass
    # 人気が無ければオッズの低い順
    if out and not all(nk for _, nk in out.values()):
        ranked = sorted(out, key=lambda u: out[u][0])
        out = {u: (out[u][0], i + 1) for i, u in enumerate(ranked)}
    return out


PAY = {"Tansho": ("単勝", 1), "Umaren": ("馬連", 2), "Wide": ("ワイド", 2), "Umatan": ("馬単", 2),
       "Fuku3": ("3連複", 3), "Tan3": ("3連単", 3)}
ORDERED = ("馬単", "3連単")      # 着順どおりの組で払い戻す券種


def parse_result(html):
    """着順・払戻・人気。列は見出し(着順・馬番・人気・単勝)から探す(中央と地方で行の class が違う)"""
    soup = BeautifulSoup(html, "html.parser")
    rows, ninki, odds = [], {}, {}
    table = soup.select_one("table.RaceTable01") or soup.select_one("#All_Result_Table")
    col = {"rank": 0, "umaban": 2, "ninki": None, "odds": None}
    if table:
        head = table.find("tr")
        ths = [L.nfkc(txt(th)).replace(" ", "") for th in head.find_all(["th", "td"])] if head else []   # 「着<br>順」
        for i, t in enumerate(ths):
            if t.startswith("着順"):
                col["rank"] = i
            elif t.startswith("馬番"):
                col["umaban"] = i
            elif t.startswith("人気"):
                col["ninki"] = i
            elif t.startswith("単勝") or t.startswith("オッズ"):
                col["odds"] = i
    for tr in (table.find_all("tr") if table else []):
        tds = tr.find_all("td")
        if len(tds) <= max(col["rank"], col["umaban"]):
            continue
        rk, um = num(L.nfkc(txt(tds[col["rank"]]))), num(L.nfkc(txt(tds[col["umaban"]])))
        if not um:
            continue
        if rk:
            rows.append((rk, um))
        # 人気と単勝オッズ(過去の日の再現で「人気」に使う)
        for key, store, pat, typ in (("ninki", ninki, r"\d{1,2}", int), ("odds", odds, r"\d+\.\d", float)):
            cells = [tds[col[key]]] if col[key] is not None and col[key] < len(tds) else \
                [td for td in tds if "Odds" in (td.get("class") or [])]
            for td in cells:
                t = L.nfkc(txt(td))
                if re.fullmatch(pat, t):
                    store[str(um)] = typ(t)
                    break
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
            c = c if typ in ORDERED else sorted(c)
            d["-".join(str(x) for x in c)] = y
        pay[typ] = d
    return {"order": order, "payout": pay, "ninki": ninki, "odds": odds}


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


def day_path(d):
    return "{}/days/{}.json".format(OUT, d)


def past_path(d):
    return "{}/past/{}.json.gz".format(OUT, d)


def load_day(d):
    """日別ファイルに馬柱(別ファイル)を戻す"""
    day = load(day_path(d), {"date": d, "races": {}})
    try:
        with gzip.open(past_path(d), "rt", encoding="utf-8") as f:
            past = json.load(f)
    except Exception:
        past = {}
    for rid, rc in day["races"].items():
        pm = past.get(rid) or {}
        for h in rc.get("horses") or []:
            if "past" not in h and str(h["umaban"]) in pm:
                h["past"] = pm[str(h["umaban"])]
    return day


def save_day(d, day):
    """馬柱は別ファイル(gz)へ。アプリが読む日別ファイルは軽くしておく"""
    past, slim = {}, dict(day, races={})
    for rid, rc in day["races"].items():
        r2 = dict(rc)
        if "horses" in rc:
            r2["horses"] = []
            for h in rc["horses"]:
                h2 = dict(h)
                runs = h2.pop("past", None)
                if runs is not None:
                    past.setdefault(rid, {})[str(h["umaban"])] = runs
                r2["horses"].append(h2)
        slim["races"][rid] = r2
    save(day_path(d), slim)
    if past:
        os.makedirs(os.path.dirname(past_path(d)), exist_ok=True)
        tmp = past_path(d) + ".tmp"
        with gzip.GzipFile(tmp, "wb", mtime=0) as f:      # 中身が同じなら同じバイト列(差分を出さない)
            f.write(json.dumps(past, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode())
        os.replace(tmp, past_path(d))
    return slim


def start_dt(d, hhmm):
    if not hhmm:
        return None
    return datetime.strptime(d + hhmm, "%Y%m%d%H:%M").replace(tzinfo=JST)


def minutes_since(iso, now):
    if not iso:
        return 1e9
    return (now - datetime.fromisoformat(iso)).total_seconds() / 60


_WF = {}
_WF_LOCK = threading.Lock()


def past_probs(rid, d):
    """過去の日の予想に使う確率。先の結果を知ったモデルで過去の成績をよく見せないため、
    1) その月より前だけで学んだモデル(keiba/model_wf.json.gz)があればそれ
    2) 今のモデルが学んだ期間より後の日なら今のモデル(None を返すと predict が使う)
    3) どちらでもなければ手作りの評価(False)"""
    with _WF_LOCK:                      # 並列に呼ばれるので、読み終わるまで他を待たせる
        if "end" not in _WF:
            try:
                with gzip.open(os.path.join(OUT, "model_wf.json.gz"), "rt", encoding="utf-8") as f:
                    _WF["wf"] = json.load(f)["probs"]
            except (OSError, ValueError, KeyError):
                _WF["wf"] = {}
            try:
                _WF["end"] = json.load(open(KM.META_FILE, encoding="utf-8")).get("data_end", "")
            except (OSError, ValueError):
                _WF["end"] = ""
    pw = _WF["wf"].get(rid)
    if pw:
        return {int(u): v for u, v in pw.items()}
    if _WF["end"] and d > _WF["end"]:
        return None
    return False


def refetch_past(d, now):
    """その日のレースの馬柱だけを今の読み方で取り直す(予想・結果はそのまま)"""
    day = load_day(d)
    iso_d = d[:4] + "-" + d[4:6] + "-" + d[6:]

    def one(rid):
        rc = day["races"][rid]
        if (not rc.get("horses") or rc.get("past_ver") == PAST_VER or venue_of(rid) == "帯広"
                or center_of(rid) not in CENTERS):
            return 0
        html = get(SITE[center_of(rid)] + "/race/shutuba_past.html", {"race_id": rid})
        if not html:
            return 0
        past = parse_past(html)
        if not past:
            return 0
        for h in rc["horses"]:
            h["past"] = [r for r in past.get(h["umaban"], []) if (r.get("date") or "") < iso_d]
        rc["past_ver"] = PAST_VER
        return 1
    n = sum(pmap(one, sorted(day["races"])))
    save_day(d, day)
    left = sum(1 for r in day["races"].values() if r.get("horses") and r.get("past_ver") != PAST_VER
               and venue_of(r["id"]) != "帯広" and center_of(r["id"]) in CENTERS)
    print("{} 馬柱を取り直した {} / 残り {}".format(d, n, left), flush=True)


def refetch_result(d, now):
    """結果(払戻)を今の読み方で取り直す。馬単の払戻が無い日に使う(予想はそのまま、照合はやり直す)"""
    day = load_day(d)

    def one(rid):
        rc = day["races"][rid]
        res = rc.get("result") or {}
        if not res or "馬単" in (res.get("payout") or {}) or venue_of(rid) == "帯広" or center_of(rid) not in CENTERS:
            return 0
        html = get(SITE[center_of(rid)] + "/race/result.html", {"race_id": rid})
        new = parse_result(html) if html else None
        if not new or len(new["order"]) < 3 or not new["payout"]:
            return 0
        rc["result"] = new
        if rc.get("pred"):
            rc["settle"] = L.settle(rc["pred"], new)
        return 1
    n = sum(pmap(one, sorted(day["races"])))
    save_day(d, day)
    left = sum(1 for r in day["races"].values() if r.get("result") and "馬単" not in (r["result"].get("payout") or {})
               and venue_of(r["id"]) != "帯広" and center_of(r["id"]) in CENTERS)
    print("{} 結果を取り直した {} / 残り {}".format(d, n, left), flush=True)


def predict_past(rc, d, today):
    pr = past_probs(rc["id"], d)
    if pr is False:
        return L.predict(rc, today, use_model=False)
    return L.predict(rc, today, probs=pr)


def run_day(d, now, backfill=False):
    """1日分を進める。
    ふつう(当日) … 発走前は出馬表・馬柱・オッズを取って予想を作り直し、発走後は固定して結果を照合
    backfill(過去の日) … 予想の無いレースは、その日の出馬表と馬柱(そのレースより前の走)で予想を作る。
                         人気は結果ページの確定人気。記録には「再現」と印を付けて当日の予想と分ける"""
    day = load_day(d)
    races = day["races"]
    if not races or minutes_since(day.get("list_at"), now) > (10 ** 8 if backfill else 180):
        ids = race_ids(d)
        print("{} レース一覧 {}".format(d, "取れない" if ids is None else "{}件".format(len(ids))))
        if ids is not None:
            day["list_at"] = now.isoformat(timespec="seconds")
        for rid in ids or []:
            races.setdefault(rid, {"id": rid})
    today = datetime.strptime(d, "%Y%m%d").date()
    iso_d = today.isoformat()

    def one(rid):
        """1レース分(レースごとに独立しているので並列に回せる)"""
        rc = races[rid]
        if venue_of(rid) == "帯広" or center_of(rid) not in CENTERS:
            return
        st = start_dt(d, rc.get("start"))
        started = st is not None and now >= st
        site = SITE[center_of(rid)]
        need_pred = not started or (backfill and not rc.get("pred"))
        # 出馬表(馬場・取消は朝から変わるので発走前は取り直す)
        if need_pred and (not rc.get("horses") or
                          (not started and minutes_since(rc.get("card_at"), now) > CARD_REFRESH)):
            html = get(site + "/race/shutuba.html", {"race_id": rid})
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
                    st = start_dt(d, rc.get("start"))
                    started = st is not None and now >= st
        if not rc.get("horses"):
            return
        # 馬柱(過去5走)は1回だけ。そのレースより前の走だけを使う
        if need_pred and rc.get("past_ver") != PAST_VER:
            html = get(site + "/race/shutuba_past.html", {"race_id": rid})
            if html:
                past = parse_past(html)
                for h in rc["horses"]:
                    h["past"] = [r for r in past.get(h["umaban"], []) if (r.get("date") or "") < iso_d]
                if past:
                    rc["past_at"] = now.isoformat(timespec="seconds")
                    rc["past_ver"] = PAST_VER
        # オッズ(人気)。予想には使わず、最後に照らすだけ
        if not started and not backfill and minutes_since(rc.get("odds_at"), now) > ODDS_REFRESH:
            od = fetch_odds(rid)
            if od:
                for h in rc["horses"]:
                    if h["umaban"] in od:
                        h["odds"], h["ninki"] = od[h["umaban"]]
                rc["odds_at"] = now.isoformat(timespec="seconds")
        # 結果
        if st and now >= st + timedelta(minutes=RESULT_AFTER) and not rc.get("result"):
            html = get(site + "/race/result.html", {"race_id": rid})
            res = parse_result(html) if html else None
            if res and len(res["order"]) >= 3:
                rc["result"] = res

    def two(rid):
        """予想。当日は発走後に固定。過去の日は予想が無いときだけ作る(人気は確定人気)"""
        rc = races[rid]
        if venue_of(rid) == "帯広" or center_of(rid) not in CENTERS or not rc.get("horses"):
            return
        st = start_dt(d, rc.get("start"))
        started = st is not None and now >= st
        if not started and not backfill and rc.get("past_at"):
            rc["pred"] = L.predict(rc, today)
            rc["pred"]["src"] = "live"
            rc["pred_at"] = now.isoformat(timespec="seconds")
        elif backfill and not rc.get("pred") and rc.get("past_at"):
            res = rc.get("result") or {}
            for h in rc["horses"]:
                u = str(h["umaban"])
                if not h.get("ninki") and u in (res.get("ninki") or {}):
                    h["ninki"] = res["ninki"][u]
                    h["odds"] = (res.get("odds") or {}).get(u)
            rc["pred"] = predict_past(rc, d, today)
            rc["pred"]["src"] = "backfill"
            rc["pred_at"] = now.isoformat(timespec="seconds")
        if rc.get("result") and rc.get("pred"):
            rc["settle"] = L.settle(rc["pred"], rc["result"])

    pmap(one, sorted(races))
    # 馬場差の表に無い日(前回の学習より後の日)を、この日の全レースの馬柱から足してから予想する
    SP.extend(SP.load(), [r for rc in races.values() for h in rc.get("horses") or [] for r in h.get("past") or []],
              KM.center_of)
    pmap(two, sorted(races))
    if backfill:
        # 結果が取れないレース(中止・取得失敗)は2回試したらあきらめる
        todo = []
        for r in races.values():
            if venue_of(r["id"]) == "帯広" or center_of(r["id"]) not in CENTERS or r.get("result"):
                continue
            r["tries"] = r.get("tries", 0) + 1
            if r["tries"] < 2:
                todo.append(r["id"])
        # 開催の無い日(一覧は取れて対象のレースが0件)も済みにする
        day["complete"] = (bool(races) or bool(day.get("list_at"))) and not todo
    day["updated"] = now.isoformat(timespec="seconds")
    return save_day(d, day)


# ---------------------------------------------------------------- 収支と分析(規則 30)
def dist_band(dd):
    if not dd:
        return "?"
    return "〜1400m" if dd <= 1400 else ("1500〜1800m" if dd <= 1800 else "1900m〜")


def field_band(n):
    return "〜10頭" if (n or 0) <= 10 else ("11〜14頭" if n <= 14 else "15頭〜")


def gap_band(g):
    return "差6〜" if g >= 6 else ("差4.5〜6" if g >= 4.5 else "差3.5〜4.5")


def bet_key(b):
    return b["type"] + ("(◎飛び保険)" if "飛" in b["role"] else "")


def alt_bets(p, res):
    """同じ◎○で、買い方を変えて1点100円で買っていたらの払戻(参考)。払戻表にその券種が無ければ None
    ワイド ◎-○(今の買い方) / 馬連 ◎-○ / 馬単 ◎→○ / 単勝 ◎"""
    pay = res.get("payout") or {}
    h, t = p.get("honmei"), p.get("taikou")
    if not h or not t:
        return {}
    pair = "-".join(str(x) for x in sorted((h, t)))
    keys = {"ワイド": pair, "馬連": pair, "馬単": "{}-{}".format(h, t), "単勝": str(h)}
    return {typ: ((pay.get(typ) or {}).get(k, 0) * L.UNIT // 100 if typ in pay else None) for typ, k in keys.items()}


# アプリの収支に出す最初の日。これより前の日は学習用に取り込んだもの(モデルの月ごとの確率が無い)
HISTORY_FROM = "20240101"


def build_history():
    """日別ファイルから、期間で絞れる形の成績を作る。
    days  … 日ごとの合計と、印・負けパターンの集計(アプリが期間で足し合わせる)
    races … 買ったレース1件ずつ(条件・券種ごとの投資と払戻)"""
    days, races = [], []
    for f in sorted(glob.glob(OUT + "/days/*.json")):
        if os.path.basename(f)[:8] < HISTORY_FROM:
            continue
        day = load(f, None)
        if not day:
            continue
        d = day["date"]
        # stats は「出どころ(live=当日 / backfill=再現)|中央/地方」ごとに分ける(アプリで絞り込む)
        row = {"date": d, "races": 0, "buy": 0, "settled": 0, "stake": 0, "ret": 0, "hit": 0,
               "src": [], "stats": {}, "races_c": {}, "buy_c": {}}
        for rid, rc in sorted(day.get("races", {}).items()):
            p = rc.get("pred")
            if not p or center_of(rid) not in CENTERS:
                continue
            row["races"] += 1
            center = "中央" if rc.get("center") == "jra" else "地方"
            row["races_c"][center] = row["races_c"].get(center, 0) + 1
            src = p.get("src", "live")
            if src not in row["src"]:
                row["src"].append(src)
            order = (rc.get("result") or {}).get("order") or []
            # 印と消しの成績(見送りも含めて精度を見る)
            if len(order) >= 3:
                sx = row["stats"].setdefault(src + "|" + center, {"marks": {}, "tags": {}, "base": [0, 0]})
                for h in p.get("horses", []):
                    rk = L.rank_of(order, h["umaban"])
                    if rk is None:
                        continue
                    m = sx["marks"].setdefault(h.get("mark") or "無", [0, 0, 0])
                    m[0] += 1
                    m[1] += rk == 1
                    m[2] += rk <= 3
                    sx["base"][0] += 1
                    sx["base"][1] += rk <= 3
                    for t in {re.sub(r"\(.*?\)", "", x["t"]) for x in h["minus"] if x["p"] <= -2}:
                        tg = sx["tags"].setdefault(t, [0, 0])
                        tg[0] += 1
                        tg[1] += rk <= 3
            if p.get("verdict") == "見送り":
                continue
            row["buy"] += 1
            row["buy_c"][center] = row["buy_c"].get(center, 0) + 1
            s = rc.get("settle")
            if not s:
                continue
            row["settled"] += 1
            row["stake"] += s["stake"]
            row["ret"] += s["ret"]
            row["hit"] += 1 if s["hit"] else 0
            bt = {}
            for b in p.get("bets", []):
                x = bt.setdefault(bet_key(b), [0, 0, 0])
                x[0] += len(b["combos"]) * b["unit"]
                x[1] += b.get("ret", 0)
                x[2] += 1 if b.get("hit") else 0
            races.append({
                "date": d, "id": rid, "venue": rc.get("venue"), "rno": rc.get("rno"),
                "center": center, "src": src,
                "verdict": p["verdict"],
                "surface": {"芝": "芝", "ダ": "ダート"}.get(rc.get("surface"), "?"),
                "dist": dist_band(rc.get("dist")), "field": field_band(rc.get("n")),
                "baba": "道悪" if (rc.get("baba") or "良") in L.HEAVY else "良・稍",
                "gap": gap_band(p.get("gap") or 0), "pace": (p.get("pace") or {}).get("pace", "?"),
                "stake": s["stake"], "ret": s["ret"], "hit": s["hit"],
                "hr": s.get("honmei_rank"), "tr": s.get("taikou_rank"), "bets": bt,
                "alt": alt_bets(p, rc.get("result") or {})})
        days.append(row)
    hist = {"updated": datetime.now(JST).isoformat(timespec="seconds"), "unit": L.UNIT,
            "days": days, "races": races}
    save(OUT + "/history.json", hist)
    return hist


# ---------------------------------------------------------------- 診断(取得がおかしいとき)
def diag(d):
    """netkeiba の応答の形をログに出す。何も保存しない。"""
    def raw(url, params=None):
        try:
            r = sess().get(url, params=params, timeout=30)
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
    ids = race_ids(d) or []
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
    yid = race_ids(y) or []
    print("==== 前日", y, len(yid), "レース")
    for rid in [x for x in yid if center_of(x) == "jra"][-1:] + [x for x in yid if center_of(x) == "nar"][-1:]:
        pass
    for rid in [x for x in yid if center_of(x) == "nar"][:1]:
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
    global WAIT, WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--history-only", action="store_true")
    ap.add_argument("--diag", action="store_true", help="取得の診断をログに出す(保存しない)")
    ap.add_argument("--backfill-date", help="過去の日 YYYYMMDD を再現する")
    ap.add_argument("--backfill-range", nargs=2, metavar=("START", "END"),
                    help="START〜END(YYYYMMDD)の日を新しい日から順に再現する。済んだ日は飛ばす")
    ap.add_argument("--backfill-dates", help="再現する日をカンマ区切りで(並列ジョブ用)")
    ap.add_argument("--workers", type=int, help="1日のレースを同時に取る数")
    ap.add_argument("--refetch-past", action="store_true",
                    help="--backfill-dates の日の馬柱だけを今の読み方で取り直す(予想はそのまま)")
    ap.add_argument("--refetch-result", action="store_true",
                    help="--backfill-dates の日の結果(払戻)だけを今の読み方で取り直す(予想はそのまま)")
    ap.add_argument("--no-history", action="store_true", help="history.json を作らない(並列ジョブ用)")
    ap.add_argument("--repredict", action="store_true",
                    help="再現の予想を今のロジックで作り直す(保存済みの馬柱を使う。取得しない)")
    a = ap.parse_args()
    if a.diag:
        diag(a.date or datetime.now(JST).strftime("%Y%m%d"))
        return
    now = datetime.now(JST)
    today_s = now.strftime("%Y%m%d")
    if a.backfill_date or a.backfill_range or a.backfill_dates:
        # 過去の日はまとめて取るので、レースを並列に取る(混んだら get が待ってやり直す)
        WAIT = 0.15
        WORKERS = a.workers or 8
    elif a.workers:
        WORKERS = a.workers
    if a.repredict:
        repredict()
    elif a.backfill_date or a.backfill_range or a.backfill_dates:
        if a.backfill_dates:
            dates = sorted({x for x in re.split(r"[,\s]+", a.backfill_dates) if x}, reverse=True)
        else:
            start, end = a.backfill_range or (a.backfill_date, a.backfill_date)
            dates, d0 = [], datetime.strptime(end, "%Y%m%d")
            while d0.strftime("%Y%m%d") >= start:
                dates.append(d0.strftime("%Y%m%d"))
                d0 -= timedelta(days=1)
        fails = 0
        for d in dates:
            if fails >= 2:
                # 一覧が続けて取れないのは netkeiba のメンテナンスなど。残りは次の再現にまわす
                print("{} から先は取らない(レース一覧が{}日続けて取れない。メンテナンス中?)".format(d, fails))
                break
            if a.refetch_past or a.refetch_result:
                if d < today_s:
                    (refetch_past if a.refetch_past else refetch_result)(d, datetime.now(JST))
                continue
            if d >= today_s:
                print("{} は今日以降なので再現しません".format(d))
            elif load(day_path(d), {}).get("complete"):
                print("{} は済み".format(d))
            else:
                t0 = time.time()
                day = run_day(d, datetime.now(JST), backfill=True)
                fails = 0 if day.get("list_at") or day["races"] else fails + 1
                show(day)
                rs = [r for r in day["races"].values() if venue_of(r["id"]) != "帯広"]
                print("  {:.0f}秒 / 出馬表なし {} / 馬柱なし {} / 結果なし {}".format(
                    time.time() - t0, sum(1 for r in rs if not r.get("horses")),
                    sum(1 for r in rs if r.get("horses") and not r.get("past_at")),
                    sum(1 for r in rs if not r.get("result"))), flush=True)
    elif a.date and a.date < today_s:
        # 過去の日を指定されたら再現として扱う(当日用の data.json は書き換えない)
        WAIT, WORKERS = 0.15, a.workers or 8
        show(run_day(a.date, now, backfill=True))
    elif not a.history_only:
        d = a.date or now.strftime("%Y%m%d")
        # 前日の結果が欠けていれば先に埋める
        y = (datetime.strptime(d, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        if os.path.exists(day_path(y)) and not load(day_path(y), {}).get("complete"):
            run_day(y, now, backfill=True)
        day = run_day(d, now)
        save(OUT + "/data.json", day)
        show(day)
        # 中央の出馬表は木・金に出る(土曜は木曜、日曜は金曜)。明日・明後日の開催も先に予想しておく
        for k in range(1, AHEAD_DAYS + 1):
            nd = (datetime.strptime(d, "%Y%m%d") + timedelta(days=k)).strftime("%Y%m%d")
            ahead = run_day(nd, now)
            if ahead["races"]:
                show(ahead)
    if not a.no_history:
        build_history()


def repredict():
    """「再現」の予想を、保存してある出馬表と馬柱から今のロジックで作り直す(取得はしない)。
    当日の予想(live)は書き換えない。ロジックを変えたあとに過去の日の成績を見直すため"""
    n = 0
    today_s = datetime.now(JST).strftime("%Y%m%d")
    for f in sorted(glob.glob(OUT + "/days/*.json")):
        d = os.path.basename(f)[:8]
        if d >= today_s:              # 当日はふつうの更新に任せる
            continue
        day = load_day(d)
        today = datetime.strptime(d, "%Y%m%d").date()
        changed = False
        for rc in day["races"].values():
            p = rc.get("pred")
            if (not rc.get("horses") or not rc.get("past_at") or (p and p.get("src") != "backfill")
                    or center_of(rc["id"]) not in CENTERS):
                continue
            rc["pred"] = predict_past(rc, d, today)
            rc["pred"]["src"] = "backfill"
            if rc.get("result"):
                rc["settle"] = L.settle(rc["pred"], rc["result"])
            n += 1
            changed = True
        if changed:
            save_day(d, day)
            show(day)
    print("作り直した予想", n)


def show(day):
    rs = day["races"].values()
    buy = sum(1 for r in rs if (r.get("pred") or {}).get("verdict") in ("A", "B"))
    st = sum((r.get("settle") or {}).get("stake", 0) for r in rs)
    rt = sum((r.get("settle") or {}).get("ret", 0) for r in rs)
    print("{} 全{}レース / 買い {}レース / 投資 {} 払戻 {}".format(day["date"], len(day["races"]), buy, st, rt))


if __name__ == "__main__":
    main()

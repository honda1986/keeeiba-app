#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_weather.py -- 競馬場ごとの気温・湿度・雨量・天気(1時間ごと)を keiba_weather.json に貯める

  python tools/fetch_weather.py --start 2022-01-01          # 過去(Open-Meteo の過去データ)
  python tools/fetch_weather.py --recent                    # 直近7日と先3日(予報。毎日の更新で使う)

■ データ
  Open-Meteo(https://open-meteo.com/ CC BY 4.0)。過去は再解析データ(archive-api)、直近と先は予報(api)。
  競馬場の場所の格子の値なので、場内の観測とは少し違う。発走のある 8〜17 時だけを残す。
  keiba_weather.json = {"hours": [8..17], "vars": [...], "data": {場: {YYYYMMDD: [[気温, 湿度, 雨量, 天気コード] × 時]}}}
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WX_FILE = os.path.join(HERE, "keiba_weather.json")
HOURS = list(range(8, 18))
VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code"]
JST = timezone(timedelta(hours=9))

# 中央10場の場所(緯度, 経度)
VENUE_LL = {
    "札幌": (43.0778, 141.3297), "函館": (41.7836, 140.7756), "福島": (37.7739, 140.4617),
    "新潟": (37.9506, 139.1747), "東京": (35.6636, 139.4864), "中山": (35.7244, 139.9608),
    "中京": (35.0667, 136.9906), "京都": (34.9067, 135.7275), "阪神": (34.7783, 135.3633),
    "小倉": (33.8417, 130.8700),
}


def load(path=WX_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"hours": HOURS, "vars": ["temp", "hum", "prec", "code"], "data": {}}


def save(wx, path=WX_FILE):
    wx["made"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    wx["source"] = "Open-Meteo.com (CC BY 4.0)"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wx, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def get(url, params):
    for k in range(5):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print("GET", r.status_code, url, params.get("start_date"), r.text[:200], flush=True)
        except (requests.RequestException, ValueError) as e:
            print("ERR", type(e).__name__, e, flush=True)
        time.sleep(5 * (k + 1))
    return None


def put(wx, venue, js):
    """1時間ごとの値を、日付ごとの 8〜17 時に詰める"""
    h = (js or {}).get("hourly") or {}
    ts = h.get("time") or []
    dv = wx["data"].setdefault(venue, {})
    n = 0
    for i, t in enumerate(ts):
        d, hh = t[:10].replace("-", ""), int(t[11:13])
        if hh not in HOURS:
            continue
        vals = [h[v][i] if h.get(v) else None for v in VARS]
        if all(v is None for v in vals):
            continue
        row = dv.setdefault(d, [None] * len(HOURS))
        row[HOURS.index(hh)] = [None if v is None else round(v, 1) for v in vals]
        n += 1
    return n


def fetch_archive(wx, start, end):
    for venue, (lat, lon) in VENUE_LL.items():
        a = start
        while a <= end:
            b = min(end, date(a.year, 12, 31))
            js = get("https://archive-api.open-meteo.com/v1/archive",
                     {"latitude": lat, "longitude": lon, "start_date": a.isoformat(), "end_date": b.isoformat(),
                      "hourly": ",".join(VARS), "timezone": "Asia/Tokyo"})
            print(venue, a, b, put(wx, venue, js), "時間", flush=True)
            a = b + timedelta(days=1)


def fetch_recent(wx, past_days=7, forecast_days=4):
    for venue, (lat, lon) in VENUE_LL.items():
        js = get("https://api.open-meteo.com/v1/forecast",
                 {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS), "timezone": "Asia/Tokyo",
                  "past_days": past_days, "forecast_days": forecast_days})
        print(venue, "直近と予報", put(wx, venue, js), "時間", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="過去データの最初の日 YYYY-MM-DD")
    ap.add_argument("--end", help="過去データの最後の日(既定は6日前。再解析は数日遅れる)")
    ap.add_argument("--recent", action="store_true", help="直近7日と先の予報")
    a = ap.parse_args()
    wx = load()
    if a.start:
        end = date.fromisoformat(a.end) if a.end else datetime.now(JST).date() - timedelta(days=6)
        fetch_archive(wx, date.fromisoformat(a.start), end)
    if a.recent:
        fetch_recent(wx)
    if not a.start and not a.recent:
        ap.print_help()
        sys.exit(1)
    save(wx)
    print("保存", WX_FILE, {v: len(d) for v, d in wx["data"].items()})


if __name__ == "__main__":
    main()

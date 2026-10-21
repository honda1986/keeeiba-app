#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keiba_weather.py -- 気温・湿度・雨量・天気をモデルの材料にする

  keiba_weather.json(tools/fetch_weather.py が作る)の、競馬場ごと・日ごと・8〜17時の値を使う。
  レース … 発走の時刻の気温・湿度・天気、その日の朝からの雨量
  馬     … 過去走のときの気温の平均と今日との差、暑い日(25℃以上)・寒い日(10℃以下)・雨の日の着順の比が
           ふだんよりどれだけ良いか(夏に強い・寒いと走らない・雨巧者)
  中央10場の分しか無いので、地方などで走った過去走は数えない。
"""

import json
import os
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
WX_FILE = os.path.join(HERE, "keiba_weather.json")
HOT, COLD, RAINY = 25.0, 10.0, 1.0      # ℃ / ℃ / その日の 8〜17 時の雨量(mm)

_WX = {"t": None, "tried": False}
_LOCK = threading.Lock()


def load(reload=False):
    with _LOCK:
        if reload or not _WX["tried"]:
            try:
                with open(WX_FILE, encoding="utf-8") as f:
                    _WX["t"] = json.load(f)
            except (OSError, ValueError):
                _WX["t"] = None
            _WX["tried"] = True
    return _WX["t"]


def _day(wx, venue, d):
    """その日の 8〜17 時の [[気温, 湿度, 雨量, 天気コード] or None, ...]"""
    if not wx or not venue or not d:
        return None
    return (wx.get("data") or {}).get(venue, {}).get(d.replace("-", ""))


def _hour_idx(wx, hhmm):
    hours = wx.get("hours") or list(range(8, 18))
    try:
        h = int(hhmm[:2]) + (1 if int(hhmm[3:5]) >= 30 else 0)
    except (TypeError, ValueError):
        h = 15
    return min(range(len(hours)), key=lambda i: abs(hours[i] - h))


def code_kind(c):
    """WMO の天気コード -> 0 晴 / 1 曇 / 2 雨 / 3 雪"""
    if c is None:
        return None
    c = int(c)
    if c in (71, 73, 75, 77, 85, 86):
        return 3
    if c >= 51:
        return 2
    if c >= 2:
        return 1
    return 0


def at(wx, venue, d, hhmm):
    """発走のころの (気温, 湿度, 天気の種類, 朝からの雨量)。無ければ None"""
    row = _day(wx, venue, d)
    if not row:
        return None
    i = _hour_idx(wx, hhmm)
    cur = row[i] or next((row[j] for j in sorted(range(len(row)), key=lambda j: abs(j - i)) if row[j]), None)
    if not cur:
        return None
    rain = sum((r[2] or 0) for r in row[:i + 1] if r)
    return cur[0], cur[1], code_kind(cur[3]), round(rain, 1)


def day_mean(wx, venue, d):
    """その日の昼(10〜16時)の気温の平均と、8〜17時の雨量"""
    row = _day(wx, venue, d)
    if not row:
        return None
    hours = wx.get("hours") or list(range(8, 18))
    ts = [r[0] for h, r in zip(hours, row) if r and 10 <= h <= 16 and r[0] is not None]
    if not ts:
        return None
    return sum(ts) / len(ts), sum((r[2] or 0) for r in row if r)


def feats(f, past, race, today, wx=None):
    """f に気象の材料を足す。past は新しい順の過去走"""
    wx = wx if wx is not None else load()
    keys = ("wx_temp", "wx_hum", "wx_kind", "wx_rain", "wx_past_temp", "wx_temp_diff",
            "wx_hot_rr", "wx_cold_rr", "wx_rain_rr", "wx_n")
    for k in keys:
        f[k] = None
    if not wx:
        return
    d = today.isoformat() if hasattr(today, "isoformat") else str(today)
    cur = at(wx, race.get("venue"), d, race.get("start"))
    if cur:
        f["wx_temp"], f["wx_hum"], f["wx_kind"], f["wx_rain"] = cur
    rr_all, rr_hot, rr_cold, rr_rain, temps = [], [], [], [], []
    for r in past or []:
        if not r.get("rank") or not r.get("n"):
            continue
        m = day_mean(wx, r.get("venue"), r.get("date"))
        if not m:
            continue
        t, rain = m
        rr = r["rank"] / r["n"]
        temps.append(t)
        rr_all.append(rr)
        if t >= HOT:
            rr_hot.append(rr)
        if t <= COLD:
            rr_cold.append(rr)
        if rain >= RAINY:
            rr_rain.append(rr)
    f["wx_n"] = len(rr_all)
    if not rr_all:
        return
    base = sum(rr_all) / len(rr_all)
    f["wx_past_temp"] = round(sum(temps) / len(temps), 1)
    if f["wx_temp"] is not None:
        f["wx_temp_diff"] = round(f["wx_temp"] - f["wx_past_temp"], 1)
    # 着順の比(小さいほど良い)が、その条件ではふだんよりどれだけ良いか(+ほど得意)
    for k, xs in (("wx_hot_rr", rr_hot), ("wx_cold_rr", rr_cold), ("wx_rain_rr", rr_rain)):
        if xs:
            f[k] = round(base - sum(xs) / len(xs), 3)

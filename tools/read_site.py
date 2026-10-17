#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指定したカテゴリの記事を全部たどって本文をログに出す(一回だけ使う読み取り用。何も保存しない)。
   python tools/read_site.py URL [最大記事数]"""
import re
import socket
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept-Language": "ja"}
start = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
host = urlparse(start).hostname
for h in (host, host.replace("www.", "")):
    try:
        print("DNS", h, socket.gethostbyname(h))
    except OSError as e:
        print("DNS", h, "失敗", e)
S = requests.Session()
S.headers.update(UA)


def get(u):
    for i in range(3):
        try:
            r = S.get(u, timeout=30)
            time.sleep(0.7)
            print("GET", r.status_code, u, len(r.content), flush=True)
            if r.status_code == 200:
                r.encoding = r.apparent_encoding if r.encoding in (None, "ISO-8859-1") else r.encoding
                return r.text
            if r.status_code in (403, 404, 410):
                return None
        except requests.RequestException as e:
            print("ERR", u, type(e).__name__, e, flush=True)
        time.sleep(2 * (i + 1))
    return None


prefix = urlparse(start).path.rstrip("/")
cats, seen_cat, arts = [start], set(), []
while cats:
    u = cats.pop(0)
    if u in seen_cat:
        continue
    seen_cat.add(u)
    html = get(u)
    if not html:
        continue
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        v = urljoin(u, a["href"]).split("#")[0]
        p = urlparse(v)
        if p.hostname not in (host, host.replace("www.", ""), "www." + host.replace("www.", "")):
            continue
        path = p.path
        if path.startswith(prefix) and v not in seen_cat:
            cats.append(v)                        # ページ送り・下位カテゴリ
            continue
        if re.search(r"/(category|tag|page|author|feed|wp-|comment)", path) or path in ("", "/"):
            continue
        # 記事: 一覧の見出し・記事カードの中のリンク
        if a.find_parent(["article", "h2", "h3"]) or "entry" in " ".join(
                c for el in a.parents for c in (el.get("class") or [])):
            if v not in arts:
                arts.append(v)
print("カテゴリのページ", len(seen_cat))
for u in sorted(seen_cat):
    print("  CAT", u)
print("記事", len(arts))
for u in arts:
    print("  ART", u)
for u in arts[:limit]:
    html = get(u)
    if not html:
        continue
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "nav", "footer", "aside", "form"]):
        t.decompose()
    title = soup.find("h1")
    body = (soup.select_one(".entry-content") or soup.select_one(".post-content")
            or soup.select_one("article") or soup.select_one("main") or soup.body)
    print("\n##### 記事 {}\n##### {}".format(u, title.get_text(" ", strip=True) if title else ""))
    if not body:
        continue
    for el in body.find_all(["h2", "h3", "h4", "p", "li", "tr"]):
        if el.find_parent(["li", "tr"]) and el.name not in ("li", "tr"):
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if t:
            print(("■ " if el.name in ("h2", "h3", "h4") else "") + t)
print("\n##### おわり")

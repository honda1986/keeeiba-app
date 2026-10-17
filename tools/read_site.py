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
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
      "Accept-Language": "ja,en-US;q=0.9,en;q=0.8", "Accept-Encoding": "gzip, deflate",
      "Upgrade-Insecure-Requests": "1", "Referer": "https://www.google.com/"}
start = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
# 直接読めないときは Wayback Machine の最新のスナップショットを読む
VIA = "direct"
ARCH = re.compile(r"^https?://web\.archive\.org/web/\d+[a-z_]*/(https?://.+)$")


def orig(u):
    m = ARCH.match(u)
    return m.group(1) if m else u


def addr(u):
    return u if VIA == "direct" else "https://web.archive.org/web/2/" + u
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
            r = S.get(addr(u), timeout=45)
            time.sleep(0.7 if VIA == "direct" else 1.5)
            print("GET", r.status_code, addr(u), len(r.content), flush=True)
            if r.status_code == 200:
                r.encoding = r.apparent_encoding if r.encoding in (None, "ISO-8859-1") else r.encoding
                return r.text
            if r.status_code in (403, 404, 410):
                print("  本文:", re.sub(r"\s+", " ", BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True))[:400])
                return None
        except requests.RequestException as e:
            print("ERR", u, type(e).__name__, e, flush=True)
        time.sleep(2 * (i + 1))
    return None


if not get(start):
    VIA = "archive"
    print("直接は読めないので Wayback Machine から読みます")
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
    print("  title:", soup.title.get_text(strip=True) if soup.title else None,
          "| links", len(soup.find_all("a", href=True)))
    for a in soup.find_all("a", href=True)[:400]:
        print("   a:", a["href"][:140], "|", a.get_text(" ", strip=True)[:40])
    for a in soup.find_all("a", href=True):
        v = orig(urljoin(addr(u), a["href"]).split("#")[0])
        p = urlparse(v)
        if p.hostname not in (host, host.replace("www.", ""), "www." + host.replace("www.", "")):
            continue
        path = p.path
        if path.startswith(prefix) and v not in seen_cat:
            if len(seen_cat) + len(cats) < 80:
                cats.append(v)                    # ページ送り・下位カテゴリ
            continue
        if re.search(r"/(category|tag|page|author|feed|wp-|comment)", path) or path in ("", "/"):
            continue
        # 記事の候補: 同じサイトの、一覧・タグ・固定ページ以外のリンクはすべて
        if not re.search(r"/(about|contact|privacy|sitemap|profile)", path) and v not in arts:
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

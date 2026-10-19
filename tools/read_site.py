#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""記事を読むための一時ツール(何も保存しない。ログに出すだけ)
   python tools/read_site.py list URL [題名の正規表現]  … カテゴリをたどって記事のURLを出す
   python tools/read_site.py read URL                    … 1記事の本文を出す
   直接読めない(403など)ときは Wayback Machine の最新のスナップショットを読む"""
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "ja,en-US;q=0.9,en;q=0.8", "Accept-Encoding": "gzip, deflate"}
S = requests.Session()
S.headers.update(UA)
ARCH = re.compile(r"^https?://web\.archive\.org/web/\d+[a-z_]*/(https?://.+)$")
VIA = {"mode": "direct"}


def orig(u):
    m = ARCH.match(u)
    return m.group(1) if m else u


def addr(u):
    return u if VIA["mode"] == "direct" else "https://web.archive.org/web/2/" + u


def get(u):
    for i in range(3):
        try:
            r = S.get(addr(u), timeout=45)
            time.sleep(1.0)
            print("GET", r.status_code, addr(u), len(r.content), flush=True)
            if r.status_code == 200:
                if r.encoding in (None, "ISO-8859-1"):
                    r.encoding = r.apparent_encoding
                return r.text
            if r.status_code in (403, 404, 410):
                if VIA["mode"] == "direct":
                    VIA["mode"] = "archive"
                    print("直接は読めないので Wayback Machine から読みます")
                    continue
                return None
        except requests.RequestException as e:
            print("ERR", u, type(e).__name__, e, flush=True)
        time.sleep(2 * (i + 1))
    return None


def same_site(a, b):
    return (urlparse(a).hostname or "").replace("www.", "") == (urlparse(b).hostname or "").replace("www.", "")


def do_list(start, pat):
    prefix = urlparse(start).path.rstrip("/")
    todo, seen, arts = [start], set(), {}
    while todo and len(seen) < 40:
        u = todo.pop(0)
        if u in seen:
            continue
        seen.add(u)
        html = get(u)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            v = orig(urljoin(addr(u), a["href"]).split("#")[0])
            if not same_site(v, start):
                continue
            path = urlparse(v).path
            if path.startswith(prefix + "/page/") and v not in seen:
                todo.append(v)
            title = a.get_text(" ", strip=True)
            if title and re.search(pat, title) and not re.search(r"/(category|tag|page)/", path):
                arts.setdefault(v, title)
    print("一覧のページ", sorted(seen))
    print("記事", len(arts))
    for v, t in arts.items():
        print("ART", v, "|", t)


def do_read(u):
    html = get(u)
    if not html:
        print("読めませんでした")
        return
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "nav", "footer", "aside", "form", "header"]):
        t.decompose()
    h1 = soup.find("h1")
    print("##### 記事", u)
    print("##### 題名", h1.get_text(" ", strip=True) if h1 else "")
    body = (soup.select_one(".entry-content") or soup.select_one(".post_content")
            or soup.select_one(".post-content") or soup.select_one("article") or soup.select_one("main")
            or soup.select_one("#main") or soup.select_one("#contents") or soup.select_one(".contents")
            or soup.body)            # 決まった入れ物が無いサイトはページ全体から拾う
    if not body:
        print("本文が見つかりません")
        return
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "tr", "dt", "dd", "figcaption"]):
        if el.name in ("p", "h2", "h3", "h4", "h5", "dt", "dd") and el.find_parent(["li", "tr"]):
            continue
        if el.name == "li" and el.find_parent("li"):
            continue
        if el.name == "tr":
            t = " | ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"]))
        else:
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if t:
            print({"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "tr": "| ", "li": "- ",
                   "dt": "* "}.get(el.name, "") + t)
    print("##### おわり")


if __name__ == "__main__":
    if sys.argv[1] == "list":
        do_list(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    else:
        for u in re.split(r"[\s,]+", sys.argv[2].strip()):
            if u:
                do_read(u)

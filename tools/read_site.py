#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""記事を読むための一時ツール(何も保存しない。ログに出すだけ)
   python tools/read_site.py list URL [題名の正規表現]  … カテゴリをたどって記事のURLを出す
   python tools/read_site.py read URL                    … 1記事の本文を出す
   python tools/read_site.py links URL [正規表現]        … ページの中の同じサイトへのリンクを出す(複数可)
   python tools/read_site.py grep URL 正規表現            … その言葉を含む要素の中身と、元のHTMLの前後を出す(表以外の形で載っているものを探す)
   python tools/read_site.py json URL 要素のid           … ページに埋め込まれた JSON(<script id=...>)の形(キーと値の例)を出す
   python tools/read_site.py raw URL                     … ページのつくり(スクリプト・表・本文の始め)を出す(データがどこにあるかを見る)
   python tools/read_site.py probe 'URL{d}' 最初の日-最後の日 … {d} に日付(YYYYMMDD)を入れて、あるページを出す(過去の分が残っているか)
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


def do_grep(u, pat):
    html = get(u)
    if not html:
        print("読めませんでした", u)
        return
    rx = re.compile(pat)
    soup = BeautifulSoup(html, "html.parser")
    # その言葉を直接含むいちばん内側の要素と、それを囲む入れ物の中身(最初の8つ)
    hits = [t.parent for t in soup.find_all(string=rx) if t.parent and t.parent.name not in ("script", "style")]
    print("##### 要素", len(hits), "件", u)
    for el in hits[:8]:
        box = el
        for _ in range(4):                    # 中身が短すぎるときは外側へ
            if len(box.get_text(" ", strip=True)) >= 300 or not box.parent:
                break
            box = box.parent
        print("--- <{} class={}> 入れ物 <{} class={} id={}>".format(el.name, el.get("class"), box.name, box.get("class"), box.get("id")))
        print(re.sub(r"\s+", " ", box.get_text(" | ", strip=True))[:1500])
    # 元のHTML(スクリプトの中のデータも含む)の前後
    ms = list(rx.finditer(html))
    print("##### 元のHTML", len(ms), "か所")
    for m in ms[:6]:
        print("---", m.start(), html[max(0, m.start() - 400):m.end() + 800].replace("\n", " "))
    print("##### おわり")


def show_shape(x, path="", depth=0, out=None):
    """JSON の形を、キーの道すじと値の例で出す(リストは最初の1つだけたどる)"""
    if depth > 8:
        return
    if isinstance(x, dict):
        for k, v in x.items():
            show_shape(v, path + "." + str(k), depth + 1, out)
    elif isinstance(x, list):
        print("{} [リスト {}個]".format(path, len(x)))
        if x:
            show_shape(x[0], path + "[0]", depth + 1, out)
    else:
        v = re.sub(r"\s+", " ", str(x))
        print("{} = {}".format(path, v[:120] + ("…({}字)".format(len(v)) if len(v) > 120 else "")))


def do_json(u, eid):
    import json
    html = get(u)
    if not html:
        print("読めませんでした", u)
        return
    el = BeautifulSoup(html, "html.parser").find("script", id=eid)
    if not el:
        print("見つかりません", eid)
        return
    data = json.loads(el.string or el.get_text())
    print("##### JSON", u, eid, len(el.string or ""), "字")
    show_shape(data)
    print("##### おわり")


def do_raw(u):
    html = get(u)
    if not html:
        print("読めませんでした", u)
        return
    soup = BeautifulSoup(html, "html.parser")
    print("##### つくり", u, len(html), "字 題名:", soup.title.get_text(strip=True) if soup.title else "")
    for sc in soup.find_all("script"):
        t = sc.string or sc.get_text() or ""
        if sc.get("src"):
            print("SCRIPT src", sc["src"])
        elif len(t.strip()) > 0:
            print("SCRIPT id={} type={} {}字: {}".format(sc.get("id"), sc.get("type"), len(t), re.sub(r"\s+", " ", t.strip())[:400]))
    for i, tb in enumerate(soup.find_all("table")[:12]):
        rows = tb.find_all("tr")
        print("TABLE", i, len(rows), "行")
        for tr in rows[:4]:
            print("  | " + " | ".join(c.get_text(" ", strip=True)[:30] for c in tr.find_all(["th", "td"])))
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    tx = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    print("TEXT", len(tx), "字:", tx[:2500])
    print("##### おわり")


def do_probe(tmpl, rng):
    import datetime
    a, b = rng.split("-")
    d = datetime.date(int(a[:4]), int(a[4:6]), int(a[6:]))
    e = datetime.date(int(b[:4]), int(b[4:6]), int(b[6:]))
    found = []
    while d <= e:
        u = tmpl.replace("{d}", d.strftime("%Y%m%d"))
        try:
            r = S.get(u, timeout=30)
            if r.status_code == 200:
                found.append(d.strftime("%Y%m%d"))
                print("ある", u, len(r.content), flush=True)
        except requests.RequestException as ex:
            print("ERR", u, type(ex).__name__, flush=True)
        time.sleep(0.3)
        d += datetime.timedelta(days=1)
    print("##### ある日", len(found), found)


def do_links(u, pat):
    html = get(u)
    if not html:
        print("読めませんでした", u)
        return
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    for a in soup.find_all("a", href=True):
        v = orig(urljoin(addr(u), a["href"]).split("#")[0])
        t = a.get_text(" ", strip=True)
        if same_site(v, u) and v not in seen and re.search(pat, t + " " + v):
            seen.add(v)
            print("LINK", v, "|", t)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        do_list(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    elif sys.argv[1] == "json":
        for u in re.split(r"[\s,]+", sys.argv[2].strip()):
            if u:
                do_json(u, sys.argv[3] if len(sys.argv) > 3 else "")
    elif sys.argv[1] == "grep":
        for u in re.split(r"[\s,]+", sys.argv[2].strip()):
            if u:
                do_grep(u, sys.argv[3] if len(sys.argv) > 3 else ".")
    elif sys.argv[1] == "raw":
        for u in re.split(r"[\s,]+", sys.argv[2].strip()):
            if u:
                do_raw(u)
    elif sys.argv[1] == "probe":
        for t in re.split(r"[\s,]+", sys.argv[2].strip()):
            if t:
                do_probe(t, sys.argv[3])
    elif sys.argv[1] == "links":
        for u in re.split(r"[\s,]+", sys.argv[2].strip()):
            if u:
                do_links(u, sys.argv[3] if len(sys.argv) > 3 else ".")
    else:
        for u in re.split(r"[\s,]+", sys.argv[2].strip()):
            if u:
                do_read(u)

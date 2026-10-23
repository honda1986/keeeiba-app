// IPAT(スマホ版)で、買い目の「金額入力」の画面まで進める。アプリが 0.8 秒ごとに呼ぶ。
//   B … {venue, dow, rno, type, method, combo: [馬番(枠連は枠番)...], axis, aite}
//       cols … フォーメーションの列(1頭目・2頭目・3頭目、馬単・3連単は1着・2着・3着)。n … 点数(金額入力の組数と見比べる)
//       2頭以上の券種は、どの買い方も IPAT のフォーメーションで入れる
// 画面の文字で今どこかを見て、押す場所を1つだけ返す。押すのはアプリ(指で押したのと同じになる。
// スクリプトからの click() では IPAT の画面が進まないため)。金額・セット・投票は押さない(金額入力の画面で止まる)。
// 戻り値(JSON の文字列): {st: tap|wait|login|final|error|unknown, msg, x, y, info}
//   x, y … 押す場所(画面の幅・高さに対する割合)。info … うまく押せないときの、押そうとした所の形
(function (B, STARTED) {
  var txt = function (e) { return ((e.innerText || e.textContent || "") + "").replace(/\s+/g, " ").trim(); };
  var body = txt(document.body || document.documentElement);
  // page … 今の画面の名前(IPAT の画面の id)。止まったときに、どの画面だったか分かるように毎回付ける
  var out = function (st, msg, extra) {
    var pg = document.querySelector(".ui-page-active");
    var o = {st: st, msg: msg || "", page: (pg && (pg.id || pg.getAttribute("data-url"))) || location.pathname};
    for (var k in extra || {}) o[k] = extra[k];
    return JSON.stringify(o);
  };
  var visible = function (e) { return !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length); };
  var esc = function (s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); };
  var z2 = function (u) { return (u < 10 ? "0" : "") + u; };
  var CLICKABLE = "a,button,input,label,[onclick],[ng-click],[role=button],[data-role=button]";
  // 押したタッチが画面に届いたか(届いた所の形)を残す。うまく進まないときの手がかり
  if (!window.__kbHook) {
    window.__kbHook = 1;
    document.addEventListener("touchstart", function (e) {
      var t = e.touches && e.touches[0];
      window.__kbSaw = short(e.target) + (t ? "@" + Math.round(t.clientX) + "," + Math.round(t.clientY) : "");
    }, true);
  }

  // 文字が合う、見えている要素(文字の短い順。画像は alt の文字)
  function findAll(re) {
    var hits = [];
    var els = document.querySelectorAll("a,button,input[type=button],input[type=submit],li,td,th,span,div,p,label,img");
    for (var i = 0; i < els.length; i++) {
      var e = els[i];
      if (!visible(e)) continue;
      var t = e.tagName === "INPUT" ? (e.value || "").trim() : e.tagName === "IMG" ? (e.alt || "").trim() : txt(e);
      if (!t || t.length > 80) continue;
      if (re.test(t)) hits.push({e: e, n: t.length, i: i});
    }
    // 同じ文字の入れ子(li の中の a の中の span など)は、いちばん内側だけ
    hits = hits.filter(function (h) { return !hits.some(function (o) { return o !== h && o.n === h.n && h.e.contains(o.e); }); });
    hits.sort(function (a, b) { return a.n - b.n || a.i - b.i; });
    return hits.map(function (h) { return h.e; });
  }
  // 要素の真ん中が画面の中にあって、ほかの物に隠れていなければ、その場所(画面に対する割合)
  function point(el) {
    try { el.scrollIntoView({block: "center"}); } catch (x) { /* 無視 */ }
    var r = el.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    var cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx > innerWidth || cy > innerHeight) return null;
    var hit = document.elementFromPoint(cx, cy);
    if (!hit) return null;
    // 押す物そのもの・その中・それを囲む物、または同じ行(li・tr)の中の物(行全体に重ねて押せるようにした作りがある)
    var top = hit === document.body || hit === document.documentElement;
    var row = el.closest ? el.closest("li,tr") : null;
    var ok = hit === el || el.contains(hit) || (!top && hit.contains(el)) || (!top && row && row.contains(hit));
    if (!ok) return null;
    var vv = window.visualViewport;
    var ox = vv ? vv.offsetLeft : 0, oy = vv ? vv.offsetTop : 0;
    var w = vv ? vv.width : innerWidth, h = vv ? vv.height : innerHeight;
    var x = (cx - ox) / w, y = (cy - oy) / h;
    return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? {x: x, y: y} : null;
  }
  // 押してみる所(この順に試す): 押せる親(リンク・ボタンなど。小さいとき) → 文字そのもの → 同じ枠の画像 → 枠(li)
  function spots(e) {
    var list = [];
    var add = function (el) { if (el && el !== document.body && list.indexOf(el) < 0) list.push(el); };
    var c = e.closest ? e.closest(CLICKABLE) : null;
    if (c && c !== e) {
      var r = c.getBoundingClientRect();
      if (r.height < 260 && r.width <= innerWidth + 1) add(c);
    }
    add(e);
    var li = e.closest ? e.closest("li") : null;
    var box = li || c;
    if (box && box.querySelector) add(box.querySelector("img"));
    add(li);
    return list;
  }
  // 要素の短い形(タグ.class と、画面での位置・大きさ)
  function short(el) {
    if (!el || !el.tagName) return "?";
    var r = el.getBoundingClientRect ? el.getBoundingClientRect() : {left: 0, top: 0, width: 0, height: 0};
    return el.tagName.toLowerCase() + (el.className && typeof el.className === "string" ? "." + el.className.trim().split(/\s+/).slice(0, 3).join(".") : "") +
      "(" + Math.round(r.left) + "," + Math.round(r.top) + "," + Math.round(r.width) + "x" + Math.round(r.height) + ")";
  }
  // 押そうとした所の形(タグ・class・属性だけ。画面の文字やURLは入れない)。うまく進まないときに送ってもらう
  function describe(el) {
    var s = [];
    for (var i = 0; el && i < 6 && el !== document.body; i++, el = el.parentElement) {
      var a = [];
      for (var j = 0; j < el.attributes.length; j++) {
        var at = el.attributes[j], v = String(at.value);
        if (at.name === "href") a.push("href=" + (/^(#|javascript)/.test(v) ? v.slice(0, 30) : "(url)"));
        else if (/^(id|class|role|type|for|name|onclick|ng-click|ontouch\w+|data-[\w-]+)$/.test(at.name))
          a.push(at.name + "=" + v.slice(0, 40));
      }
      s.push(el.tagName.toLowerCase() + (a.length ? "[" + a.join(" ") + "]" : ""));
    }
    return s.join(" < ").slice(0, 1200);
  }
  // cands の中で押せる(画面の中にあって隠れていない)ものの、n 回目なら n 番目を押す(前の所で進まなかったら次を試す)
  function tapAny(cands, n, msg) {
    var ok = cands.filter(function (el) { return !!point(el); });
    if (!ok.length) return null;
    var el = ok[(n - 1) % ok.length], p = point(el);
    if (!p) return null;
    window.__kbInfo = n + "回目 " + location.pathname + " | 押した所 " + short(el) + " 候補" + ok.length + "/" + cands.length +
      " | 画面 " + innerWidth + "x" + innerHeight + " | " + describe(el);
    return out("tap", msg, {x: p.x, y: p.y, info: window.__kbInfo});
  }
  // 同じ画面で続けて押さない(画面が変わるまで 4 秒は待つ)。同じボタンを続けて4回(押す所を変えて)押しても変わらなければ止める
  function once(key, fn) {
    if (window.__kbKey === key && Date.now() - (window.__kbAt || 0) < 4000) return out("wait", "画面が変わるのを待っています");
    var n = window.__kbN = window.__kbKey === key ? (window.__kbN || 0) + 1 : 1;
    if (n > 4)
      return out("error", "押す所を変えて4回押しても画面が変わりません。手で押して進むか確かめてください",
        {info: (window.__kbInfo || "") + " | 届いたタッチ " + (window.__kbSaw || "なし")});
    window.__kbKey = key; window.__kbAt = Date.now();
    window.__kbSaw = "";
    return fn(n);
  }
  // 押せる所が無い(画面が切り替わる途中など)ときは少し待つ。6秒ほど待っても無ければ止める
  function missWait(key, msg, el) {
    window.__kbMiss = window.__kbMissKey === key ? (window.__kbMiss || 0) + 1 : 1;
    window.__kbMissKey = key;
    if (window.__kbMiss < 8) return out("wait", msg + " を押せるようになるのを待っています");
    var r = el.getBoundingClientRect();
    var hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return out("error", msg + " を押せる場所が見つかりません。手で続けてください", {info: short(el) + " | 真ん中にある物 " +
      (hit ? short(hit) + " " + describe(hit) : "なし") + " | 画面 " + innerWidth + "x" + innerHeight + " | " + describe(el)});
  }
  // 画面(ページ)の名前と見出し。切り替わったばかり(1秒以内・切り替えの動きの途中)なら押さずに待つ
  function pageEl() { return document.querySelector(".ui-page-active") || document.body; }
  function heading() {
    var hd = pageEl().querySelector("[data-role=header], .ui-header");
    if (!hd) return "";
    var c = hd.cloneNode(true);
    Array.prototype.forEach.call(c.querySelectorAll("a,button"), function (x) { x.remove(); });   // 「方式」「1頭目」の戻るボタンは除く
    return txt(c);
  }
  function settled() {
    var key = (pageEl().id || location.pathname) + "|" + heading();
    if (window.__kbPk !== key) { window.__kbPk = key; window.__kbPkAt = Date.now(); }
    var moving = /ui-mobile-viewport-transitioning/.test(document.documentElement.className + " " + document.body.className);
    return !moving && Date.now() - window.__kbPkAt > 1000;
  }
  function press(key, cands, msg, el) {
    if (!settled()) return out("wait", "画面が切り替わるのを待っています");
    if (!cands.some(function (c) { return !!point(c); })) return missWait(key, msg, el);
    return once(key, function (n) { return tapAny(cands, n, msg) || missWait(key, msg, el); });
  }
  // 文字に loose が入っている、見えている短い物(見つからないときの手がかり。最大6つ)
  function near(loose) {
    var xs = [], els = document.querySelectorAll("a,button,input[type=button],input[type=submit],li,td,span,p,label");
    for (var i = 0; i < els.length && xs.length < 6; i++) {
      var e = els[i], t = e.tagName === "INPUT" ? (e.value || "") : txt(e);
      if (visible(e) && t.length <= 40 && loose.test(t)) xs.push(short(e) + "「" + t + "」");
    }
    return xs.join(" / ") || "なし";
  }
  // 見えているボタン・リンクの文字(分からない画面の手がかり。最大12個)
  function buttons() {
    var xs = [], els = document.querySelectorAll("a,button,input[type=button],input[type=submit]");
    for (var i = 0; i < els.length && xs.length < 12; i++) {
      var e = els[i], t = e.tagName === "INPUT" ? (e.value || "") : txt(e);
      if (visible(e) && t) xs.push(t.slice(0, 20));
    }
    return xs.join(" / ") || "なし";
  }
  function clickText(key, re, msg, loose) {
    var es = findAll(re);
    if (!es.length) return out("error", msg + " が見つかりません。手で続けてください",
      {info: loose ? "似た所 " + near(loose) : "ボタン " + buttons()});
    var cands = [];
    es.forEach(function (e) { spots(e).forEach(function (el) { if (cands.indexOf(el) < 0) cands.push(el); }); });
    return press(key, cands, msg, es[0]);
  }
  // 馬番(枠番)の枠(span.waku6 など)と、その行。IPAT は行を押して選ぶ(チェックの input は無い)
  function rowFor(u) {
    var cells = Array.prototype.filter.call(pageEl().querySelectorAll("span,div,p,td,b,strong,em,label"), function (e) {
      var t = txt(e);
      return visible(e) && e.children.length === 0 && (t === String(u) || t === z2(u));
    });
    var score = function (e) { return (/waku|umaban|num/i.test(e.className || "") ? 2 : 0) + (e.closest && e.closest("li") ? 1 : 0); };
    cells.sort(function (a, b) { return score(b) - score(a); });
    var c = cells[0];
    return c ? {num: c, row: (c.closest && c.closest("li,tr")) || c.parentElement} : null;
  }
  // 行の形(タグ・class・画像・選択の属性。文字と、枠の色などの数字の入った class、押した瞬間だけの class は除く)。
  // 押す前と比べて変わったら、選ぶ・外すが入れ替わった(=押せた)ことにする
  function shape(row) {
    var els = [row].concat(Array.prototype.slice.call(row.querySelectorAll("*")));
    return els.map(function (e) {
      var cls = (typeof e.className === "string" ? e.className : "").split(/\s+/).filter(function (c) {
        return c && !/\d/.test(c) && !/hover|down|focus/i.test(c);
      }).sort().join(".");
      var x = e.tagName === "IMG" ? (e.getAttribute("src") || "").split("/").pop().replace(/[?#].*/, "") : "";
      ["aria-checked", "aria-selected", "aria-pressed", "data-checked", "data-selected", "data-status"].forEach(function (a) {
        if (e.hasAttribute(a)) x += a + "=" + e.getAttribute(a);
      });
      return e.tagName + "." + cls + x;
    }).join("|");
  }
  // 行の中で押してみる所(この順): 右のチェックの四角 → 行のリンク → 行の真ん中 → 馬番の枠
  function rowSpots(rf) {
    var rr = rf.row.getBoundingClientRect(), box = null;
    Array.prototype.forEach.call(rf.row.querySelectorAll("*"), function (e) {
      var r = e.getBoundingClientRect();
      var square = r.width >= 14 && r.width <= 64 && Math.abs(r.width - r.height) <= 8 && r.right > rr.right - 90;
      if (!box && (square || /check|chk|cbox/i.test(typeof e.className === "string" ? e.className : "")) && visible(e)) box = e;
    });
    return [box, rf.row.querySelector("a[data-value]") || rf.row.querySelector("a"), rf.row, rf.num]
      .filter(function (el, i, a) { return el && a.indexOf(el) === i; });
  }

  var NEED = {"単勝": 1, "複勝": 1, "枠連": 2, "馬連": 2, "ワイド": 2, "馬単": 2, "3連複": 3, "3連単": 3};
  var K = NEED[B.type] || B.combo.length;
  var ORD = /^(馬単|3連単)$/.test(B.type);
  // IPAT には、2頭以上の券種はどの買い方もフォーメーションで入れる(通常=1頭ずつ、ボックス=全部の列に全部、ながし=軸と相手)
  var COLS = B.cols && B.cols.length ? B.cols
    : K === 1 ? [B.combo]
    : B.method === "ボックス" ? Array.apply(null, Array(K)).map(function () { return B.combo; })
    : B.method === "ながし" ? [[B.axis]].concat(Array.apply(null, Array(K - 1)).map(function () { return B.aite || []; }))
    : B.combo.map(function (u) { return [u]; });
  var HOUSHIKI = K >= 2 ? "フォーメーション" : "通常";
  var colName = function (i) { return K === 1 ? "" : (i + 1) + (ORD ? "着" : "頭目"); };
  var label = B.venue + "(" + B.dow + ") " + B.rno + "R " + B.type + " " +
    COLS.map(function (c) { return c.join(","); }).join(ORD ? "→" : " / ");
  // 金額入力の画面で、見出し(「1頭目」など)と同じ高さの、右にある数字だけの枠(馬番)。枠が並ぶと文字がつながる(379)ので位置で見る
  function shownNums(labelRe) {
    var labs = findAll(labelRe);
    if (!labs.length) return null;
    var lr = labs[0].getBoundingClientRect();
    var nums = Array.prototype.filter.call(pageEl().querySelectorAll("*"), function (e) {
      if (e.children.length || !visible(e) || !/^\d{1,2}$/.test(txt(e))) return false;
      var r = e.getBoundingClientRect(), cy = r.top + r.height / 2;
      return cy >= lr.top - 8 && cy <= lr.bottom + 8 && r.left >= lr.right - 2;
    }).map(function (e) { return +txt(e); });
    return nums.length ? nums : null;
  }
  var sortN = function (xs) { return xs.slice().sort(function (a, b) { return a - b; }).join(","); };

  if (!STARTED) window.__kbSel = {};          // 新しい買い目: 前に選んだ記録は使わない

  // 金額入力: ここで止まる(金額・セットは自分で)。画面の列の馬番と組数が買い目と合っているかを見る
  if (/金額入力/.test(body) && /セット/.test(body) && !/合計金額入力/.test(body)) {
    var bad = [];
    if (K >= 2) {
      COLS.forEach(function (c, i) {
        var shown = shownNums(new RegExp("^" + (i + 1) + "\\s*(頭目|着)$"));
        if (shown && sortN(shown) !== sortN(c)) bad.push(colName(i) + " " + shown.join(",") + "(買い目は " + c.join(",") + ")");
      });
    } else {
      var one = shownNums(/^(馬番|枠番)$/);
      if (one && sortN(one) !== sortN(B.combo)) bad.push("馬番 " + one.join(","));
    }
    var km = body.match(/組数\s*(\d+)/);
    if (B.n && km && +km[1] !== B.n) bad.push("組数 " + km[1] + "(買い目は " + B.n + "点)");
    return out("final", bad.length ? "画面の内容が買い目と違います: " + bad.join("、") + "。直してから金額を入れてください"
      : "金額入力まで進めました" + (km ? "(組数 " + km[1] + ")" : "") + "。金額を入れて「セット」を押してください");
  }
  if (/合計金額入力/.test(body)) return out("final", "合計金額の画面です。内容を確かめて自分で入れてください");
  // ログイン
  if (document.querySelector("input[type=password]") && /加入者番号|暗証番号|INET/.test(body))
    return out("login", "ログインしてください(ログインすると続きを自動で進めます)");
  // 投票一覧(前の買い目をセットしたあと): 続けて入れる
  if (/投票一覧/.test(body) && /場名から続けて入力/.test(body)) return clickText("list", /^場名から続けて入力$/, "場名から続けて入力");
  // 馬番(枠番)を選ぶ画面(フォーメーションの 1頭目・2頭目・3頭目 / 1着・2着・3着、単勝・複勝は1画面)
  if (/(金額入力画面へ|次へ)/.test(body) && /(全選択|選択してください)/.test(body)) {
    var hm = heading().match(/([1-3])\s*(?:頭目|着)/);
    var ci = hm ? +hm[1] - 1 : 0, want = COLS[ci] || [];
    var name = B.type === "枠連" ? "枠番 " : "馬番 ", miss = [];
    if (!settled()) return out("wait", "画面が切り替わるのを待っています");
    // 行は押すたびに選ぶ・外すが入れ替わる。1頭ずつ押して、行の形が変わったら選べたことにする。
    // 変わらなければ(押せていない)押す所を変えて押し直す(4か所まで)
    for (var j = 0; j < want.length; j++) {
      var v = want[j], rf = rowFor(v), key = ci + ":" + v, d = window.__kbSel[key], wt = colName(ci) + " " + name + v + " を選ぶ";
      if (!rf) { miss.push(v); continue; }
      if (d && d.ok) continue;
      var cands = rowSpots(rf);
      if (d) {
        if (shape(rf.row) !== d.shape) { d.ok = true; continue; }
        if (Date.now() - d.at < 2500) return out("wait", colName(ci) + " " + name + v + " が選ばれるのを待っています");
        if (d.n >= cands.length) {
          var kids = Array.prototype.slice.call(rf.row.querySelectorAll("*"), 0, 10).map(short).join(" ");
          return out("error", colName(ci) + " " + name + v + " を押しても選べませんでした。手で選んでください",
            {info: (window.__kbInfo || "") + " | 届いたタッチ " + (window.__kbSaw || "なし") + " | 行 " + describe(rf.row) + " | 行の中 " + kids});
        }
      }
      var n = d ? d.n + 1 : 1, before = shape(rf.row), res = tapAny(cands, n, wt);
      if (!res) return missWait("row" + key, wt, rf.num);
      window.__kbSaw = "";
      window.__kbSel[key] = {shape: before, at: Date.now(), n: n};
      return res;
    }
    if (miss.length) return out("error", colName(ci) + " " + name + miss.join("・") + " が見つかりません。手で選んでください",
      {info: "似た所 " + near(new RegExp("^0?" + miss[0] + "$"))});
    if (ci < COLS.length - 1 && findAll(/^次へ$/).length) return clickText("next" + ci, /^次へ$/, colName(ci) + "を選びました。次へ");
    return clickText("next" + ci, /^金額入力画面へ$/, "金額入力画面へ");
  }
  // 方式: 2頭以上の券種はフォーメーション
  if (/フォーメーション/.test(body) && /ボックス/.test(body)) return clickText("houshiki", new RegExp("^" + HOUSHIKI + "$"), "方式: " + HOUSHIKI);
  // 式別(3連複・3連単は全角の数字のこともある)
  if (/枠連/.test(body) && /(3|３)連単/.test(body)) {
    var t = B.type.replace(/^3/, "[3３]");
    return clickText("shiki", new RegExp("^" + t + "$"), "式別: " + B.type, new RegExp(t));
  }
  // レース
  if (/締切時刻/.test(body)) {
    var vh = new RegExp(esc(B.venue) + "\\s*[(（]" + esc(B.dow) + "[)）]");
    if (!vh.test(body)) return clickText("back", /^競馬場名$/, "競馬場を選び直します");
    // 「11R」「11R 15:30」「11R15:30」「１１Ｒ」など(1R が 11R に合わないよう、数字のすぐ後に R)
    var zen = String(B.rno).replace(/\d/g, function (d) { return String.fromCharCode(0xFF10 + +d); });
    return clickText("race", new RegExp("^\\s*(" + B.rno + "|" + zen + ")\\s*[RＲ]"), B.rno + "R", /[RＲ]/);
  }
  // 競馬場
  if (/競馬場名/.test(body)) {
    var vre = new RegExp("^" + esc(B.venue) + "\\s*[(（]" + esc(B.dow) + "[)）]$");
    if (!findAll(vre).length) return out("error", B.venue + "(" + B.dow + ") がまだ発売されていません(または見つかりません)",
      {info: "似た所 " + near(new RegExp(esc(B.venue) + "|[(（][土日月][)）]"))});
    return clickText("venue", vre, B.venue + "(" + B.dow + ")");
  }
  // 購入限度額が 0 のときの注意: このまま進む(入金は自分で)
  if (/ご注意/.test(body) && /このまま進む/.test(body))
    return clickText("caution", /^このまま進む$/, "購入限度額が0円です(投票の前に入金が必要)。入力は進めます");
  // トップメニュー
  if (/通常投票/.test(body) && /照会メニュー/.test(body)) return clickText("top", /^通常投票$/, "通常投票");
  if (!STARTED && findAll(/^トップメニュー$/).length) return clickText("tomenu", /^トップメニュー$/, "トップメニューに戻ります");
  return out("unknown", "この画面は分かりません。トップメニューに戻ると続きを進めます(" + label + ")", {info: "ボタン " + buttons()});
})

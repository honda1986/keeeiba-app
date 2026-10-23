// IPAT(スマホ版)で、買い目の「金額入力」の画面まで進める。アプリが 0.8 秒ごとに呼ぶ。
//   B … {venue, dow, rno, type, method, combo: [馬番(枠連は枠番)...], axis, aite}
//       method = 通常 / ボックス / ながし / フォーメーション。ながし・フォーメーションは馬番を選ぶ画面で止まる
// 画面の文字で今どこかを見て、1回に1つだけ押す。金額・セット・投票は押さない(金額入力の画面で止まる)。
// 戻り値(JSON の文字列): {st: step|wait|login|final|error|unknown, msg}
(function (B, STARTED) {
  var txt = function (e) { return ((e.innerText || e.textContent || "") + "").replace(/\s+/g, " ").trim(); };
  var body = txt(document.body || document.documentElement);
  var out = function (st, msg) { return JSON.stringify({st: st, msg: msg || ""}); };
  var visible = function (e) { return !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length); };
  var esc = function (s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); };
  var z2 = function (u) { return (u < 10 ? "0" : "") + u; };
  // 文字が合う、いちばん小さい(内側の)見えている要素
  function find(re) {
    var best = null, bl = 1e9;
    var els = document.querySelectorAll("a,button,input[type=button],input[type=submit],li,td,th,span,div,p,label");
    for (var i = 0; i < els.length; i++) {
      var e = els[i];
      if (!visible(e)) continue;
      var t = e.tagName === "INPUT" ? (e.value || "").trim() : txt(e);
      if (!t || t.length > 80) continue;
      if (re.test(t) && t.length < bl) { best = e; bl = t.length; }
    }
    return best;
  }
  function tap(e) {
    var c = e.closest ? (e.closest("a,button,input,li,[onclick],[ng-click],[role=button]") || e) : e;
    try { c.scrollIntoView({block: "center"}); } catch (x) { /* 無視 */ }
    c.click();
  }
  // 同じ画面で続けて押さない(画面が変わるまで 4 秒は待つ)
  function once(key, fn) {
    if (window.__kbKey === key && Date.now() - (window.__kbAt || 0) < 4000) return out("wait", "画面が変わるのを待っています");
    window.__kbKey = key; window.__kbAt = Date.now();
    return fn();
  }
  function clickText(key, re, msg) {
    return once(key, function () {
      var e = find(re);
      if (!e) return out("error", msg + " が見つかりません。手で続けてください");
      tap(e);
      return out("step", msg);
    });
  }
  var ORDERED = /^(馬単|3連単|３連単)$/.test(B.type) && B.method === "通常";
  var label = B.venue + "(" + B.dow + ") " + B.rno + "R " + B.type + " " + B.method + " " +
    (B.method === "ながし" ? "軸" + B.axis + " 相手" + (B.aite || []).join(",") : B.combo.join(ORDERED ? "→" : "-"));

  // 金額入力: ここで止まる(金額・セットは自分で)。画面の馬番が合っているかだけ見る
  if (/金額入力/.test(body) && /セット/.test(body) && !/合計金額入力/.test(body)) {
    var m = body.match(/馬番\s*([0-9０-９\s]+?)\s*金額/);
    var shown = m ? (m[1].match(/\d+/g) || []).map(Number) : null;
    var want = B.combo.slice();
    var same = !shown || B.method !== "通常" ||
      (ORDERED ? shown.join("-") === want.join("-") : shown.slice().sort(function (a, b) { return a - b; }).join("-") ===
        want.slice().sort(function (a, b) { return a - b; }).join("-"));
    return out("final", same ? "金額入力まで進めました。金額を入れて「セット」を押してください"
      : "画面の馬番(" + shown.join("-") + ")が買い目と違います。直してから金額を入れてください");
  }
  if (/合計金額入力/.test(body)) return out("final", "合計金額の画面です。内容を確かめて自分で入れてください");
  // ログイン
  if (document.querySelector("input[type=password]") && /加入者番号|暗証番号|INET/.test(body))
    return out("login", "ログインしてください(ログインすると続きを自動で進めます)");
  // 投票一覧(前の買い目をセットしたあと): 続けて入れる
  if (/投票一覧/.test(body) && /場名から続けて入力/.test(body)) return clickText("list", /^場名から続けて入力$/, "場名から続けて入力");
  // 馬番(枠番)を選ぶ
  if (/選択してください/.test(body) && /金額入力画面へ/.test(body)) {
    if (B.method === "ながし" || B.method === "フォーメーション")
      return out("final", B.method + "の画面まで進めました。馬番を選んで「金額入力画面へ」を押してください");
    return once("horses", function () {
      var miss = [];
      B.combo.forEach(function (u, pos) {
        var cells = Array.prototype.filter.call(document.querySelectorAll("td,th,span,div,p,label,b,strong"), function (e) {
          var t = txt(e);
          return visible(e) && e.children.length === 0 && (t === String(u) || t === z2(u));
        });
        var done = false;
        for (var k = 0; k < cells.length && !done; k++) {
          var r = cells[k];
          for (var up = 0; up < 6 && r && !done; up++) {
            r = r.parentElement;
            if (!r) break;
            var cbs = r.querySelectorAll("input[type=checkbox]");
            if (!cbs.length) continue;
            // 馬単・3連単の通常は 1着・2着・3着 の列。それ以外は1列
            var cb = cbs[ORDERED ? pos : 0];
            if (!cb) break;
            if (!cb.checked) {
              var lab = cb.id ? document.querySelector("label[for='" + cb.id + "']") : null;
              (lab || cb).click();
            }
            done = true;
          }
        }
        if (!done) miss.push(u);
      });
      if (miss.length) return out("error", (B.type === "枠連" ? "枠番 " : "馬番 ") + miss.join("・") + " を選べませんでした。手で選んでください");
      setTimeout(function () { var e = find(/^金額入力画面へ$/); if (e) tap(e); }, 700);
      return out("step", B.combo.join(ORDERED ? "→" : "-") + " を選びました");
    });
  }
  // 方式
  if (/フォーメーション/.test(body) && /ボックス/.test(body)) return clickText("houshiki", new RegExp("^" + esc(B.method) + "$"), "方式: " + B.method);
  // 式別(3連複・3連単は全角の数字のこともある)
  if (/枠連/.test(body) && /(3|３)連単/.test(body)) {
    var t = B.type.replace(/^3/, "[3３]");
    return clickText("shiki", new RegExp("^" + t + "$"), "式別: " + B.type);
  }
  // レース
  if (/締切時刻/.test(body)) {
    var vh = new RegExp(esc(B.venue) + "\\s*[(（]" + esc(B.dow) + "[)）]");
    if (!vh.test(body)) return clickText("back", /^競馬場名$/, "競馬場を選び直します");
    return clickText("race", new RegExp("^" + B.rno + "R(\\s|$)"), B.rno + "R");
  }
  // 競馬場
  if (/競馬場名/.test(body)) {
    var vre = new RegExp("^" + esc(B.venue) + "\\s*[(（]" + esc(B.dow) + "[)）]$");
    if (!find(vre)) return out("error", B.venue + "(" + B.dow + ") がまだ発売されていません(または見つかりません)");
    return clickText("venue", vre, B.venue + "(" + B.dow + ")");
  }
  // 購入限度額が 0 のときの注意: このまま進む(入金は自分で)
  if (/ご注意/.test(body) && /このまま進む/.test(body))
    return clickText("caution", /^このまま進む$/, "購入限度額が0円です(投票の前に入金が必要)。入力は進めます");
  // トップメニュー
  if (/通常投票/.test(body) && /照会メニュー/.test(body)) return clickText("top", /^通常投票$/, "通常投票");
  if (!STARTED && find(/^トップメニュー$/)) return clickText("tomenu", /^トップメニュー$/, "トップメニューに戻ります");
  return out("unknown", "この画面は分かりません。トップメニューに戻ると続きを進めます(" + label + ")");
})

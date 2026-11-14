// ============================================================
// odds.js — 発走が近い中央のレースのオッズ(単勝・ワイド・3連複)を netkeiba から取る(PC 用)
//
// GitHub の時刻指定は 3〜8時間遅れることがあり、開催日にオッズが古いままになる(2026-11-14 は朝の更新が昼まで始まらなかった)。
// 競輪(honda1986/keirin の pc/)と同じく、ふだんは PC のタスクスケジューラが1分おきにこれを呼び、
// 取ったオッズを odds-live ブランチの latest.json に置く(pc/runner.js が送る)。アプリはそれを読む。
//
// ・レースの一覧(発走時刻・◎○)は GitHub Pages の keiba/days/YYYYMMDD.json から(10分おきに取り直す)
// ・発走まで 35分〜0分のレースを見る。12分前からは毎回(1分おき)、それより前は3分おき
// ・取るもの: 単勝(type=1)・ワイド(type=5)・3連複の全部の組(type=7)。keiba.py の fetch_odds / fetch_wide_odds / fetch_trio_all と同じ API
// ・3連複は、2頭の組ごとの強さ(その2頭を含む組の 1/オッズ の合計)にして残す(keiba.py の trio_pair_sums と同じ。割安ワイドに使う)
// ・出すもの(--out のフォルダ):
//     latest.json        … これからのレースと、発走から20分以内のレースの、いちばん新しいオッズ(アプリが読む)
//     snap/YYYYMMDD.json … その日の全部のレースの、発走前の最後のオッズ(GitHub の更新が取り込んで記録に残す)
//
// 使い方: node pc/odds.js --out <フォルダ>       … 1回だけ見る(タスクスケジューラが1分おきに呼ぶ)
//         node pc/odds.js --dry                  … 取って表示するだけ(書かない)
//         node pc/odds.js --dry --win=-600,600   … 見る範囲を変える(動作確認用。発走まで何分)
// 終わりの印(最後の行): "changed=1" 新しいオッズを書いた / "changed=0" 変わらない / "noraces=1" 今日は中央の開催が無い
// ============================================================
"use strict";
const fs = require("fs");
const path = require("path");

// (KEIBA_DAYS_URL・KEIBA_ODDS_API は動作確認用。ふだんは設定しない)
const PAGES = process.env.KEIBA_DAYS_URL || "https://honda1986.github.io/keeeiba-app/keiba/days/";
const RAW = "https://raw.githubusercontent.com/honda1986/keeeiba-app/main/keiba/days/";
const API = process.env.KEIBA_ODDS_API || "https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&input=UTF-8&output=json&action=init&compress=0";
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";
const LIST_EVERY_MIN = 10;      // レースの一覧を取り直す間隔
const NEAR_MIN = 12, NEAR_SEC = 50, FAR_SEC = 170;   // 発走12分前からは毎回、それより前は3分おき
const KEEP_AFTER_MIN = 20;      // latest.json には、発走から20分たつまで残す
const WAIT_MS = 400;            // 1リクエストごとの間隔(並べて取らない)
const BUDGET_MS = 45 * 1000;    // 1回の持ち時間。超えたら残りは次の回へ
const TRIO_PAIR_MIN = 0.003;    // keiba.py と同じ(これより小さい組は残さない)

// ---- 読み取り(keiba.py の parse_* と同じ形にする) ----
const num = (v) => { const x = parseFloat(String(v == null ? "" : v).replace(/,/g, "")); return isFinite(x) ? x : 0; };
// 単勝: 馬番 -> [オッズ, 人気]
function parseWin(j) {
  const out = {};
  const o = (((j || {}).data || {}).odds || {})["1"] || {};
  for (const k of Object.keys(o)) {
    const od = num(o[k][0]), nk = parseInt(o[k][2], 10);
    if (od > 0) out[String(parseInt(k, 10))] = [od, nk || null];
  }
  const us = Object.keys(out);
  if (us.length && !us.every((u) => out[u][1])) us.sort((a, b) => out[a][0] - out[b][0]).forEach((u, i) => { out[u][1] = i + 1; });
  return out;
}
// ワイド: "小さい馬番-大きい馬番" -> [下限, 上限]
function parseWide(j) {
  const out = {};
  const o = (((j || {}).data || {}).odds || {})["5"] || {};
  for (const k of Object.keys(o)) {
    const a = parseInt(k.slice(0, 2), 10), b = parseInt(k.slice(2, 4), 10), lo = num(o[k][0]), hi = num(o[k][1]);
    if (lo > 0 && a && b) out[Math.min(a, b) + "-" + Math.max(a, b)] = [lo, hi > 0 ? hi : lo];
  }
  return out;
}
// 3連複: "小さい順の馬番-…" -> オッズ
function parseTrio(j) {
  const out = {};
  const o = (((j || {}).data || {}).odds || {})["7"] || {};
  for (const k of Object.keys(o)) {
    const us = [0, 2, 4].map((i) => parseInt(k.slice(i, i + 2), 10)).sort((a, b) => a - b), od = num(o[k][0]);
    if (od > 0 && us.every(Boolean)) out[us.join("-")] = od;
  }
  return out;
}
// 3連複の全部の組 -> 2頭の組ごとの強さ(keiba.py の trio_pair_sums・アプリの trioPairSums と同じ)
function pairSums(trio) {
  const s = {};
  for (const k of Object.keys(trio)) {
    const us = k.split("-"), o = trio[k];
    if (us.length !== 3 || !(o > 1)) continue;
    for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) { const key = us[a] + "-" + us[b]; s[key] = (s[key] || 0) + 1 / o; }
  }
  const out = {};
  for (const k of Object.keys(s)) if (s[k] >= TRIO_PAIR_MIN) out[k] = Math.round(s[k] * 1e5) / 1e5;
  return out;
}

// ---- 時刻(日本時間) ----
const jstNow = (ms) => new Date((ms == null ? Date.now() : ms) + 9 * 3600e3);
const day8 = (ms) => jstNow(ms).toISOString().slice(0, 10).replace(/-/g, "");
const isoJst = (ms) => jstNow(ms).toISOString().slice(0, 19) + "+09:00";
// その日の HH:MM(日本時間)の時刻(ミリ秒)
const startMs = (day, hhmm) => Date.UTC(+day.slice(0, 4), +day.slice(4, 6) - 1, +day.slice(6, 8), +hhmm.slice(0, 2) - 9, +hhmm.slice(3, 5));

// 見るレース: 発走まで lo〜hi 分。前に見てから 間隔 がたっているもの
function dueRaces(list, day, now, last, win) {
  const [lo, hi] = win;
  return list.filter((r) => {
    const m = (startMs(day, r.start) - now) / 60000;
    if (!(m >= lo && m <= hi)) return false;
    const every = m <= NEAR_MIN ? NEAR_SEC : FAR_SEC;
    return !last[r.id] || (now - last[r.id]) / 1000 >= every;
  }).sort((a, b) => a.start.localeCompare(b.start));
}
// latest.json に載せるレース: これからのレースと、発走から KEEP_AFTER_MIN 分以内
function latestOf(snap, list, day, now) {
  const out = {};
  for (const r of list) {
    const m = (startMs(day, r.start) - now) / 60000;
    if (snap[r.id] && m >= -KEEP_AFTER_MIN) out[r.id] = snap[r.id];
  }
  return out;
}

module.exports = { parseWin, parseWide, parseTrio, pairSums, dueRaces, latestOf, startMs, isoJst, day8 };

// ---- ここから実行 ----
if (require.main === module) {
  const argv = process.argv.slice(2);
  const arg = (name) => { const a = argv.find((x) => x.startsWith("--" + name + "=")); return a ? a.slice(name.length + 3) : ""; };
  const DRY = argv.includes("--dry");
  const OUT = arg("out") || (argv.includes("--out") ? argv[argv.indexOf("--out") + 1] : "");
  const WORK = arg("work") || path.join(__dirname, "..", "..", "work");
  const WIN = arg("win") ? arg("win").split(",").map(Number) : [0, 35];
  const t0 = Date.now();
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const readJson = (f, d) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { return d; } };
  const writeJson = (f, o) => { fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, JSON.stringify(o)); };

  async function getJson(url) {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), 15000);
    try {
      const r = await fetch(url, { headers: { "User-Agent": UA }, signal: ctl.signal });
      if (!r.ok) throw new Error("HTTP " + r.status);
      return await r.json();
    } finally { clearTimeout(t); }
  }

  // レースの一覧(中央だけ)。work/races-YYYYMMDD.json に覚えておく
  async function loadList(day) {
    const f = path.join(WORK, "races-" + day + ".json");
    const c = readJson(f, null);
    if (c && (Date.now() - c.at) / 60000 < LIST_EVERY_MIN) return c.list;
    let j = null, why = "";
    for (const base of [PAGES, RAW]) {
      try { j = await getJson(base + day + ".json?t=" + Date.now()); break; } catch (e) { why = e.message; }
    }
    if (!j) {
      if (c) { console.log("レースの一覧を取り直せず(前のものを使う): " + why); return c.list; }
      console.log("レースの一覧が取れない(" + why + ")。今日の出馬表がまだ無いか、ネットにつながっていない");
      return null;
    }
    const list = Object.values(j.races || {}).filter((r) => r.center === "jra" && r.venue && r.start && r.id)
      .map((r) => { const p = r.pred || {}; return { id: r.id, venue: r.venue, rno: r.rno, start: r.start,
        pair: ["A", "B"].includes(p.verdict) && p.honmei && p.taikou ? [p.honmei, p.taikou] : null }; });
    if (!DRY) writeJson(f, { at: Date.now(), list });
    return list;
  }

  (async () => {
    const now = Date.now(), day = arg("day") || day8(now);
    const list = await loadList(day);
    if (list == null) { console.log("changed=0"); return; }
    if (!list.length) { console.log(day + " は中央の開催が無い"); console.log("noraces=1"); return; }
    const stF = path.join(WORK, "odds-state.json"), snF = path.join(WORK, "snap-" + day + ".json");
    const st = readJson(stF, {}), last = st.day === day ? st.last || {} : {};
    const snap = readJson(snF, {});
    const due = dueRaces(list, day, now, last, WIN);
    let got = 0;
    for (const r of due) {
      if (Date.now() - t0 > BUDGET_MS) { console.log("持ち時間切れ。残りは次の回へ"); break; }
      try {
        const win = parseWin(await getJson(API + "&race_id=" + r.id + "&type=1")); await sleep(WAIT_MS);
        const wide = parseWide(await getJson(API + "&race_id=" + r.id + "&type=5")); await sleep(WAIT_MS);
        const trio = parseTrio(await getJson(API + "&race_id=" + r.id + "&type=7")); await sleep(WAIT_MS);
        last[r.id] = Date.now();
        if (!Object.keys(win).length || !Object.keys(wide).length) { console.log(r.venue + r.rno + "R: オッズがまだ無い"); continue; }
        const s = { at: isoJst(Date.now()), start: r.start, odds: win, wide, tp: pairSums(trio) };
        if (r.pair) {      // モデルが買うレースは、3連複◎-○-x のオッズも(買い目のオッズとして出す)
          s.trio = {};
          for (const k of Object.keys(trio)) { const us = k.split("-").map(Number); if (r.pair.every((u) => us.includes(u))) s.trio[k] = trio[k]; }
        }
        snap[r.id] = s; got++;
        const m = Math.round((startMs(day, r.start) - Date.now()) / 60000);
        console.log(r.venue + r.rno + "R(発走まで" + m + "分): 単勝 " + Object.keys(win).length + "頭・ワイド " + Object.keys(wide).length + "組・3連複 " + Object.keys(trio).length + "組");
      } catch (e) { console.log(r.venue + r.rno + "R: 取れない(" + e.message + ")"); }
    }
    if (DRY) { console.log("見るレース " + due.length + " / 取れた " + got + "(--dry なので書かない)"); console.log("changed=0"); return; }
    writeJson(stF, { day, last });
    if (got) writeJson(snF, snap);
    if (OUT && got) {
      writeJson(path.join(OUT, "latest.json"), { updated: isoJst(Date.now()), date: day, source: "PC", races: latestOf(snap, list, day, Date.now()) });
      writeJson(path.join(OUT, "snap", day + ".json"), { updated: isoJst(Date.now()), date: day, source: "PC", races: snap });
    }
    console.log("changed=" + (got ? 1 : 0));
  })().catch((e) => { console.log("失敗: " + (e && e.stack || e)); console.log("changed=0"); process.exitCode = 1; });
}

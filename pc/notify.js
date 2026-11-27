// ============================================================
// notify.js — 発走の少し前(既定 3分前)に、そのときのオッズで買い目を出して、スマホに通知する(ntfy。PC 用。2026-11-27)
//
// 競輪(honda1986/keirin の bet/notify.py)と同じく ntfy(https://ntfy.sh)に送る。スマホに ntfy アプリを入れて、
// 同じトピック名を購読すると届く。★トピック名を知っている人は誰でも読めるので、推測されにくい名前にする(リポジトリには入れない)。
//
// ・pc/odds.js が1分おきにオッズを取ったあとで呼ぶ(発走12分前からは毎回取っている)。発走まで MINUTES 分を切ったレースを1回だけ知らせる
// ・買い目はアプリ(keiba/index.html)と同じ選び方で、取ったばかりのオッズで出す:
//     後押しの買い目 … 後押しが上5%の馬(1番人気でない・単勝20倍以下)× 1番人気 の馬連(keiba_logic.oshi_picks・アプリの oshiPicks)
//     期待値ワイド   … 確率の高い6組と ◎-○ のうち、期待値0.95以上・後押し込み1.0以上の組(keiba_logic.ev_wide_picks・アプリの evWideOf)
//     モデルの買い目 … 予想の買い目(3連複◎-○-▲ など。オッズでは変わらない)
//     後押しワイド(試し)… 後押し(2頭とも3着内の式)が上5%の馬 × 2番人気 のワイド(keiba_logic.oshiw_picks・アプリの oshiwPicks)
//   買い目が1つも無いレースは送らない
// ・通知に入れるのは 場・R・発走・買い目・オッズ・期待値だけ(お金・ログイン情報は入れない)。送れなくてもオッズの記録は止めない
//
// 設定: C:\keiba\notify.json(リポジトリの外){"ntfy_topic": "…", "ntfy_server": "https://ntfy.sh", "minutes": 3, "kinds": ["oshi", "evw", "model"]}
//   kinds … 通知する買い目の種類(oshi 後押しの買い目 / evw 期待値ワイド / model モデルの買い目 / oshiw 後押しワイド)。全部だと1開催日に20通くらい
//   pc\set_notify.bat(= node pc\notify.js --setup)で作る。トピック名が空なら何もしない
// 試す: node pc\notify.js --test   … 1通送る
// ============================================================
"use strict";
const fs = require("fs");
const path = require("path");

const CONF = process.env.KEIBA_NOTIFY_CONF || path.join(__dirname, "..", "..", "notify.json");
const APP = "https://honda1986.github.io/keeeiba-app/keiba/";
// 買い目の決め方の数字(keiba_logic.py と同じ。test_keiba.py が見比べる)
const PRE = { pt: 0.75, alpha: 0.35 };
const EV_WIDE = { min_ev: 0.95, top: 6, push_ev: 1.0 };
const OSHI = { q: "0.95", max_odds: 20.0 };
const OSHIW = { q: "0.95", max_odds: 20.0 };
const KINDS = ["oshi", "evw", "model", "oshiw"];

// ---- 買い目(アプリ・keiba_logic と同じ計算) ----
// 単勝: odds = {馬番: [オッズ, 人気]}(pc/odds.js の parseWin の形)-> {馬番: オッズ}。取消は除く
const winOdds = (odds, gone) => {
  const o = {};
  for (const k of Object.keys(odds || {})) { const u = +k, v = odds[k]; const x = Array.isArray(v) ? v[0] : v; if (x > 1 && !(gone || []).includes(u)) o[u] = +x; }
  return o;
};
// 後押しの買い目: push = {馬番: 後押し}、od = {馬番: 単勝オッズ}、cut = 後押しの境目 -> [{u, fav}]
function oshiPicks(push, od, cut) {
  const us = Object.keys(od).map(Number);
  if (us.length < 5 || cut == null) return [];
  const fav = us.reduce((a, b) => (od[b] < od[a] || (od[b] === od[a] && b < a)) ? b : a);
  return us.sort((a, b) => a - b).filter((u) => u !== fav && od[u] <= OSHI.max_odds && push[u] != null && push[u] >= cut).map((u) => ({ u, fav }));
}
// 後押しワイド: push2 = {馬番: 後押し(2頭とも3着内の式)} -> [{u, mate(2番人気。u が2番人気なら3番人気)}]
function oshiwPicks(push2, od, cut) {
  const us = Object.keys(od).map(Number);
  if (us.length < 5 || cut == null) return [];
  const rank = [...us].sort((a, b) => od[a] - od[b] || a - b);
  return us.sort((a, b) => a - b).filter((u) => u !== rank[0] && od[u] <= OSHIW.max_odds && push2[u] != null && push2[u] >= cut)
    .map((u) => ({ u, mate: rank.filter((v) => v !== u)[1] }));
}
// keiba_push.zs と同じ(平均0・ばらつき1。±3で切る)
function zs(vals) {
  const n = vals.length, mean = vals.reduce((a, v) => a + v, 0) / n;
  const sd = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / n);
  return vals.map((v) => (sd <= 1e-9 ? 0 : Math.max(-3, Math.min(3, (v - mean) / sd))));
}
// 後押しの式の組の確率(keiba_push.pair_probs・アプリの pushPairProbs と同じ)。式・後押しが無ければ null
function pairProbs(od, tp, wide, push2, pb) {
  if (!pb || !tp || !wide) return null;
  const us = Object.keys(od).map(Number).sort((a, b) => a - b);
  if (us.length < 5 || !Object.keys(push2).length) return null;
  const tot = us.reduce((a, u) => a + 1 / od[u], 0), lq = {}, zq = {};
  for (const u of us) lq[u] = Math.log(1 / od[u] / tot);
  zs(us.map((u) => lq[u])).forEach((z, i) => { zq[us[i]] = z; });
  const lg = (p) => { p = Math.min(Math.max(p, 1e-4), 1 - 1e-4); return Math.log(p / (1 - p)); };
  const out = {};
  for (const k of Object.keys(wide)) {
    const s = tp[k], v = wide[k], lo = Array.isArray(v) ? v[0] : v, hi = Array.isArray(v) ? v[1] : v;
    if (!s || !lo || lo <= 1) continue;
    const [a, b] = k.split("-").map(Number);
    if (!(a in lq) || !(b in lq) || push2[a] == null || push2[b] == null) continue;
    const x0 = lg(Math.min(PRE.pt * s, 0.98)), xw = lg(Math.min(0.775 / ((lo + hi) / 2), 0.98));
    const f = [1, x0, x0 * x0, xw, xw * xw, zq[a] + zq[b], lq[a] + lq[b]];
    out[k] = 1 / (1 + Math.exp(-(f.reduce((acc, x, i) => acc + pb[i] * x, 0) + push2[a] + push2[b])));
  }
  return out;
}
// 期待値ワイド: 表に出る組(確率の高い順に6組 + ◎-○)のうち、期待値0.95以上・後押し込み1.0以上(後押しが無ければ絞らない)
function evWidePicks(tp, wide, hk, pp, gone) {
  const rows = [];
  for (const k of Object.keys(wide || {})) {
    const s = (tp || {})[k], v = wide[k], lo = Array.isArray(v) ? v[0] : v, hi = Array.isArray(v) ? v[1] : v;
    if (!s || !lo || lo <= 1) continue;
    const p = Math.min(PRE.pt * s, 0.98), eff = lo + PRE.alpha * (hi - lo);
    rows.push({ key: k, pair: k.split("-").map(Number), p, odds: [lo, hi], ev: p * eff, pev: pp && pp[k] != null ? pp[k] * eff : null });
  }
  rows.sort((a, b) => b.p - a.p);       // 同じ確率なら元の並び(Array.sort は安定)
  const show = rows.slice(0, EV_WIDE.top);
  if (hk && !show.some((x) => x.key === hk)) show.push(...rows.filter((x) => x.key === hk));
  const pushed = !!(pp && Object.keys(pp).length);
  return show.filter((x) => !x.pair.some((u) => (gone || []).includes(u)) && x.ev >= EV_WIDE.min_ev &&
    (!pushed || (x.pev != null && x.pev >= EV_WIDE.push_ev)));
}
// レース1つの買い目。race = pc/odds.js の一覧の1件(pred つき)、s = そのときのオッズ {odds, wide, tp}、pm = push_model.json
function racePicks(race, s, pm) {
  const gone = race.gone || [], pr = race.pred || {};
  const od = winOdds(s.odds, gone);
  const cut = ((((pm || {}).cuts || {}).win_top) || {})[OSHI.q];
  const oshi = oshiPicks(pr.push || {}, od, cut).filter((x) => !gone.includes(x.u) && !gone.includes(x.fav));
  const hk = pr.honmei && pr.taikou ? [pr.honmei, pr.taikou].sort((a, b) => a - b).join("-") : null;
  const pp = pairProbs(od, s.tp, s.wide, pr.push2 || {}, (pm || {}).pair_base);
  const evw = evWidePicks(s.tp, s.wide, hk, pp, gone);
  const model = ["A", "B"].includes(pr.verdict) ? (pr.bets || []) : [];
  const cut2 = ((((pm || {}).cuts || {}).pair_top) || {})[OSHIW.q];
  const oshiw = oshiwPicks(pr.push2 || {}, od, cut2).filter((x) => !gone.includes(x.u) && !gone.includes(x.mate));
  return { od, oshi, evw, model, oshiw };
}
// 通知の中身。買い目が無ければ null
function message(race, day, minutes, pk, kinds) {
  const ks = kinds || KINDS;
  pk = { od: pk.od, oshi: ks.includes("oshi") ? pk.oshi : [], evw: ks.includes("evw") ? pk.evw : [], model: ks.includes("model") ? pk.model : [],
         oshiw: ks.includes("oshiw") ? (pk.oshiw || []) : [] };
  if (!pk.oshi.length && !pk.evw.length && !pk.model.length && !pk.oshiw.length) return null;
  const nm = (u) => (race.names || {})[u] ? " " + race.names[u] : "";
  const f1 = (x) => (x >= 100 ? Math.round(x) : x.toFixed(1));
  const lines = [];
  for (const x of pk.oshi) {
    const c = [x.u, x.fav].sort((a, b) => a - b).join("-");
    lines.push("後押し 馬連 " + c + "(" + x.u + nm(x.u) + " " + f1(pk.od[x.u]) + "倍 × 1番人気 " + x.fav + nm(x.fav) + " " + f1(pk.od[x.fav]) + "倍)");
  }
  for (const x of pk.oshiw) {
    const c = [x.u, x.mate].sort((a, b) => a - b).join("-");
    lines.push("後押しワイド(試し) " + c + "(" + x.u + nm(x.u) + " " + f1(pk.od[x.u]) + "倍 × " + x.mate + nm(x.mate) + " " + f1(pk.od[x.mate]) + "倍)");
  }
  for (const x of pk.evw)
    lines.push("期待値ワイド " + x.key + "(" + (x.odds[0] === x.odds[1] ? x.odds[0] : x.odds[0] + "〜" + x.odds[1]) + "倍 期待値" + x.ev.toFixed(2) +
      (x.pev != null ? "・後押し込み" + x.pev.toFixed(2) : "") + ")");
  for (const b of pk.model) lines.push("モデル " + b.type + " " + (b.combos || []).map((c) => c.join(b.type === "3連単" ? "→" : "-")).join(" "));
  return {
    title: race.venue + race.rno + "R " + race.start + " 発走" + minutes + "分前",
    message: lines.join("\n") + "\n(締切は発走2分前)",
    tags: ["horse_racing"], priority: 4,
    click: APP + "#d=" + day + "&r=" + race.id,
  };
}

// ---- 送る ----
const readJson = (f, d) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { return d; } };
function config() {
  const c = readJson(CONF, {}) || {};
  const ks = (Array.isArray(c.kinds) ? c.kinds : KINDS).filter((k) => KINDS.includes(k));
  return { topic: String(c.ntfy_topic || "").trim(), server: String(c.ntfy_server || "https://ntfy.sh").replace(/\/+$/, ""),
           minutes: +c.minutes > 0 ? +c.minutes : 3, kinds: ks.length ? ks : KINDS };
}
async function post(cfg, msg) {
  const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), 8000);
  try {
    const r = await fetch(cfg.server + "/", { method: "POST", signal: ctl.signal, headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(Object.assign({ topic: cfg.topic }, msg)) });
    if (!r.ok) throw new Error("HTTP " + r.status);
  } finally { clearTimeout(t); }
}
// pc/odds.js から呼ぶ: 発走まで minutes 分を切ったレース(まだ知らせていないもの)を知らせる。
// list = レースの一覧(pred つき)、snap = そのときのオッズ、fresh = この回で取ったレースの id、pm = push_model.json
async function notifyDue({ list, snap, fresh, day, now, work, pm, send, cfg: cfgIn }) {
  const cfg = cfgIn || config();
  if (!cfg.topic && !send) return 0;
  const f = path.join(work, "notified-" + day + ".json"), done = readJson(f, {}) || {};
  let n = 0;
  for (const r of list) {
    const m = (startMsOf(day, r.start) - now) / 60000;
    if (done[r.id] || !(m > 0 && m <= cfg.minutes + 0.5) || !fresh.includes(r.id) || !snap[r.id]) continue;
    const msg = message(r, day, Math.max(1, Math.round(m)), racePicks(r, snap[r.id], pm), cfg.kinds);
    done[r.id] = msg ? "sent" : "none";
    if (!msg) continue;
    try { await (send || ((x) => post(cfg, x)))(msg); n++; console.log("通知した: " + msg.title); }
    catch (e) { done[r.id] = "failed"; console.log("通知できず(" + e.message + "): " + msg.title); }
  }
  try { fs.mkdirSync(work, { recursive: true }); fs.writeFileSync(f, JSON.stringify(done)); } catch (e) { /* 次の回で */ }
  return n;
}
const startMsOf = (day, hhmm) => Date.UTC(+day.slice(0, 4), +day.slice(4, 6) - 1, +day.slice(6, 8), +hhmm.slice(0, 2) - 9, +hhmm.slice(3, 5));

module.exports = { oshiPicks, oshiwPicks, pairProbs, evWidePicks, racePicks, message, notifyDue, zs, PRE, EV_WIDE, OSHI, OSHIW };

// ---- 設定・試し(node pc\notify.js --setup / --test) ----
if (require.main === module) {
  const argv = process.argv.slice(2);
  (async () => {
    if (argv.includes("--setup")) {
      const rl = require("readline").createInterface({ input: process.stdin, output: process.stdout });
      const ask = (q) => new Promise((r) => rl.question(q, r));
      const cur = config();
      const rnd = "keiba-" + require("crypto").randomBytes(6).toString("hex");
      console.log("通知先(ntfy のトピック名)。★知っている人は誰でも読めるので、推測されにくい名前にする(半角の英数字・_・-)");
      console.log("  いま: " + (cur.topic || "(なし)") + "   おすすめ: " + rnd + "(+ を入れるとこれを使う)");
      let t = (await ask("トピック名(空で今のまま): ")).trim();
      if (t === "+") t = rnd;
      if (t && !/^[A-Za-z0-9_-]{6,64}$/.test(t)) { console.log("使えない名前です(半角の英数字・_・- で6〜64文字)"); rl.close(); return; }
      const mi = (await ask("発走の何分前に知らせるか(空で " + cur.minutes + "。締切は発走2分前): ")).trim();
      console.log("知らせる買い目: 1 後押しの買い目 / 2 期待値ワイド / 3 モデルの買い目 / 4 後押しワイド(試し)(全部だと1開催日に20通くらい)");
      const kn = (await ask("番号を続けて(例 124。空で今のまま: " + cur.kinds.map((k) => ({ oshi: 1, evw: 2, model: 3, oshiw: 4 })[k]).join("") + "): ")).trim();
      rl.close();
      const kinds = kn ? [["1", "oshi"], ["2", "evw"], ["3", "model"], ["4", "oshiw"]].filter(([d]) => kn.includes(d)).map(([, k]) => k) : cur.kinds;
      const c = Object.assign(readJson(CONF, {}) || {}, { ntfy_topic: t || cur.topic, ntfy_server: cur.server,
        minutes: mi ? Math.max(1, Math.min(30, +mi || 3)) : cur.minutes, kinds: kinds.length ? kinds : cur.kinds });
      fs.writeFileSync(CONF, JSON.stringify(c, null, 1));
      console.log("保存しました: " + CONF);
      console.log("スマホの ntfy アプリで「" + c.ntfy_topic + "」を購読(Subscribe to topic)してください。続けて試しに1通送ります");
    }
    if (argv.includes("--setup") || argv.includes("--test")) {
      const cfg = config();
      if (!cfg.topic) { console.log("トピック名がまだありません。pc\\set_notify.bat で入れてください"); process.exitCode = 1; return; }
      await post(cfg, { title: "通知のテスト(競馬)", message: "keeeiba-app の PC から送りました。発走" + cfg.minutes + "分前に買い目が届きます", tags: ["bell"], priority: 3 });
      console.log(cfg.server + "/" + cfg.topic + " に送りました。スマホに届いたか見てください");
    }
  })().catch((e) => { console.log("送れませんでした: " + e.message); process.exitCode = 1; });
}

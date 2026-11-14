// ============================================================
// runner.js — PC のタスクスケジューラが1分おきに呼ぶ(pc\task.bat → node pc\runner.js)
//
// 競輪(honda1986/keirin の pc/runner.js)と同じ作り。GitHub の時刻指定は遅れるので、開催日のオッズはふだん PC が取る。
//   1. 二重に動かさない(ロック)
//   2. 10分おきに GitHub(main)の pc/ を取り込む(このフォルダは pc/ だけを置いた軽い写し。手では直さない)
//   3. pc/odds.js で、発走が近いレースのオッズを取る
//   4. 新しいオッズがあれば odds-live ブランチへ送る(latest.json と snap/日付.json。履歴は残さない = いつも1コミット)。
//      無くても5分おきに「動いている」の印(heartbeat.json)を送る
//
// フォルダ(install.ps1 が作る):
//   C:\keiba\keeeiba-app … このリポジトリの pc/ だけ      C:\keiba\odds-live … 送る枝の写し
//   C:\keiba\work       … 覚え書き(レースの一覧・見た時刻)  C:\keiba\logs      … ログ(odds_日付.log)
//
// 使い方: node pc\runner.js              (ふだん。開催の時間帯 8:40〜17:40 だけ動く)
//         node pc\runner.js --dry        (取って表示するだけ。送らない)
//         node pc\runner.js --force      (時間帯の外でも動かす)
// ============================================================
"use strict";
const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const argv = process.argv.slice(2);
const DRY = argv.includes("--dry");
const FORCE = argv.includes("--force") || DRY;
const REPO = path.join(__dirname, "..");
const ROOT = path.join(REPO, "..");
const LIVE = path.join(ROOT, "odds-live");
const WORK = path.join(ROOT, "work");
const LOGDIR = argv.includes("--log-dir") ? argv[argv.indexOf("--log-dir") + 1] : path.join(ROOT, "logs");
const URL = process.env.KEIBA_GIT_URL || "https://github.com/honda1986/keeeiba-app.git";      // KEIBA_GIT_URL は動作確認用
const TASK = "odds";
const STATE = path.join(WORK, "runner.json");
const LOCK = path.join(LOGDIR, TASK + ".lock");
const LOCK_STALE_MIN = 10;      // これより古いロックは、強制終了で残ったものとみなして捨てる
const PULL_EVERY_MIN = 10;      // GitHub(main)の pc/ を取り込む間隔
const BEAT_EVERY_MIN = 5;       // 新しいオッズが無くても「動いている」の印を送る間隔
const HOURS = [8 * 60 + 40, 17 * 60 + 40];   // 動く時間帯(日本時間)。中央の発走は 9:30 ごろ〜16:40 ごろ
const KEEP_DAYS = 7;            // work の日別の覚え書きを残す日数

// ★git に「人に聞く」をさせない(競輪・競艇と同じ)。タスクスケジューラには画面が無いので、認証を聞かれると固まる
const GIT_ENV = { GIT_TERMINAL_PROMPT: "0", GCM_INTERACTIVE: "never", GIT_ASKPASS: "", SSH_ASKPASS: "" };
const AUTHOR = { GIT_AUTHOR_NAME: "odds-live", GIT_AUTHOR_EMAIL: "odds-live@users.noreply.github.com",
  GIT_COMMITTER_NAME: "odds-live", GIT_COMMITTER_EMAIL: "odds-live@users.noreply.github.com" };
const GIT_TIMEOUT_MS = 120 * 1000;

const jst = () => new Date(Date.now() + 9 * 3600e3);
const today8 = () => jst().toISOString().slice(0, 10).replace(/-/g, "");
const hhmmss = () => jst().toISOString().slice(11, 19);
const isoJst = () => jst().toISOString().slice(0, 19) + "+09:00";

function log(msg) {
  const lines = String(msg).split(/\r?\n/).filter((l) => l.trim());
  const text = lines.map((l) => hhmmss() + " " + l).join("\n");
  if (!text) return;
  console.log(text);
  try { fs.mkdirSync(LOGDIR, { recursive: true }); fs.appendFileSync(path.join(LOGDIR, TASK + "_" + today8() + ".log"), text + "\n"); }
  catch (e) { console.log("(ログに書けません: " + e.message + ")"); }
}
function run(cmd, args, { cwd = REPO, env = {}, timeout = GIT_TIMEOUT_MS } = {}) {
  const p = spawnSync(cmd, args, { cwd, env: { ...process.env, ...env }, encoding: "utf8", timeout, windowsHide: true });
  if (p.error) {
    const why = p.error.code === "ETIMEDOUT" ? (timeout / 1000) + "秒で戻ってこないので諦めました" : p.error.message;
    return { ok: false, out: cmd + " " + args.join(" ") + ": " + why };
  }
  return { ok: p.status === 0, out: ((p.stdout || "") + (p.stderr || "")).trim() };
}
const git = (args, o = {}) => run("git", args, { ...o, env: { ...GIT_ENV, ...(o.env || {}) } });
const lastLine = (s) => String(s).split("\n").filter((l) => l.trim()).pop() || "";
const loadState = () => { try { return JSON.parse(fs.readFileSync(STATE, "utf8")); } catch (e) { return {}; } };
const saveState = (st) => { try { fs.mkdirSync(WORK, { recursive: true }); fs.writeFileSync(STATE, JSON.stringify(st)); } catch (e) { /* 次の回で */ } };
const minsSince = (t) => (t ? (Date.now() - t) / 60000 : Infinity);

// ---- 1. ロック ----
function takeLock() {
  fs.mkdirSync(LOGDIR, { recursive: true });
  for (let a = 0; a < 2; a++) {
    try { const fd = fs.openSync(LOCK, "wx"); fs.writeSync(fd, String(process.pid) + " " + new Date().toISOString()); fs.closeSync(fd); return true; }
    catch (e) {
      if (e.code !== "EEXIST") throw e;
      let age = 0;
      try { age = (Date.now() - fs.statSync(LOCK).mtimeMs) / 60000; } catch (e2) { continue; }
      if (age < LOCK_STALE_MIN) return false;
      log("古いロック(" + Math.round(age) + "分前)を捨てます。前の回が強制終了されたようです");
      try { fs.unlinkSync(LOCK); } catch (e3) { /* 次の回で */ }
    }
  }
  return false;
}
const dropLock = () => { try { fs.unlinkSync(LOCK); } catch (e) { /* 無ければよい */ } };

// ---- 2. GitHub(main)の pc/ を取り込む ----
function pullCode(st) {
  if (minsSince(st.lastPull) < PULL_EVERY_MIN) return;
  st.lastPull = Date.now();
  if (!fs.existsSync(path.join(REPO, ".git"))) return;            // 手で置いただけのフォルダなら取り込まない
  const f = git(["fetch", "-q", "--depth", "1", "origin", "main"]);
  if (!f.ok) { log("GitHub から取り込めず(手元のまま続ける): " + lastLine(f.out)); return; }
  const before = git(["rev-parse", "--short", "HEAD"]).out;
  const r = git(["reset", "-q", "--hard", "FETCH_HEAD"]);
  if (!r.ok) { log("取り込みに失敗(手元のまま続ける): " + lastLine(r.out)); return; }
  const after = git(["rev-parse", "--short", "HEAD"]).out;
  if (before !== after) log("GitHub を取り込みました " + before + " → " + after);
}

// ---- 3. 送る枝(odds-live)の写し ----
function ensureLive() {
  if (fs.existsSync(path.join(LIVE, ".git"))) return true;
  const c = git(["clone", "-q", "--depth", "1", "--single-branch", "-b", "odds-live", URL, LIVE], { cwd: ROOT });
  if (c.ok) return true;
  // 枝がまだ無い(最初の1回): 空の枝を作る
  fs.mkdirSync(LIVE, { recursive: true });
  const ok = git(["init", "-q"], { cwd: LIVE }).ok && git(["remote", "add", "origin", URL], { cwd: LIVE }).ok
    && git(["checkout", "-q", "--orphan", "odds-live"], { cwd: LIVE }).ok;
  if (!ok) log("送る枝の写しを作れません: " + lastLine(c.out));
  return ok;
}
function pushLive(st, what) {
  git(["add", "-A"], { cwd: LIVE });
  const has = git(["rev-parse", "-q", "--verify", "HEAD"], { cwd: LIVE }).ok;
  // いつも1コミットだけにする(--amend で上書き)。1分おきに送っても枝が膨らまない
  const c = git(["commit", "-q", "--allow-empty", ...(has ? ["--amend"] : []), "-m", "odds live " + isoJst()], { cwd: LIVE, env: AUTHOR });
  if (!c.ok) { log("まとめられず(次の回に持ち越し): " + lastLine(c.out)); return false; }
  const p = git(["push", "-q", "-f", "origin", "HEAD:refs/heads/odds-live"], { cwd: LIVE });
  if (!p.ok) { log("送れず(次の回に持ち越し): " + lastLine(p.out)); return false; }
  st.lastPush = Date.now();
  log("送りました(" + what + ")");
  // 1日1回、古い中身を片づける(--amend の残りが溜まるため)
  if (st.gcDay !== today8()) {
    st.gcDay = today8();
    git(["reflog", "expire", "--expire=now", "--all"], { cwd: LIVE });
    git(["gc", "-q", "--prune=now"], { cwd: LIVE });
  }
  return true;
}

function cleanup() {
  try {
    const limit = Date.now() - KEEP_DAYS * 86400e3;
    for (const f of fs.readdirSync(WORK)) {
      if (!/^(races|snap)-\d{8}\.json$/.test(f)) continue;
      const p = path.join(WORK, f);
      if (fs.statSync(p).mtimeMs < limit) fs.unlinkSync(p);
    }
  } catch (e) { /* 片づけは失敗してもよい */ }
}

function main() {
  const hm = jst().getUTCHours() * 60 + jst().getUTCMinutes();
  if (!FORCE && (hm < HOURS[0] || hm > HOURS[1])) return;          // 時間帯の外は何もしない(ログも出さない)
  if (!takeLock()) { console.log("前の回がまだ動いています"); return; }
  try {
    const st = loadState();
    pullCode(st);
    // 今日は中央の開催が無いと分かっていれば、30分は見に行かない
    if (!FORCE && st.noRacesDay === today8() && minsSince(st.noRacesAt) < 30) { saveState(st); return; }
    if (!DRY && !ensureLive()) { saveState(st); return; }
    const args = [path.join(REPO, "pc", "odds.js"), "--work=" + WORK];
    if (DRY) args.push("--dry"); else args.push("--out=" + LIVE);
    const r = run(process.execPath, args, { timeout: 90 * 1000 });
    const lines = r.out.split(/\r?\n/).filter((l) => l.trim());
    const changed = lines.includes("changed=1"), noraces = lines.includes("noraces=1");
    const show = lines.filter((l) => !/^(changed|noraces)=/.test(l));
    if (show.length && (changed || DRY || !noraces || st.noRacesDay !== today8())) log(show.join("\n"));
    if (noraces) { st.noRacesDay = today8(); st.noRacesAt = Date.now(); }
    if (!DRY && !noraces) {
      if (changed) pushLive(st, "新しいオッズ");
      else if (minsSince(st.lastPush) >= BEAT_EVERY_MIN) {
        try { fs.writeFileSync(path.join(LIVE, "heartbeat.json"), JSON.stringify({ at: isoJst(), source: "PC" })); } catch (e) { /* 次の回で */ }
        pushLive(st, "動いている印");
      }
    }
    if (st.cleanDay !== today8()) { st.cleanDay = today8(); cleanup(); }
    saveState(st);
  } catch (e) {
    log("例外: " + (e && e.stack || e));
  } finally {
    dropLock();
  }
}
main();

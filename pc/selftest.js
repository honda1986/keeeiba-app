// selftest.js — pc/odds.js の読み取りと時刻の計算を、ネットにつながずに確かめる(install.ps1 が呼ぶ)
//   node pc/selftest.js      全部通れば終了コード 0
"use strict";
const assert = require("assert");
const O = require("./odds.js");
let n = 0;
const t = (name, fn) => { try { fn(); n++; console.log("  ○ " + name); } catch (e) { console.log("  × " + name + ": " + e.message); process.exitCode = 1; } };

t("単勝を読む(人気が無ければオッズの低い順)", () => {
  const j = { data: { odds: { "1": { "01": ["2.4", "", "1"], "02": ["15.0", "", "3"], "10": ["4.5", "", "2"], "11": ["0.0", "", ""] } } } };
  assert.deepStrictEqual(O.parseWin(j), { "1": [2.4, 1], "2": [15, 3], "10": [4.5, 2] });
  assert.deepStrictEqual(O.parseWin({ data: { odds: { "1": { "01": ["3.0", "", ""], "02": ["1.5", "", ""] } } } }), { "1": [3, 2], "2": [1.5, 1] });
  assert.deepStrictEqual(O.parseWin({}), {});
});
t("ワイドを読む(小さい馬番-大きい馬番 → 下限・上限)", () => {
  const j = { data: { odds: { "5": { "0102": ["2.0", "2.4", "1"], "0310": ["1,234.5", "1,300.0", "9"], "0204": ["0.0", "0.0", ""] } } } };
  assert.deepStrictEqual(O.parseWide(j), { "1-2": [2, 2.4], "3-10": [1234.5, 1300] });
});
t("3連複を読んで、組ごとの強さにする(keiba.py の trio_pair_sums と同じ)", () => {
  const j = { data: { odds: { "7": { "010203": ["5.0", "", "1"], "010204": ["10.0", "", "2"], "010304": ["20.0", "", "3"], "020304": ["40.0", "", "4"], "010210": ["1,000.0", "", "9"] } } } };
  const trio = O.parseTrio(j);
  assert.deepStrictEqual(trio, { "1-2-3": 5, "1-2-4": 10, "1-3-4": 20, "2-3-4": 40, "1-2-10": 1000 });
  assert.deepStrictEqual(O.pairSums(trio), { "1-2": 0.301, "1-3": 0.25, "2-3": 0.225, "1-4": 0.15, "2-4": 0.125, "3-4": 0.075 });
});
t("見るレース: 発走まで35分以内。12分前からは毎回、それより前は3分おき", () => {
  const day = "20261114", now = O.startMs(day, "10:00");
  const list = [{ id: "a", start: "10:05" }, { id: "b", start: "10:30" }, { id: "c", start: "10:40" }, { id: "d", start: "09:59" }];
  assert.deepStrictEqual(O.dueRaces(list, day, now, {}, [0, 35]).map((r) => r.id), ["a", "b"]);
  // a は1分前に見た(12分以内なので50秒たてば見る)、b は1分前に見た(3分おきなのでまだ)
  assert.deepStrictEqual(O.dueRaces(list, day, now, { a: now - 60e3, b: now - 60e3 }, [0, 35]).map((r) => r.id), ["a"]);
  assert.deepStrictEqual(O.dueRaces(list, day, now, { a: now - 30e3, b: now - 180e3 }, [0, 35]).map((r) => r.id), ["b"]);
});
t("latest.json に載せるのは、これからのレースと発走から20分以内", () => {
  const day = "20261114", now = O.startMs(day, "10:00");
  const list = [{ id: "a", start: "09:30" }, { id: "b", start: "09:45" }, { id: "c", start: "10:10" }, { id: "d", start: "11:00" }];
  const snap = { a: 1, b: 2, c: 3 };
  assert.deepStrictEqual(O.latestOf(snap, list, day, now), { b: 2, c: 3 });
});
t("時刻(日本時間)", () => {
  assert.strictEqual(O.startMs("20261114", "09:00"), Date.UTC(2026, 10, 14, 0, 0));
  assert.strictEqual(O.isoJst(Date.UTC(2026, 10, 14, 0, 5, 7)), "2026-11-14T09:05:07+09:00");
  assert.strictEqual(O.day8(Date.UTC(2026, 10, 13, 15, 0)), "20261114");
});
// ---- 通知(pc/notify.js。2026-11-27) ----
const N = require("./notify.js");
t("後押しの買い目: 1番人気の後押しが弱いレースで、2頭とも3着内の式の上5%の馬 × 2番人気(その馬が2番人気なら3番人気)", () => {
  const od = { 1: 2.0, 2: 4.0, 3: 6.0, 4: 12.0, 5: 25.0 };
  const p2 = { 1: 0.5, 2: 0.3, 4: 0.2, 5: 0.4 };
  assert.deepStrictEqual(N.oshi2Picks({ 1: -0.5 }, p2, od, -0.3, 0.15), [{ u: 2, mate: 3, fav: 1 }, { u: 4, mate: 2, fav: 1 }]);  // 5 は20倍超え
  assert.deepStrictEqual(N.oshi2Picks({ 1: 0.1 }, p2, od, -0.3, 0.15), []);                                  // 1番人気の後押しが弱くない
  assert.deepStrictEqual(N.oshi2Picks({}, p2, od, -0.3, 0.15), []);                                          // 1番人気の後押しが無い
  assert.deepStrictEqual(N.oshi2Picks({ 1: -0.5 }, { 4: 0.2 }, { 1: 2, 4: 12 }, -0.3, 0.15), []);             // 5頭いない
});
t("期待値ワイド: 期待値0.95以上で、後押し込み1.0以上(後押しが無ければ絞らない)", () => {
  const tp = { "1-2": 0.8, "1-3": 0.5, "2-3": 0.45, "1-4": 0.3, "2-4": 0.25, "3-4": 0.2, "1-5": 0.15, "5-6": 0.1 };
  const wide = { "1-2": [1.2, 1.4], "1-3": [2.6, 3.0], "2-3": [3.0, 3.4], "1-4": [3.8, 4.6], "2-4": [6.0, 7.0], "3-4": [5.0, 6.0], "1-5": [9.0, 10.0], "5-6": [20.0, 25.0] };
  assert.deepStrictEqual(N.evWidePicks(tp, wide, null, null, []).map((x) => x.key), ["1-3", "2-3", "2-4"]);     // test_keiba の test_ev_wide と同じ
  assert.deepStrictEqual(N.evWidePicks(tp, wide, "5-6", null, []).map((x) => x.key), ["1-3", "2-3", "2-4", "5-6"]);
  const eff = 2.6 + 0.35 * 0.4;
  assert.deepStrictEqual(N.evWidePicks(tp, wide, null, { "1-3": 1 / eff - 0.01, "2-3": 0.5, "2-4": 0.3 }, []).map((x) => x.key), ["2-3", "2-4"]);
  assert.deepStrictEqual(N.evWidePicks(tp, wide, null, null, [3]).map((x) => x.key), ["2-4"]);                  // 取消の馬の組は外す
});
t("通知の中身: 買い目が無ければ送らない。種類を絞れる", () => {
  const race = { id: "202605050811", venue: "東京", rno: 11, start: "15:40", names: { 3: "ア", 7: "イ" } };
  const pk = { od: { 1: 1.8, 3: 2.1, 7: 12.3 }, oshi: [{ u: 7, mate: 3, fav: 1 }], evw: [], model: [{ type: "3連複", combos: [[1, 4, 8]] }] };
  const m = N.message(race, "20261129", 3, pk);
  assert.strictEqual(m.title, "東京11R 15:40 発走3分前");
  assert.ok(m.message.includes("後押し 馬連・ワイド 3-7(7 イ 12.3倍 × 3 ア 2.1倍。1番人気 1 の後押しが弱い)") && m.message.includes("モデル 3連複 1-4-8"));
  assert.ok(m.click.endsWith("#d=20261129&r=202605050811"));
  assert.strictEqual(N.message(race, "20261129", 3, { od: {}, oshi: [], evw: [], model: [] }), null);
  assert.strictEqual(N.message(race, "20261129", 3, pk, ["evw"]), null);
  assert.ok(!N.message(race, "20261129", 3, pk, ["oshi"]).message.includes("モデル"));
});
(async () => {
  // 通知する時: 発走まで3分を切ったレースを1回だけ。この回でオッズを取ったレースだけ。送り先はにせもの
  const os = require("os"), fs = require("fs"), path = require("path");
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "kbnotify-"));
  const day = "20261129", now = O.startMs(day, "15:37");
  const pred = { verdict: "A", honmei: 1, taikou: 2, bets: [{ type: "3連複", combos: [[1, 2, 3]] }], push: {}, push2: {} };
  const list = [{ id: "r1", venue: "東京", rno: 11, start: "15:40", pred }, { id: "r2", venue: "東京", rno: 12, start: "15:50", pred },
                { id: "r3", venue: "京都", rno: 11, start: "15:39", pred }];
  const s = { odds: { 1: [2, 1], 2: [3, 2], 3: [5, 3], 4: [9, 4], 5: [20, 5] }, wide: {}, tp: {} };
  const snap = { r1: s, r2: s, r3: s }, sent = [];
  const go = (fresh, at) => N.notifyDue({ list, snap, fresh, day, now: at, work, pm: null, send: async (m) => sent.push(m.title),
    cfg: { topic: "x", server: "", minutes: 3, kinds: ["oshi", "evw", "model"] } });
  try {
    await go(["r1", "r2"], now);                    // r1 は3分前 → 送る。r2 は13分前 → まだ。r3 はこの回で取っていない
    assert.deepStrictEqual(sent, ["東京11R 15:40 発走3分前"]);
    await go(["r1", "r2", "r3"], now + 60e3);       // r1 は送ったので送らない。r3(1分前)を送る
    assert.deepStrictEqual(sent, ["東京11R 15:40 発走3分前", "京都11R 15:39 発走1分前"]);
    n++; console.log("  ○ 通知する時: 発走まで3分を切ったレースを1回だけ(この回でオッズを取ったものだけ)");
  } catch (e) { console.log("  × 通知する時: " + e.message); process.exitCode = 1; }
  console.log(process.exitCode ? "自己テスト: 失敗あり" : "自己テスト: " + n + " 個とも通りました");
})();

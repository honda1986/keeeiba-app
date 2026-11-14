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
console.log(process.exitCode ? "自己テスト: 失敗あり" : "自己テスト: " + n + " 個とも通りました");

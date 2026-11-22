"""開催日の10分ごとの更新(keiba_live.yml)を、次にいつ動かすかを決める(2026-11-22)。

GitHub の時刻指定は何時間も遅れたり飛ばされたりする(2026-11-22 は朝の分が5時間以上動かなかった)ので、
ワークフローが自分で次の回を起動する。これはその「次の回」を決めるだけで、起動は呼んだワークフローが gh でする。
出力(標準出力の1行):
  now                 … 今日が開催日で、最後のレースの結果をまだ取り終えていない(朝は START の15分前から)
  wait YYYYMMDDHHMM   … 次の開催日の START に始めたい(LOOKAHEAD 時間以内のときだけ。待つのは keiba_live.yml の wait_until)
  none                … 何もしない(開催が無い・まだ先すぎる)
  python tools/arm_live.py               # 決めるだけ(keiba/days の中央の開催を見る)
  python tools/arm_live.py --dispatch    # 決めて、keiba_live.yml がほかに動いて(待って)いなければ gh で起動する
                                         # (環境変数 GH_TOKEN・GITHUB_RUN_ID。LIVE_STARTED = この実行が始まった時刻 秒)
"""
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
START = "0831"        # 開催日の10分ごとの更新を始める時刻(keiba_live.yml の朝の時刻指定と同じ)
EARLY = 15            # START の何分前から「now」にするか
AFTER = 40            # 最後のレースの発走から何分後まで続けるか(tools/raceday.py と同じ)
LOOKAHEAD = 20        # 次の開催日の START まで何時間以内なら待ちに入るか
SHORT = 20            # 開催中なのにこの分より短く終わった実行は、すぐには起こし直さず RETRY 分後にする(起動が回り続けないように)
RETRY = 30
WF = "keiba_live.yml"


def race_days(out="keiba", since=""):
    """中央の開催日(since 以降)-> [(日付, 最後のレースの発走 HH:MM)] 日付の順"""
    days = []
    for f in sorted(glob.glob(os.path.join(out, "days", "*.json"))):
        d = os.path.basename(f)[:8]
        if d < since:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                day = json.load(fh)
        except (OSError, ValueError):
            continue
        starts = [r["start"] for r in (day.get("races") or {}).values()
                  if r.get("center") == "jra" and r.get("venue") and r.get("start")]
        if starts:
            days.append((d, max(starts)))
    return days


def decide(now, days):
    """now(日本時間の datetime)と開催日の一覧から、次にすることを返す"""
    today = now.strftime("%Y%m%d")
    for d, last in days:
        if d < today:
            continue
        start = datetime.strptime(d + START, "%Y%m%d%H%M").replace(tzinfo=JST)
        if d == today:
            h, m = map(int, last.split(":"))
            end = now.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(minutes=AFTER)
            if now > end:
                continue                         # 今日はもう終わった。次の開催日を見る
            if now >= start - timedelta(minutes=EARLY):
                return "now"
            return "wait " + d + START
        if start - now <= timedelta(hours=LOOKAHEAD):
            return "wait " + d + START
        return "none"
    return "none"


def guard(what, now, started):
    """開催中(now)なのに、この実行が SHORT 分より短く終わった(更新に失敗した・データが古いなど)ときは RETRY 分後に待つ"""
    if what == "now" and started and now.timestamp() - started < SHORT * 60:
        return "wait " + (now + timedelta(minutes=RETRY)).strftime("%Y%m%d%H%M")
    return what


def busy(run_id=""):
    """keiba_live.yml の、終わっていない実行(動いている・待っている)の数。この実行は数えない。読めなければ 0"""
    try:
        out = subprocess.run(["gh", "run", "list", "--workflow", WF, "--limit", "20", "--json", "status,databaseId"],
                             capture_output=True, text=True, timeout=60)
        runs = json.loads(out.stdout) if out.returncode == 0 else []
    except (OSError, ValueError, subprocess.SubprocessError):
        runs = []
    return sum(1 for r in runs if r.get("status") != "completed" and str(r.get("databaseId")) != str(run_id))


def dispatch(what):
    args = ["gh", "workflow", "run", WF, "--ref", "main"]
    if what.startswith("wait "):
        args += ["-f", "wait_until=" + what.split()[1]]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        print("起動できなかった:", e)
        return False
    print(" ".join(args[3:]), "→", "起動した" if r.returncode == 0 else "起動できなかった " + (r.stderr or "").strip())
    return r.returncode == 0


def main(argv):
    now = datetime.now(JST)
    what = decide(now, race_days(os.environ.get("KEIBA_OUT", "keiba"), now.strftime("%Y%m%d")))
    if "--dispatch" not in argv:
        print(what)
        return 0
    try:
        started = float(os.environ.get("LIVE_STARTED") or 0)
    except ValueError:
        started = 0
    what = guard(what, now, started)
    print("次の回:", what)
    if what == "none":
        return 0
    n = busy(os.environ.get("GITHUB_RUN_ID", ""))
    if n:
        print("ほかに {} 件動いて(待って)いるので起動しない".format(n))
        return 0
    dispatch(what)
    return 0                                     # 起動できなくても失敗にはしない(次の「競馬の予想」がまた起こす)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

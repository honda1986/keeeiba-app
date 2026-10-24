"""今日(日本時間)が中央の開催日で、まだ最後のレースの結果が出ていないかを、終了コードで返す。
0 = 続けて更新する / 1 = 開催が無い・もう終わった。keiba_live.yml の繰り返しで使う。

  python tools/raceday.py            # keiba/data.json を見る
"""
import json
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
AFTER = 40          # 最後のレースの発走から何分後まで続けるか(結果と払戻を取るため)


def main(path="keiba/data.json"):
    now = datetime.now(JST)
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        print("data.json が読めない")
        return 1
    if d.get("date") != now.strftime("%Y%m%d"):
        print("今日の開催は無い(data.json は", d.get("date"), ")")
        return 1
    starts = [r["start"] for r in (d.get("races") or {}).values()
              if r.get("center") == "jra" and r.get("venue") and r.get("start")]
    if not starts:
        print("今日の中央のレースが無い")
        return 1
    h, m = map(int, max(starts).split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(minutes=AFTER)
    if now > end:
        print("最後のレースの結果まで取り終えた", max(starts))
        return 1
    print("開催中: 最後のレース", max(starts), "/ 残り", int((end - now).total_seconds() // 60), "分")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

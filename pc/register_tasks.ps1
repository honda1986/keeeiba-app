# register_tasks.ps1 -- 競馬(中央)のオッズ取得をタスクスケジューラに登録する
#
#   install.ps1 が呼ぶ。直接なら PowerShell を管理者で開いて:
#     powershell -ExecutionPolicy Bypass -File C:\keiba\keeeiba-app\pc\register_tasks.ps1
#
# 競輪(honda1986/keirin の pc\register_tasks.ps1)と同じ設定・同じ注意。
# 消すとき: Unregister-ScheduledTask -TaskName keiba_odds -Confirm:$false

$repo = Split-Path -Parent $PSScriptRoot          # C:\keiba\keeeiba-app
$bat = Join-Path $repo "pc\task.bat"
if (-not (Test-Path $bat)) { throw "$bat がありません" }

# 毎日 08:40 から1分おきに 9時間(17:40 まで)
#   ・中央の発走は 9:30 ごろ〜16:40 ごろ。開催の無い日は、レースの一覧を見てすぐ終わる(30分に1回だけ見る)
#   ・1回ごとに終わる作り。発走まで35分以内のレースが無ければ数秒で終わる
$trigger = New-ScheduledTaskTrigger -Daily -At 8:40am
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At 8:40am `
    -RepetitionInterval (New-TimeSpan -Minutes 1) `
    -RepetitionDuration (New-TimeSpan -Hours 9)).Repetition

$action = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $repo
$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable
# ユーザーがログオンしていなくても動かす
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

# ★-ErrorAction Stop を付ける。既定では登録に失敗しても赤い字が出るだけで続いてしまう
Register-ScheduledTask -TaskName "keiba_odds" -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force -ErrorAction Stop | Out-Null
Write-Host "登録しました: keiba_odds"

# ★登録し終わったら、自分で確かめる
$t = Get-ScheduledTask -TaskName "keiba_odds" -ErrorAction SilentlyContinue
if (-not $t) {
    Write-Host "★タスク keiba_odds が登録できていません" -ForegroundColor Red
    throw "タスクがありません"
}
$rep = $t.Triggers[0].Repetition
if ($rep.Interval -ne "PT1M") {
    Write-Host "★1分おきの繰り返しが設定されていません(Interval=$($rep.Interval))" -ForegroundColor Red
    throw "繰り返しの設定が違います"
}
Write-Host "登録できています: keiba_odds(毎日 08:40 から1分おき、$($rep.Duration))" -ForegroundColor Green
Write-Host ""
Write-Host "確認:  Get-ScheduledTask keiba_* | Format-Table TaskName, State"
Write-Host "試す:  Start-ScheduledTask -TaskName keiba_odds"
Write-Host "       そのあと $(Join-Path (Split-Path -Parent $repo) 'logs')\odds_<日付>.log を見る"

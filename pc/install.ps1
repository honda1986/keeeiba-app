# install.ps1 -- 競馬(中央)のオッズ取得を、PC で1回で始める
#
#   スタートメニューで「PowerShell」を開いて、次の1行を貼って Enter:
#
#     irm https://raw.githubusercontent.com/honda1986/keeeiba-app/main/pc/install.ps1 | iex
#
#   やること(何度やり直しても壊れない。もうあるものは作り直さない):
#     1. 管理者に切り替える(「このアプリがデバイスに変更を加えることを許可しますか」で「はい」)
#     2. Git と Node.js が無ければ入れる(winget)。競輪で入れてあればそのまま使う
#     3. C:\keiba\keeeiba-app にリポジトリの pc フォルダだけを入れる(全部は大きいので入れない。あれば最新にする)
#     4. 送る枝(odds-live)の写しを C:\keiba\odds-live に作り、GitHub へ送れるか確かめる。送れなければトークンの画面を開いて、直すのを待つ
#     5. 自己テストと試し運転(送らない)
#     6. タスク keiba_odds を登録して、1回動かしてログを見せる
#
#   更新したいときも同じ1行を貼ればよい。
#   ★このファイルは BOM なしの UTF-8(irm で読むと BOM が混ざって1行目が壊れるため)。
#     ファイルとして直接動かさず、上の1行で使うこと。競輪(honda1986/keirin の pc\install.ps1)と同じ作り。

# ★$ErrorActionPreference は既定(Continue)のまま。Stop にすると、Windows PowerShell 5.1 では
#   git が標準エラーに書いただけ(進み具合の表示など)で止まってしまう
$URL = "https://raw.githubusercontent.com/honda1986/keeeiba-app/main/pc/install.ps1"
$GIT = "https://github.com/honda1986/keeeiba-app.git"
$ROOT = "C:\keiba"
$REPO = Join-Path $ROOT "keeeiba-app"
$LIVE = Join-Path $ROOT "odds-live"
$LOGS = Join-Path $ROOT "logs"

function Say($m) { Write-Host $m }
function Ok($m) { Write-Host "  ○ $m" -ForegroundColor Green }
function Ng($m) { Write-Host "  × $m" -ForegroundColor Red }
function Step($n, $t) { Write-Host ""; Write-Host "--- $n. $t" -ForegroundColor Cyan }
function Done($code) { Write-Host ""; Read-Host "Enter で閉じます" | Out-Null; exit $code }

# ---- 1. 管理者に切り替える(タスクの登録に要る) ----
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Say "管理者に切り替えます。「はい」を押してください(新しい窓で続きます)..."
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"irm $URL | iex`""
    return
}
$Host.UI.RawUI.WindowTitle = "競馬 オッズ取得 PC のセットアップ"
Say "競馬(中央)のオッズ取得 — PC のセットアップ"

# winget で入れたものを、この窓でもすぐ使えるようにする
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") +
                ";C:\Program Files\nodejs;C:\Program Files\Git\cmd;$env:LOCALAPPDATA\Programs\Git\cmd"
}
function Has($cmd) { Refresh-Path; return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function WingetInstall($id, $name) {
    Say "  $name を入れます(数分かかることがあります)..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Ng "winget がありません。Microsoft Store で「アプリ インストーラー」を入れてから、もう一度この1行を貼ってください"
        Done 1
    }
    winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements | Out-Host
}

# ---- 2. Git と Node.js ----
Step 2 "Git と Node.js"
if (-not (Has git)) { WingetInstall "Git.Git" "Git" }
if (-not (Has git)) { Ng "Git を入れられませんでした。PC を再起動してから、もう一度この1行を貼ってください"; Done 1 }
Ok (git --version)
if (-not (Has node)) { WingetInstall "OpenJS.NodeJS.LTS" "Node.js" }
if (-not (Has node)) { Ng "Node.js を入れられませんでした。PC を再起動してから、もう一度この1行を貼ってください"; Done 1 }
Ok ("Node.js " + (node --version))

# ---- 3. リポジトリ(pc フォルダだけ) ----
Step 3 "リポジトリ($REPO。pc フォルダだけ)"
# 改行を勝手に変えさせない(競輪・競艇と同じ。clone の前に)
if ((git config --global core.autocrlf) -ne "false") { git config --global core.autocrlf false; Ok "core.autocrlf を false にしました" }
New-Item -ItemType Directory -Force -Path $ROOT, $LOGS | Out-Null
$env:GIT_TERMINAL_PROMPT = "0"
if (-not (Test-Path (Join-Path $REPO ".git"))) {
    # 全部入れると数百MBあるので、いちばん新しい版の pc フォルダだけを入れる
    git clone --quiet --depth 1 --filter=blob:none --no-checkout $GIT $REPO 2>&1 | Out-Host
    if (-not (Test-Path (Join-Path $REPO ".git"))) { Ng "リポジトリを取れませんでした(ネットにつながっているか確認)"; Done 1 }
    git -C $REPO sparse-checkout set --no-cone "/pc/" 2>&1 | Out-Host
    git -C $REPO checkout --quiet main 2>&1 | Out-Host
} else {
    git -C $REPO fetch --quiet --depth 1 origin main 2>&1 | Out-Host
    git -C $REPO reset --quiet --hard FETCH_HEAD 2>&1 | Out-Host
}
if (-not (Test-Path (Join-Path $REPO "pc\runner.js"))) { Ng "pc フォルダを取り出せませんでした(上の赤い字をそのまま貼って相談してください)"; Done 1 }
Ok "入れました(最新)"

# ---- 4. 送る枝(odds-live)と、GitHub へ送れるか ----
Step 4 "GitHub へ送れるか"
if (-not (Test-Path (Join-Path $LIVE ".git"))) {
    git clone --quiet --depth 1 --single-branch -b odds-live $GIT $LIVE 2>&1 | Out-Null
    if (-not (Test-Path (Join-Path $LIVE ".git"))) {
        # 枝がまだ無い(最初の1回): 空の枝を作る
        New-Item -ItemType Directory -Force -Path $LIVE | Out-Null
        git -C $LIVE init --quiet 2>&1 | Out-Null
        git -C $LIVE remote add origin $GIT 2>&1 | Out-Null
        git -C $LIVE checkout --quiet --orphan odds-live 2>&1 | Out-Null
    }
}
git -C $LIVE rev-parse -q --verify HEAD 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    # まだ1つも無い: 「動いている」の印だけ入れて最初のまとまりを作る
    Set-Content -Path (Join-Path $LIVE "heartbeat.json") -Value '{"at":"","source":"PC"}' -Encoding ASCII
    git -C $LIVE add -A 2>&1 | Out-Null
    git -C $LIVE -c user.name=odds-live -c user.email=odds-live@users.noreply.github.com commit --quiet -m "odds live" 2>&1 | Out-Null
}
Remove-Item Env:GIT_TERMINAL_PROMPT
Say "  オッズを GitHub へ送るためのログインを確かめます。ログインの窓が出たら、そこでログインしてください。"
for ($try = 1; $try -le 5; $try++) {
    # 何も変えない送り方(--dry-run)で、ログインと権限だけ確かめる
    $out = git -C $LIVE push --dry-run -f origin HEAD:refs/heads/odds-live 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) { Ok "送れます"; break }
    Ng "まだ送れません: $(($out -split "`n" | Where-Object { $_.Trim() } | Select-Object -Last 1))"
    if ($try -eq 5) { Say "  5回試してもだめでした。あとで同じ1行をもう一度貼ってください(ここまでの準備は残ります)"; Done 1 }
    Say ""
    Say "  GitHub のトークンに keeeiba-app を足してください(競輪・競艇で使っているトークンを直すだけ):"
    Say "    1. いま開くページで、使っているトークンを選んで [Edit]"
    Say "    2. Repository access に honda1986/keeeiba-app を追加"
    Say "    3. Permissions の Contents を [Read and write]"
    Say "    4. 一番下の [Update]"
    Start-Process "https://github.com/settings/personal-access-tokens"
    Read-Host "  直したら Enter(もう一度確かめます)" | Out-Null
}

# ---- 5. 自己テストと試し運転 ----
Step 5 "自己テストと試し運転(送らない)"
Push-Location $REPO
& node pc\selftest.js
$testOk = ($LASTEXITCODE -eq 0)
if ($testOk) { & node pc\runner.js --dry --log-dir $LOGS }
Pop-Location
if (-not $testOk) { Ng "自己テストが通りませんでした(上の × をそのまま貼って相談してください)"; Done 1 }
Ok "試し運転まで済みました(開催の無い日・発走が近いレースの無い時間は、何も取らないのが正常です)"

# ---- 6. タスクの登録 ----
Step 6 "タスクの登録"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $REPO "pc\register_tasks.ps1")
if ($LASTEXITCODE -ne 0) { Ng "タスクを登録できませんでした(上の赤い字をそのまま貼って相談してください)"; Done 1 }
Say ""
Say "  試しに1回動かします(20秒ほど)..."
Start-ScheduledTask -TaskName keiba_odds
Start-Sleep -Seconds 20
$log = Join-Path $LOGS ("odds_" + (Get-Date -Format yyyyMMdd) + ".log")
if (Test-Path $log) { Say "  --- $log の終わりのほう ---"; Get-Content -Tail 12 -Encoding UTF8 $log | Out-Host }
else { Say "  ログはまだありません(08:40〜17:40 の外・開催の無い日は、何もしないのが正常です)" }

Say ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  できました。あとは毎日 08:40〜17:40 に1分おきで、ひとりでに動きます。" -ForegroundColor Green
Write-Host "  中央の開催がある日だけ、発走が近いレースのオッズを取って GitHub へ送ります。" -ForegroundColor Green
Write-Host "  ログ: $LOGS\odds_日付.log" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Done 0

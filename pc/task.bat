@echo off
rem Fetch JRA odds once (called every minute by Task Scheduler "keiba_odds").
rem   usage: task.bat            (normal)
rem          task.bat --dry      (fetch and print only, no push)
rem          task.bat --force    (run even outside 08:40-17:40 JST)
rem   repo = parent of this folder (C:\keiba\keeeiba-app), logs = C:\keiba\logs\odds_YYYYMMDD.log
setlocal
set "PATH=%PATH%;C:\Program Files\nodejs;%LOCALAPPDATA%\Programs\nodejs;C:\Program Files\Git\cmd;%LOCALAPPDATA%\Programs\Git\cmd"
cd /d "%~dp0.."
node pc\runner.js %* --log-dir "%~dp0..\..\logs"
exit /b %ERRORLEVEL%

@echo off
rem Set up phone notifications (ntfy) for picks a few minutes before each race. Just double-click.
rem   Saves C:\keiba\notify.json (outside the repo; the topic name is not pushed to GitHub) and sends one test message.
setlocal
chcp 65001 >nul
title keiba notify setup
set "PATH=%PATH%;C:\Program Files\nodejs;%LOCALAPPDATA%\Programs\nodejs;C:\Program Files\Git\cmd;%LOCALAPPDATA%\Programs\Git\cmd"
cd /d "%~dp0.."
node pc\notify.js --setup
set RC=%ERRORLEVEL%
echo.
pause
exit /b %RC%

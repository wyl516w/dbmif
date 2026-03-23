@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train.ps1" %*
exit /b %ERRORLEVEL%

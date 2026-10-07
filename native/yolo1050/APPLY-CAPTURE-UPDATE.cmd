@echo off
setlocal
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0apply_runtime_update.ps1"
if errorlevel 1 echo Update was not completed. Read the message above.
pause

@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0runtime;%PATH%"
echo INT8 detection view: boxes and scores, mouse movement and shooting disabled.
echo Uses the existing INT8 engine. Settings, confidence, and calibration are retained.
"runtime\yolo1050.exe" --config "settings.json" --precision int8 --view-detections
pause

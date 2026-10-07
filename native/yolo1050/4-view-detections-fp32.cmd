@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0runtime;%PATH%"
echo FP32 detection view: boxes and scores, mouse movement and shooting disabled.
echo Uses the existing FP32 engine. Settings, confidence, and calibration are retained.
"runtime\yolo1050.exe" --config "settings.json" --precision fp32 --view-detections
pause

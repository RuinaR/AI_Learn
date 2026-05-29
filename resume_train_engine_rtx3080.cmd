@echo off
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "TEMP=%ROOT%.tmp_runtime"
set "TMP=%ROOT%.tmp_runtime"
if not exist "%TEMP%" mkdir "%TEMP%"

py -3.12 scripts\check_train_export_env.py || exit /b 1

set "RUNS_DIR=%ROOT%runs\detect\runs"
set "LATEST_RUN="
for /f "delims=" %%D in ('powershell -NoProfile -Command "$dirs = Get-ChildItem -LiteralPath '%RUNS_DIR%' -Directory | Sort-Object LastWriteTime -Descending; if ($dirs) { $dirs[0].FullName }"') do (
  set "LATEST_RUN=%%D"
)

if not defined LATEST_RUN (
  echo No previous run directory found. Starting a fresh run.
  py -3.12 scripts\train_yolo.py --model yolo26x.pt --data datasets\hiyoung_ppe_local.yaml --device 0 --epochs 120 --imgsz 960 --batch 1 --workers 0 --name helmet_person_yolo26x_960_rtx3080 --cos-lr
  endlocal
  exit /b %errorlevel%
)

set "LAST_PT=%LATEST_RUN%\weights\last.pt"
for %%I in ("%LATEST_RUN%") do set "RUN_NAME=%%~nxI"

if exist "%LAST_PT%" (
  echo Resuming from: %LAST_PT%
  py -3.12 scripts\train_yolo.py --model "%LAST_PT%" --device 0 --resume
) else (
  echo last.pt not found in latest run. Restarting with run name: %RUN_NAME%
  py -3.12 scripts\train_yolo.py --model yolo26x.pt --data datasets\hiyoung_ppe_local.yaml --device 0 --epochs 120 --imgsz 960 --batch 1 --workers 0 --name "%RUN_NAME%" --exist-ok --cos-lr
)

endlocal

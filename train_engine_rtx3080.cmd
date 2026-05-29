@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "TEMP=%ROOT%.tmp_runtime"
set "TMP=%ROOT%.tmp_runtime"
if not exist "%TEMP%" mkdir "%TEMP%"
py -3.12 scripts\check_train_export_env.py || exit /b 1
py -3.12 scripts\train_yolo.py --model yolo26x.pt --data datasets\hiyoung_ppe_local.yaml --device 0 --epochs 120 --imgsz 960 --batch 1 --workers 0 --name helmet_person_yolo26x_960_rtx3080 --cos-lr
endlocal

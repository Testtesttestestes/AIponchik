@echo off
setlocal
cd /d "%~dp0\.."

rem Capture a lossless PNG screenshot via ADB, calculate the best move, and open
rem the overlay image. This does not tap or swipe on the phone.

set "PYTHON_CMD=python"
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

set "OUT_DIR=run_outputs"
if not "%AIPONCHIK_OUT_DIR%"=="" set "OUT_DIR=%AIPONCHIK_OUT_DIR%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "JSON_OUT=%OUT_DIR%\adb_capture.json"
set "OVERLAY_OUT=%OUT_DIR%\adb_MOVE.jpg"
set "SERIAL_ARG="
if not "%AIPONCHIK_SERIAL%"=="" set "SERIAL_ARG=--serial %AIPONCHIK_SERIAL%"

echo Reading the phone screen through ADB screencap...
echo Output JSON:    %JSON_OUT%
echo Output overlay: %OVERLAY_OUT%
echo.

%PYTHON_CMD% game_parser.py --adb %SERIAL_ARG% --stable-frames 2 --timeout 30 --out "%JSON_OUT%" --overlay "%OVERLAY_OUT%"
if errorlevel 1 (
    echo.
    echo Failed to calculate a move. Check that adb devices shows exactly one device in the 'device' state.
    pause
    exit /b 1
)

start "" "%OVERLAY_OUT%"

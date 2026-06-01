@echo off
setlocal
cd /d "%~dp0\.."

rem Persistent ADB GUI assistant. It reads lossless ADB screencaps, updates a
rem topmost OpenCV hint window, and never taps or swipes the phone.

set "PYTHON_CMD=python"
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

set "ADB_PATH=%AIPONCHIK_ADB%"
if "%ADB_PATH%"=="" (
    for /f "delims=" %%A in ('where adb 2^>nul') do if "%ADB_PATH%"=="" set "ADB_PATH=%%A"
)
if "%ADB_PATH%"=="" (
    for /f "delims=" %%S in ('where scrcpy 2^>nul') do if "%ADB_PATH%"=="" if exist "%%~dpSadb.exe" set "ADB_PATH=%%~dpSadb.exe"
)
if "%ADB_PATH%"=="" (
    echo Could not find adb.exe.
    echo Install Android platform-tools, add adb.exe to PATH, or run:
    echo   set AIPONCHIK_ADB=C:\path\to\adb.exe
    pause
    exit /b 1
)

set "OUT_DIR=run_outputs"
if not "%AIPONCHIK_OUT_DIR%"=="" set "OUT_DIR=%AIPONCHIK_OUT_DIR%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "JSON_OUT=%OUT_DIR%\adb_capture.json"
set "OVERLAY_OUT=%OUT_DIR%\adb_MOVE.jpg"
set "SERIAL_ARG="
if not "%AIPONCHIK_SERIAL%"=="" set "SERIAL_ARG=--serial %AIPONCHIK_SERIAL%"

set "STABLE_FRAMES=2"
if not "%AIPONCHIK_STABLE_FRAMES%"=="" set "STABLE_FRAMES=%AIPONCHIK_STABLE_FRAMES%"
set "TIMEOUT_SECONDS=30"
if not "%AIPONCHIK_TIMEOUT%"=="" set "TIMEOUT_SECONDS=%AIPONCHIK_TIMEOUT%"

echo Starting persistent ADB GUI assistant...
echo ADB:           %ADB_PATH%
echo Output JSON:   %JSON_OUT%
echo Saved overlay: %OVERLAY_OUT%
echo Close the GUI with q or Esc. This script does not control the phone.
echo.

%PYTHON_CMD% game_parser.py --adb --gui --adb-path "%ADB_PATH%" %SERIAL_ARG% --stable-frames %STABLE_FRAMES% --timeout %TIMEOUT_SECONDS% --out "%JSON_OUT%" --overlay "%OVERLAY_OUT%"
if errorlevel 1 (
    echo.
    echo ADB GUI assistant failed. Check that adb devices shows exactly one device in the 'device' state.
    pause
    exit /b 1
)

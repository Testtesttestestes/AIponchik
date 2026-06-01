@echo off
setlocal
cd /d "%~dp0\.."

rem Capture a lossless PNG screenshot via ADB, calculate the best move, and open
rem the overlay image. This does not tap or swipe on the phone.

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

echo Reading the phone screen through ADB screencap...
echo ADB:           %ADB_PATH%
echo Output JSON:   %JSON_OUT%
echo Saved overlay: %OVERLAY_OUT%
echo Overlay window will open after calculation; press any key in it to close.
echo.

%PYTHON_CMD% game_parser.py --adb --adb-path "%ADB_PATH%" %SERIAL_ARG% --stable-frames 2 --timeout 30 --out "%JSON_OUT%" --overlay "%OVERLAY_OUT%" --show-overlay-window --window-ms 0
if errorlevel 1 (
    echo.
    echo Failed to calculate a move. Check that adb devices shows exactly one device in the 'device' state.
    pause
    exit /b 1
)

rem The overlay was shown by OpenCV. The saved image remains available at %OVERLAY_OUT%.

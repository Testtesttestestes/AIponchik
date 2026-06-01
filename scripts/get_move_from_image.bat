@echo off
setlocal
cd /d "%~dp0\.."

if "%~1"=="" (
    echo Usage: scripts\get_move_from_image.bat path\to\screenshot.jpg
    echo Example: scripts\get_move_from_image.bat test_images\Screenshot_2026-06-01-13-30-11-51.jpg
    exit /b 2
)

set "PYTHON_CMD=python"
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

set "OUT_DIR=run_outputs"
if not "%AIPONCHIK_OUT_DIR%"=="" set "OUT_DIR=%AIPONCHIK_OUT_DIR%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "JSON_OUT=%OUT_DIR%\image_capture.json"
set "OVERLAY_OUT=%OUT_DIR%\image_MOVE.jpg"

%PYTHON_CMD% game_parser.py --image "%~1" --out "%JSON_OUT%" --overlay "%OVERLAY_OUT%"
if errorlevel 1 (
    echo.
    echo Failed to calculate a move from "%~1".
    pause
    exit /b 1
)

start "" "%OVERLAY_OUT%"

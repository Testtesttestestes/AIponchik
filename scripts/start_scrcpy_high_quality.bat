@echo off
setlocal

rem High-quality, read-only scrcpy mirror for inspecting the game manually.
rem The solver does not read pixels from the Direct3D renderer; use get_move_adb.bat
rem for a full-resolution PNG capture from the phone when you need a move.

set "SCRCPY=scrcpy"
set "VIDEO_BIT_RATE=32M"
set "VIDEO_CODEC=h264"
set "MAX_FPS=60"
set "RENDER_DRIVER=direct3d"

if not "%AIPONCHIK_SCRCPY%"=="" set "SCRCPY=%AIPONCHIK_SCRCPY%"
if not "%AIPONCHIK_BITRATE%"=="" set "VIDEO_BIT_RATE=%AIPONCHIK_BITRATE%"
if not "%AIPONCHIK_CODEC%"=="" set "VIDEO_CODEC=%AIPONCHIK_CODEC%"
if not "%AIPONCHIK_MAX_FPS%"=="" set "MAX_FPS=%AIPONCHIK_MAX_FPS%"
if not "%AIPONCHIK_RENDER_DRIVER%"=="" set "RENDER_DRIVER=%AIPONCHIK_RENDER_DRIVER%"

if not "%~1"=="" set "VIDEO_BIT_RATE=%~1"
if not "%~2"=="" set "VIDEO_CODEC=%~2"

echo Starting scrcpy in read-only high-quality mode...
echo   bitrate: %VIDEO_BIT_RATE%
echo   codec:   %VIDEO_CODEC%
echo   fps:     %MAX_FPS%
echo.
echo Tip: if h264 still looks blocky, try:  start_scrcpy_high_quality.bat 48M h265
echo.

"%SCRCPY%" --no-control --no-audio --video-codec=%VIDEO_CODEC% --video-bit-rate=%VIDEO_BIT_RATE% --max-fps=%MAX_FPS% --render-driver=%RENDER_DRIVER%
if errorlevel 1 (
    echo.
    echo New bitrate flag failed; retrying with the legacy -b flag...
    "%SCRCPY%" --no-control --no-audio --video-codec=%VIDEO_CODEC% -b %VIDEO_BIT_RATE% --max-fps=%MAX_FPS% --render-driver=%RENDER_DRIVER%
)

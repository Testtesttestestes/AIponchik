@echo off
setlocal
cd /d "%~dp0\.."

if not exist "run_outputs" mkdir "run_outputs"
python game_parser.py --adb --gui --out run_outputs/adb_capture.json --overlay run_outputs/adb_MOVE.jpg

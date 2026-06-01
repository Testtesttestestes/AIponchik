@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON_CMD=python"
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

if not exist "run_outputs" mkdir "run_outputs"
%PYTHON_CMD% game_parser.py --simulate --simulate-games 50 --out run_outputs/simulation_report.json

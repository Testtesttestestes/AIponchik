@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON_CMD=python"
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

if not exist "models" mkdir "models"
if not exist "run_outputs" mkdir "run_outputs"
%PYTHON_CMD% train_policy_network.py --out models/policy_network.json --metrics-out run_outputs/policy_training_metrics.json

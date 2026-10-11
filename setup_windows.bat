@echo off
cd /d "%~dp0"
py -3.13 setup.py
if errorlevel 1 (
  echo Setup failed. See the error above and START_HERE.txt.
  pause
  exit /b 1
)
pause

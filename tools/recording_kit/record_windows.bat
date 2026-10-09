@echo off
cd /d "%~dp0"
set "KIT_PY="
where py >nul 2>nul && set "KIT_PY=py -3"
if not defined KIT_PY (
  where python >nul 2>nul && set "KIT_PY=python"
)
if not defined KIT_PY (
  echo Install Python 3.10 or newer and enable Add Python to PATH. See START_HERE.txt.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  %KIT_PY% -m venv .venv
  if errorlevel 1 goto failed
)
echo Checking/installing dependencies - internet needed the first time...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" record.py
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Setup or recording failed. See START_HERE.txt. Fix the issue and run this launcher again.
pause
exit /b 1

@echo off
rem Double-click to start ResQGraph on Windows. Works from any folder, including ones with spaces or brackets in the name.
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py)
echo Using: %PY%
%PY% --version
echo.
echo Checking / installing the packages the app needs (the first time this takes a few minutes)...
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo The install failed. Read the message above. Python 3.10 to 3.13 from python.org works.
  pause
  exit /b 1
)
echo.
echo Starting ResQGraph. Your browser should open; if not, open the Local URL shown below.
%PY% -m streamlit run app.py
pause

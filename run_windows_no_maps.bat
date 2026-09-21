@echo off
rem Use this if the normal launcher makes the screen blank, freeze or crawl: it starts ResQGraph with all map pictures switched off.
rem Every result (routes, verdicts, tables, findings) is still there; only the map drawings are replaced by a note.
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py)
set RESQGRAPH_VISUALS=off
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
  echo The install failed. Read the message above.
  pause
  exit /b 1
)
%PY% -m streamlit run app.py
pause

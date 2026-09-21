#!/usr/bin/env bash
# Start ResQGraph on macOS / Linux:  bash run_mac_linux.sh
cd "$(dirname "$0")" || exit 1
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
$PY --version
$PY -m pip install -r requirements.txt || { echo "Install failed. Read the message above."; exit 1; }
exec $PY -m streamlit run app.py

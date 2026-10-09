#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null 2>&1 || { echo 'Install Python 3.10 or newer, then try again. See START_HERE.txt.'; exit 1; }
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
echo 'Checking/installing dependencies (internet needed the first time)...'
.venv/bin/python -m pip install -r requirements.txt || { echo 'Installation failed. Check internet and run this launcher again.'; exit 1; }
.venv/bin/python record.py

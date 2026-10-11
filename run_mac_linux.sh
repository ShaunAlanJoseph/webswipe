#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run: sh setup_mac_linux.sh first."
  exit 1
fi
exec .venv/bin/python run_demo.py "$@"

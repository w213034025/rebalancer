#!/usr/bin/env bash
# One-command runner.   ./run.sh [dashboard|test [--demo]|check|backtest]   (default: dashboard)
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || { echo ">> Creating virtualenv (.venv)..."; python3 -m venv .venv; }
# (Re)install whenever requirements.txt is newer than the last install.
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo ">> Installing requirements..."
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  touch .venv/.installed
fi
[ -f .env ] && set -a && . ./.env && set +a      # optional; dashboard/tests need no keys

case "${1:-dashboard}" in
  dashboard) exec .venv/bin/streamlit run dashboard.py ;;
  test)      if [ "${2:-}" = "--demo" ]; then
               # verbose names + plain-English narration lines (for presenting)
               TEST_DEMO=1 exec .venv/bin/python -m pytest -v -s -p no:cacheprovider
             else exec .venv/bin/python -m pytest -q; fi ;;
  check)     exec .venv/bin/python main.py check ;;        # dry-run; needs network (+ keys for broker)
  backtest)  exec .venv/bin/python main.py backtest --compare-ml ;;
  *) echo "usage: ./run.sh [dashboard|test [--demo]|check|backtest]"; exit 1 ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required but was not found on PATH."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required but was not found on PATH."
  echo "Install it first (e.g., 'sudo apt install ffmpeg' on Debian/Ubuntu)."
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if ! .venv/bin/python -c "import tkinter" >/dev/null 2>&1; then
  echo "ERROR: tkinter is not available for this Python installation."
  echo "Install system package python3-tk (Debian/Ubuntu) or python3-tkinter (Fedora/RHEL)."
  exit 1
fi

echo "Environment ready."
echo "Activate with: source .venv/bin/activate"

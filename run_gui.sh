#!/usr/bin/env bash
set -euo pipefail
lingbot_gui_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$lingbot_gui_root"
unset PYTHONPATH
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
exec "$lingbot_gui_root/.venv-optimization/bin/python" -u -m tools.realsense_gui.app "$@"

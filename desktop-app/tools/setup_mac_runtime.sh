#!/usr/bin/env bash
# Встроенный Python для Mac (Apple Silicon) с MLX-whisper для GPU-транскрипции.
set -euo pipefail

PYTHON_BIN="${1:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DESKTOP_APP_DIR}/.." && pwd)"
RUNTIME_DIR="${DESKTOP_APP_DIR}/src-tauri/runtime"
VENV_DIR="${RUNTIME_DIR}/mac-venv"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup_mac_runtime.sh: пропуск (не macOS)" >&2
  exit 0
fi

echo "Runtime dir: ${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}"

if [[ -d "${VENV_DIR}" ]]; then
  echo "Removing old mac-venv..."
  rm -rf "${VENV_DIR}"
fi

echo "Creating mac-venv with MLX-whisper..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "Venv python not found: ${VENV_PYTHON}" >&2
  exit 1
fi

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install \
  "mlx-whisper" \
  "mlx" \
  "huggingface_hub" \
  "faster-whisper"

echo "Copying pipeline scripts into runtime..."
cp "${PROJECT_ROOT}/run_pipeline_cli.py" "${RUNTIME_DIR}/"
cp "${PROJECT_ROOT}/pipeline_core.py" "${RUNTIME_DIR}/"

echo "Verifying MLX-whisper..."
"${VENV_PYTHON}" -c "import mlx_whisper; print('mlx_whisper OK')"

echo "Done: ${VENV_DIR}/bin/python"

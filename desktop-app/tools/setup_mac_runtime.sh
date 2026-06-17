#!/usr/bin/env bash
# Переносимый Python для Mac (Apple Silicon) + MLX-whisper.
# venv из CI не работает на чужих Mac — ссылается на Python.framework сборочной машины.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DESKTOP_APP_DIR}/.." && pwd)"
RUNTIME_DIR="${DESKTOP_APP_DIR}/src-tauri/runtime"
PYTHON_DIR="${RUNTIME_DIR}/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup_mac_runtime.sh: пропуск (не macOS)" >&2
  exit 0
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "setup_mac_runtime.sh: bundled python только для Apple Silicon (arm64)" >&2
  exit 1
fi

# python-build-standalone: relocatable CPython, не требует системного Python.framework
PB_RELEASE="20251202"
PB_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PB_RELEASE}/cpython-3.11.14%2B${PB_RELEASE}-aarch64-apple-darwin-install_only_stripped.tar.gz"

echo "Runtime dir: ${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}"

rm -rf "${PYTHON_DIR}" "${RUNTIME_DIR}/mac-venv"

echo "Downloading portable Python (${PB_RELEASE}, aarch64-apple-darwin)..."
TMP_TAR="$(mktemp -t pbs-python.XXXXXX.tar.gz)"
curl -fsSL -o "${TMP_TAR}" "${PB_URL}"
tar -xzf "${TMP_TAR}" -C "${RUNTIME_DIR}"
rm -f "${TMP_TAR}"

PYTHON_BIN="${PYTHON_DIR}/bin/python3"
if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "Portable python not found after extract: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Installing MLX stack into bundled python..."
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --default-timeout=120 \
  "mlx-whisper" \
  "mlx" \
  "huggingface_hub" \
  "faster-whisper"

echo "Copying pipeline scripts into runtime..."
cp "${PROJECT_ROOT}/run_pipeline_cli.py" "${RUNTIME_DIR}/"
cp "${PROJECT_ROOT}/pipeline_core.py" "${RUNTIME_DIR}/"

echo "Verifying bundled python + MLX..."
"${PYTHON_BIN}" -c "import mlx_whisper; print('mlx_whisper OK')"

echo "Done: ${PYTHON_BIN}"

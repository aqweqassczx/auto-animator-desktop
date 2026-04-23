#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DESKTOP_APP_DIR}/.." && pwd)"
RUNTIME_DIR="${DESKTOP_APP_DIR}/src-tauri/runtime"
BUILD_DIR="${DESKTOP_APP_DIR}/.build-pipeline-runner"
VENV_DIR="${BUILD_DIR}/venv"

echo "Desktop app dir: ${DESKTOP_APP_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Preparing runtime dir: ${RUNTIME_DIR}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${RUNTIME_DIR}"

echo "Creating isolated virtual environment..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "Venv python not found: ${VENV_PYTHON}" >&2
  exit 1
fi

echo "Installing minimal build dependencies into venv..."
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install pyinstaller faster-whisper

ENTRY_SCRIPT="${PROJECT_ROOT}/run_pipeline_cli.py"
if [[ ! -f "${ENTRY_SCRIPT}" ]]; then
  echo "Entry script not found: ${ENTRY_SCRIPT}" >&2
  exit 1
fi

echo "Building pipeline_runner..."
"${VENV_PYTHON}" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --collect-all faster_whisper \
  --collect-submodules faster_whisper \
  --exclude-module torch \
  --exclude-module torchvision \
  --exclude-module torchaudio \
  --exclude-module tensorflow \
  --exclude-module sklearn \
  --exclude-module scipy \
  --exclude-module pandas \
  --exclude-module matplotlib \
  --exclude-module spacy \
  --exclude-module nltk \
  --exclude-module moviepy \
  --name pipeline_runner \
  --distpath "${BUILD_DIR}/dist" \
  --workpath "${BUILD_DIR}/work" \
  --specpath "${BUILD_DIR}" \
  --paths "${PROJECT_ROOT}" \
  "${ENTRY_SCRIPT}"

BUILT_BIN="${BUILD_DIR}/dist/pipeline_runner"
if [[ ! -f "${BUILT_BIN}" ]]; then
  echo "Build output not found: ${BUILT_BIN}" >&2
  exit 1
fi

cp "${BUILT_BIN}" "${RUNTIME_DIR}/pipeline_runner"

echo "Running smoke-check for pipeline_runner..."
"${RUNTIME_DIR}/pipeline_runner" --help >/dev/null

echo "Done: ${RUNTIME_DIR}/pipeline_runner"

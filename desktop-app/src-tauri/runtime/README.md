# Runtime (bundled pipeline)

## Windows
- `pipeline_runner.exe` — CPU-only fallback (PyInstaller)
- `run_pipeline_cli.py` + `pipeline_core.py` — для Python + CUDA

## macOS Apple Silicon (M1/M2/M3)
- `mac-venv/` — встроенный Python с **mlx-whisper** (GPU через MLX)
- Собирается: `bash tools/setup_mac_runtime.sh`
- На Mac **не используется** медленный CPU-only `pipeline_runner` — только MLX

## Сборка
- Windows: `tools/build_pipeline_runner.ps1`
- macOS: `tools/build_pipeline_runner.sh` (на arm64 автоматически вызывает setup_mac_runtime.sh)

Не коммить большие бинарники и `mac-venv/` вручную — генерируются при сборке.

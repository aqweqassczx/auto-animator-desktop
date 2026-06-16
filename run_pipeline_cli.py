import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from pipeline_core import ensure_apple_silicon_mlx_runtime, log_whisper_runtime_plan, run_pipeline_from_json


def _configure_stdio_utf8() -> None:
    """GUI/piped stdout on Windows often defaults to cp1251 — force UTF-8 for logs/JSON."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")
            sys.stdout.buffer.flush()
        else:
            print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _print_json_line(payload: dict) -> None:
    _safe_print(json.dumps(payload, ensure_ascii=False))


def _classify_error(exc: Exception) -> tuple[str, int, str]:
    text = str(exc)
    low = text.lower()

    if "unable to open file 'model.bin'" in low:
        return (
            "WHISPER_MODEL_CACHE",
            21,
            "Поврежден или недокачан кэш Whisper-модели. Перезапусти задачу для повторной загрузки модели.",
        )

    if "mlx-whisper" in low or "mlx_whisper" in low:
        return (
            "MLX_WHISPER_MISSING",
            25,
            "На Mac Apple Silicon нужен mlx-whisper для GPU. Перезапусти задачу — установка должна пройти автоматически.",
        )

    if "no such file or directory" in low and ("ffmpeg" in low or "ffprobe" in low):
        return (
            "FFMPEG_NOT_FOUND",
            22,
            "Не найден ffmpeg/ffprobe. Проверь, что бинарники доступны приложению.",
        )

    if "no such file or directory" in low:
        return (
            "MISSING_INPUT_FILE",
            23,
            "Не найден входной файл или путь. Проверь аудио, сценарий и папки медиа.",
        )

    if "permission denied" in low or "access is denied" in low:
        return (
            "ACCESS_DENIED",
            24,
            "Нет прав доступа к файлам/папкам. Запусти с нужными правами или смени папку результата.",
        )

    return (
        "PIPELINE_RUNTIME_ERROR",
        1,
        "Внутренняя ошибка пайплайна. Смотри traceback ниже.",
    )


def _prepare_media_tool_env() -> None:
    """
    Make ffmpeg/ffprobe discoverable when app is launched as GUI on macOS.
    GUI apps often start with a minimal PATH that does not include /usr/local/bin
    or /opt/homebrew/bin, so subprocess('ffmpeg') fails even when it is installed.
    """
    extra_dirs = [
        os.path.expanduser("~/bin"),
        "/usr/local/bin",
        "/opt/homebrew/bin",
    ]
    current_path = os.environ.get("PATH", "")
    path_parts = [p for p in current_path.split(os.pathsep) if p]
    for directory in extra_dirs:
        if directory and directory not in path_parts:
            path_parts.append(directory)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    # Explicit fallback for code paths that may read dedicated env vars.
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and "FFMPEG_BIN" not in os.environ:
        os.environ["FFMPEG_BIN"] = ffmpeg
    if ffprobe and "FFPROBE_BIN" not in os.environ:
        os.environ["FFPROBE_BIN"] = ffprobe


def _prepare_hf_auth() -> None:
    """
    Configure Hugging Face token for model downloads.
    Priority:
    1) Existing process env vars.
    2) Token file in user profile: ~/.auto_animator/hf_token.txt
    """
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if not token:
        token_file = Path.home() / ".auto_animator" / "hf_token.txt"
        if token_file.is_file():
            try:
                token = token_file.read_text(encoding="utf-8").strip()
            except Exception:
                token = ""

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print(
            "HF token найден (используется только если модель нужно скачать).",
            flush=True,
        )
    else:
        print(
            "HF token не задан (скачивание моделей — анонимно, если понадобится).",
            flush=True,
        )


def main() -> int:
    _configure_stdio_utf8()
    _prepare_media_tool_env()
    _prepare_hf_auth()
    ensure_apple_silicon_mlx_runtime()
    log_whisper_runtime_plan()
    # Use stable HTTP download path in bundled app.
    # Xet acceleration is optional and often unavailable in end-user setups.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    parser = argparse.ArgumentParser(description="Auto Animator pipeline CLI")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--result-file", required=False, help="Path to JSON result file")
    args = parser.parse_args()

    def _write_result(payload: dict) -> None:
        result_file = getattr(args, "result_file", None)
        if not result_file:
            return
        try:
            result_path = Path(result_file)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            # Do not shadow the original pipeline result if writing fails.
            pass

    try:
        result = run_pipeline_from_json(args.config)
        payload = {"ok": True, "result": result}
        _write_result(payload)
        _print_json_line(payload)
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        error_code, exit_code, hint = _classify_error(exc)
        payload = {
            "ok": False,
            "error_code": error_code,
            "hint": hint,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "exit_code": exit_code,
        }
        _write_result(payload)
        _print_json_line(payload)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

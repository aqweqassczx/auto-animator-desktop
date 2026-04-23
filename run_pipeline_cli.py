import argparse
import json
import os
import shutil
import traceback
from pathlib import Path

from pipeline_core import run_pipeline_from_json


def _classify_error(exc: Exception) -> tuple[str, int, str]:
    text = str(exc)
    low = text.lower()

    if "unable to open file 'model.bin'" in low:
        return (
            "WHISPER_MODEL_CACHE",
            21,
            "Поврежден или недокачан кэш Whisper-модели. Перезапусти задачу для повторной загрузки модели.",
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
        print("HF token detected: using authenticated model download.", flush=True)
    else:
        print("HF token not found: using anonymous model download.", flush=True)


def main() -> int:
    _prepare_media_tool_env()
    _prepare_hf_auth()
    # Use stable HTTP download path in bundled app.
    # Xet acceleration is optional and often unavailable in end-user setups.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    parser = argparse.ArgumentParser(description="Auto Animator pipeline CLI")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        result = run_pipeline_from_json(args.config)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        error_code, exit_code, hint = _classify_error(exc)
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code": error_code,
                    "hint": hint,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

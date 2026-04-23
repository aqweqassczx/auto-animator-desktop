import argparse
import json
import os
import shutil
import traceback

from pipeline_core import run_pipeline_from_json


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


def main() -> int:
    _prepare_media_tool_env()
    parser = argparse.ArgumentParser(description="Auto Animator pipeline CLI")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        result = run_pipeline_from_json(args.config)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import traceback

from pipeline_core import run_pipeline_from_json


def main() -> int:
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

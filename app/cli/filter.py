"""CLI for strict filtering; outputs all eligible candidates before ranking."""
import argparse
import json
import sys

from ..data_loader import DEFAULT_CSV, DataValidationError, load_contractors
from ..filtering import RequestValidationError, filter_contractors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python main.py filter", description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--city", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--duration-hours")
    parser.add_argument("--language")
    args = vars(parser.parse_args(argv))
    try:
        contractors = load_contractors(args.pop("csv"))
        result = filter_contractors(contractors, args)
    except (DataValidationError, RequestValidationError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

"""Run the website or CLI: python main.py [serve|recommend|filter] [options]."""
import sys

from app.cli import main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

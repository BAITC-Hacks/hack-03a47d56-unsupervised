"""Command dispatcher for the single project entry point, main.py."""
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Подбор подрядчиков: сайт, рекомендации и проверка фильтров.",
        epilog="Без команды запускается сайт. Справка команды: python main.py recommend --help",
    )
    parser.add_argument(
        "command", nargs="?", default="serve", choices=("serve", "recommend", "filter"),
        help="serve — сайт; recommend — до трёх рекомендаций; filter — все допустимые кандидаты",
    )
    # Each command owns its flags and help. Import lazily so that top-level help
    # and CLI validation don't start the server or load its catalog.
    command = parser.parse_args(arguments[:1]).command
    if command == "serve":
        from ..server import main as run
    elif command == "recommend":
        from .recommend import main as run
    else:
        from .filter import main as run
    return run(arguments[1:]) or 0

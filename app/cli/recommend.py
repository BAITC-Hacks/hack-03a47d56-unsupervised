"""CLI: strict filters, AI ranking and factual top-3 explanations."""
import argparse
import json
import sys

from ..data_loader import DEFAULT_CSV, DataValidationError, load_contractors
from ..embeddings import EmbeddingError
from ..filtering import RequestValidationError, filter_contractors
from ..recommendations import recommend_from_filtered
from ..scorer import RankingEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python main.py recommend", description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--city", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--duration-hours")
    parser.add_argument("--language")
    parser.add_argument("--preferences", default="")
    parser.add_argument("--offline", action="store_true", help="Сравнение по словам без API")
    parser.add_argument("--require-ai", action="store_true", help="Код 3, если вместо AI использован fallback")
    parser.add_argument("--warm-cache", action="store_true", help="Сначала подготовить эмбеддинги всего каталога")
    args = vars(parser.parse_args(argv))
    try:
        catalog = load_contractors(args.pop("csv"))
        offline, require_ai, warm_cache = args.pop("offline"), args.pop("require_ai"), args.pop("warm_cache")
        if offline and (require_ai or warm_cache):
            raise ValueError("--offline несовместим с --require-ai и --warm-cache")
        preferences = args.pop("preferences")
        engine = RankingEngine(mode="offline" if offline else "auto")
        filtered = filter_contractors(catalog, args)
        if warm_cache:
            engine.embedder.embed([c["description"] for c in catalog if c["description"].strip()])
        result = recommend_from_filtered(filtered, preferences=preferences, engine=engine)
    except (DataValidationError, RequestValidationError, EmbeddingError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 3 if require_ai and result["cards"] and result["ai"]["mode"] != "openai" else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

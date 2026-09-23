"""Local browser demo for the contractor recommendation pipeline.

Run: python web_preview_server.py
Open: http://127.0.0.1:8765
Only the local machine can connect by default. The API key stays server-side.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from data_loader import CALENDAR_END, CALENDAR_START, load_contractors
from embeddings import EmbeddingError, OpenAIEmbeddings
from filtering import RequestValidationError, filter_contractors
from recommendations import recommend_from_filtered
from scorer import RankingEngine


ROOT = Path(__file__).resolve().parent
LOCAL_ENV = ROOT / ".env"
WEB_RANK_CACHE = ROOT / ".cache" / "web_preview_rankings.sqlite3"
ASSETS = {
    "/": (ROOT / "web_preview" / "index.html", "text/html; charset=utf-8"),
    "/index.html": (ROOT / "web_preview" / "index.html", "text/html; charset=utf-8"),
    "/style.css": (ROOT / "web_preview" / "style.css", "text/css; charset=utf-8"),
    "/app.js": (ROOT / "web_preview" / "app.js", "text/javascript; charset=utf-8"),
    "/catalog.js": (ROOT / "web_preview" / "catalog.js", "text/javascript; charset=utf-8"),
    "/hero.png": (ROOT / "web_preview" / "hero.png", "image/png"),
}
CATALOG = load_contractors()
ENGINE = RankingEngine(cache_path=WEB_RANK_CACHE)
ENV_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ENGINE_KEY = ENV_API_KEY
ENGINE_LOCK = RLock()
META = {
    "cities": sorted({item["city"] for item in CATALOG}),
    "categories": sorted({category for item in CATALOG for category in item["categories"]}),
    "event_types": sorted({event_type for item in CATALOG for event_type in item["event_formats"]}),
    "languages": sorted({language for item in CATALOG for language in item["languages"]}),
    "calendar": {"start": CALENDAR_START.isoformat(), "end": CALENDAR_END.isoformat()},
    "profiles": len(CATALOG),
}


def configured_api_key() -> str:
    """Prefer an inherited key; also accept one from the ignored local .env file."""
    if ENV_API_KEY:
        return ENV_API_KEY
    try:
        lines = LOCAL_ENV.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ""
    for line in lines:
        if line.strip().startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return ""


def current_engine() -> RankingEngine:
    """Pick up local .env changes on the next request without logging the key."""
    global ENGINE, ENGINE_KEY
    key = configured_api_key()
    with ENGINE_LOCK:
        if key != ENGINE_KEY:
            ENGINE = (RankingEngine(embedder=OpenAIEmbeddings(api_key=key), cache_path=WEB_RANK_CACHE)
                      if key else RankingEngine(cache_path=WEB_RANK_CACHE))
            ENGINE_KEY = key
        return ENGINE


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"Недопустимое число в JSON: {value}.")


def public_result(result: dict) -> dict:
    """Expose the result needed by the UI, without the full contractor catalog."""
    card_fields = (
        "id", "name", "category", "city", "price_from_kzt", "explanation",
        "score", "score_details", "evidence", "warnings", "synthetic",
        "city_imputed", "price_imputed",
    )
    return {
        "status": result["status"],
        "message": result["message"],
        "request": result["request"],
        "cards": [{key: card[key] for key in card_fields} for card in result["cards"]],
        "counts": result["counts"],
        "reason_counts": result["reason_counts"],
        "excluded": [
            {
                "anon_name": item["anon_name"],
                "reasons": [reason["message"] for reason in item["reasons"]],
            }
            for item in result["excluded"]
        ],
        "ai": result["ai"],
    }


class PreviewHandler(BaseHTTPRequestHandler):
    def send_content(self, status: int, data: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
        except ConnectionError:
            # Switching presets aborts the previous browser request. Its result
            # may still be cached, but there is no client to send an error to.
            self.close_connection = True

    def send_json(self, status: int, payload: dict) -> None:
        self.send_content(
            status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802 - standard library handler API
        path = urlsplit(self.path).path
        if path == "/api/meta":
            self.send_json(200, {**META, "ai_available": bool(configured_api_key())})
        elif path == "/api/catalog":
            fields = ("id", "anon_name", "categories", "city", "price_from_kzt", "languages",
                      "event_formats", "max_hours", "description", "synthetic", "city_imputed", "price_imputed")
            self.send_json(200, {"profiles": [{key: item[key] for key in fields} for item in CATALOG]})
        elif path == "/healthz":
            self.send_json(200, {"status": "ok"})
        elif path in ASSETS:
            file_path, content_type = ASSETS[path]
            try:
                self.send_content(200, file_path.read_bytes(), content_type)
            except OSError:
                self.send_json(500, {"error": "Не удалось открыть страницу демо."})
        else:
            self.send_json(404, {"error": "Страница не найдена."})

    def do_POST(self) -> None:  # noqa: N802 - standard library handler API
        if urlsplit(self.path).path != "/api/recommend":
            self.send_json(404, {"error": "Адрес не найден."})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self.send_json(415, {"error": "Ожидается JSON-запрос."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_384:
                raise ValueError("Размер запроса должен быть от 1 до 16384 байт.")
            body = self.rfile.read(length)
            query = json.loads(body.decode("utf-8"), parse_constant=reject_nonfinite)
            if not isinstance(query, dict):
                raise ValueError("Запрос должен быть объектом JSON.")
            filtered = filter_contractors(CATALOG, query)
            result = recommend_from_filtered(
                filtered, preferences=query.get("preferences"), engine=current_engine()
            )
            self.send_json(200, public_result(result))
        except (ValueError, UnicodeDecodeError, RequestValidationError) as exc:
            self.send_json(400, {"error": str(exc)})
        except (EmbeddingError, OSError) as exc:
            self.log_error("Recommendation failed: %s", exc)
            self.send_json(500, {"error": "Подбор временно недоступен. Попробуйте ещё раз."})
        except Exception as exc:
            self.log_error("Unexpected recommendation error: %s", exc)
            self.send_json(500, {"error": "Произошла ошибка подбора. Попробуйте ещё раз."})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local contractor matching website")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

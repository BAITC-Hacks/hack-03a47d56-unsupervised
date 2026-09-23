"""Exercise the browser's HTTP contract without API keys or external services."""
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.embeddings import EmbeddingError
from app.scorer import RankingEngine
from app import server as app


class UnavailableEmbeddings:
    model = "test-unavailable"

    def embed(self, texts):
        raise EmbeddingError("AI недоступен")


class QuietHandler(app.PreviewHandler):
    def log_message(self, *_args):
        pass


class WebPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.engine = RankingEngine(mode="offline", cache_path=None)
        self.engine_patch = patch.object(app, "current_engine", return_value=self.engine)
        self.engine_patch.start()
        self.addCleanup(self.engine_patch.stop)

    def post(self, changes=None, *, raw=None):
        query = dict(city="Алматы", date="2026-09-23", event_type="свадьба",
                     category="Ведущий", budget=2_000_000)
        query.update(changes or {})
        payload = json.dumps(query).encode("utf-8") if raw is None else raw
        request = Request(self.url + "/api/recommend", data=payload,
                          headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_fractional_durations_are_preserved_through_http(self):
        for duration in (0.5, 1.25):
            with self.subTest(duration=duration):
                status, result = self.post({"duration_hours": duration})
                self.assertEqual(status, 200)
                self.assertEqual(result["request"]["duration_hours"], duration)
                self.assertEqual(len(result["cards"]), 3)
                for card in result["cards"]:
                    limit = next(fact["value"] for fact in card["evidence"]
                                 if fact["field"] == "max_hours")
                    self.assertTrue(limit is None or limit >= duration)

    def test_nonpositive_duration_is_an_input_error(self):
        for duration in (0, -0.5):
            with self.subTest(duration=duration):
                status, result = self.post({"duration_hours": duration})
                self.assertEqual(status, 400)
                self.assertIn("duration_hours", result["error"])

    def test_three_outcomes_and_shortage_reach_the_browser(self):
        for changes, outcome, count in [
            ({}, "matched", 3),
            ({"category": "Флорист", "date": "2026-10-04"}, "matched", 2),
            ({"category": "Флорист", "city": "Зарубежье"}, "no_category_in_city", 0),
            ({"budget": 1}, "no_matches", 0),
        ]:
            with self.subTest(outcome=outcome, changes=changes):
                status, result = self.post(changes)
                self.assertEqual(status, 200)
                self.assertEqual(result["status"], outcome)
                self.assertEqual(len(result["cards"]), count)
                self.assertTrue(result["message"])
                self.assertNotIn("candidates", result)
                if count == 2:
                    self.assertIn("Меньше трёх", result["message"])
                if not count:
                    self.assertEqual(result["ai"]["mode"], "not_used")

    def test_fallback_reason_and_repeatability_reach_the_browser(self):
        engine = RankingEngine(embedder=UnavailableEmbeddings(), cache_path=None)
        with patch.object(app, "current_engine", return_value=engine):
            status, first = self.post({"preferences": "Спокойная подача"})
            repeated_status, repeated = self.post({"preferences": "Спокойная подача"})
        self.assertEqual((status, repeated_status), (200, 200))
        self.assertEqual(first, repeated)
        self.assertEqual(first["ai"]["mode"], "lexical")
        self.assertIn("AI недоступен", first["ai"]["message"])

    def test_malformed_json_and_preferences_return_readable_errors(self):
        for payload in (b"{", b"[]", b'{"budget":NaN}'):
            with self.subTest(payload=payload):
                status, result = self.post(raw=payload)
                self.assertEqual(status, 400)
                self.assertTrue(result["error"])
        status, result = self.post({"preferences": ["стиль"]})
        self.assertEqual(status, 400)
        self.assertIn("preferences", result["error"])

    def test_static_frontend_files_are_served_from_the_project_directory(self):
        frontend = Path(__file__).resolve().parents[1] / "frontend"
        for route, filename, content_type in (
            ("/", "index.html", "text/html"),
            ("/index.html", "index.html", "text/html"),
            ("/style.css", "style.css", "text/css"),
            ("/app.js", "app.js", "text/javascript"),
        ):
            with self.subTest(route=route), urlopen(self.url + route, timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get_content_type(), content_type)
                self.assertEqual(response.read(), (frontend / filename).read_bytes())

    def test_catalog_metadata_uses_the_relocated_dataset(self):
        with patch.object(app, "configured_api_key", return_value=""):
            with urlopen(self.url + "/api/meta", timeout=3) as response:
                self.assertEqual(response.status, 200)
                metadata = json.load(response)
        self.assertEqual(metadata["profiles"], 100)
        self.assertEqual(set(metadata["cities"]), {"Алматы", "Астана", "Зарубежье"})
        self.assertIn("Банкетный зал", metadata["categories"])
        self.assertEqual(metadata["calendar"], {"start": "2026-09-23", "end": "2026-12-31"})
        self.assertFalse(metadata["ai_available"])


if __name__ == "__main__":
    unittest.main()

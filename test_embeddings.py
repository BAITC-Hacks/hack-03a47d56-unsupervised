"""Offline verification: python -m unittest -v test_embeddings."""

import hashlib
from contextlib import closing
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from threading import BoundedSemaphore, Event, Thread
from time import monotonic, sleep
import unittest
from unittest.mock import MagicMock, patch
from urllib import error

from embeddings import EmbeddingError, OpenAIEmbeddings, _NoRedirects


def api_response(vectors):
    return {"data": [{"index": index, "embedding": vector}
                     for index, vector in enumerate(vectors)]}


class EmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "cache.sqlite3"
        self.transport_patch = patch("embeddings.request.build_opener")
        self.transport = self.transport_patch.start()
        self.addCleanup(self.transport_patch.stop)
        self.opener = self.transport.return_value
        self.client = OpenAIEmbeddings(api_key="test-key-do-not-leak", cache_path=self.cache)

    def respond(self, payload):
        response = MagicMock()
        response.status = 200
        response.read1.side_effect = [json.dumps(payload).encode("utf-8"), b""]
        self.opener.open.return_value.__enter__.return_value = response
        return response

    def test_payload_unique_batch_and_response_reordering(self):
        self.respond({"data": [{"index": 1, "embedding": [0, 2]},
                               {"index": 0, "embedding": [1, 0]}]})
        result = self.client.embed(["ведущий", "музыка", "ведущий"])
        self.assertEqual(result, [[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        self.opener.open.assert_called_once()
        actual = self.opener.open.call_args.args[0]
        self.assertEqual(actual.full_url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(actual.method, "POST")
        self.assertEqual(actual.get_header("Authorization"), "Bearer test-key-do-not-leak")
        self.assertEqual(json.loads(actual.data), {
            "model": "text-embedding-3-small", "input": ["ведущий", "музыка"],
            "encoding_format": "float",
        })
        self.assertGreater(self.opener.open.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(self.opener.open.call_args.kwargs["timeout"], 6.0)
        self.assertIsInstance(self.transport.call_args.args[0], _NoRedirects)
        result[0][0] = 9
        self.assertEqual(result[2], [1.0, 0.0])

    def test_warm_persistent_cache_needs_neither_network_nor_key(self):
        self.respond(api_response([[1, 2], [3, 4]]))
        self.client.embed(["first", "second"])
        self.transport.reset_mock()
        warm = OpenAIEmbeddings(api_key="", cache_path=self.cache)
        self.assertEqual(warm.embed(["second", "first"]), [[3.0, 4.0], [1.0, 2.0]])
        self.transport.assert_not_called()

    def test_partial_cache_only_fetches_missing_text(self):
        self.respond(api_response([[1, 0]]))
        self.client.embed(["old"])
        self.respond(api_response([[0, 1]]))
        self.assertEqual(self.client.embed(["new", "old", "new"]), [[0, 1], [1, 0], [0, 1]])
        self.assertEqual(json.loads(self.opener.open.call_args.args[0].data)["input"], ["new"])

    def test_exact_text_and_model_are_distinct_cache_keys(self):
        self.respond(api_response([[1, 0]]))
        self.client.embed(["text"])
        other_model = OpenAIEmbeddings(api_key="test-key", model="different-model", cache_path=self.cache)
        self.respond(api_response([[0, 1]]))
        self.assertEqual(other_model.embed(["text"]), [[0, 1]])
        self.assertEqual(json.loads(self.opener.open.call_args.args[0].data)["model"], "different-model")
        self.respond(api_response([[1, 1]]))
        self.assertEqual(self.client.embed(["text "]), [[1, 1]])
        self.assertEqual(self.opener.open.call_count, 3)

    def test_cache_contains_hash_and_vector_only(self):
        self.respond(api_response([[1, 2]]))
        self.client.embed(["private original preference"])
        with closing(sqlite3.connect(self.cache)) as connection:
            row = connection.execute("SELECT * FROM embeddings").fetchone()
        self.assertEqual(row[0], "text-embedding-3-small")
        self.assertEqual(row[1], hashlib.sha256(b"private original preference").hexdigest())
        self.assertNotIn(b"private original preference", self.cache.read_bytes())
        self.assertNotIn(b"test-key-do-not-leak", self.cache.read_bytes())

    def test_invalid_vectors_and_indices_fail_without_caching(self):
        invalid = [
            {"data": []}, {"data": [None]}, {"data": {}}, [],
            {"data": [{"index": True, "embedding": [1]}]},
            {"data": [{"index": -1, "embedding": [1]}]},
            {"data": [{"index": 1, "embedding": [1]}]},
            {"data": [{"index": 0, "embedding": [True]}]},
            {"data": [{"index": 0, "embedding": ["1"]}]},
            {"data": [{"index": 0, "embedding": []}]},
            {"data": [{"index": 0, "embedding": [0, 0]}]},
            {"data": [{"index": 0, "embedding": [float("nan")]}]},
            {"data": [{"index": 0, "embedding": [float("inf")]}]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.respond(payload)
                with self.assertRaises(EmbeddingError):
                    self.client.embed(["invalid"])
        with closing(sqlite3.connect(self.cache)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM embeddings").fetchone()[0], 0)

    def test_duplicate_indices_and_mixed_dimensions_fail(self):
        for payload in [
            {"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]},
            api_response([[1, 2], [1]]),
        ]:
            with self.subTest(payload=payload):
                self.respond(payload)
                with self.assertRaises(EmbeddingError):
                    self.client.embed(["one", "two"])

    def test_malformed_json_is_a_safe_failure(self):
        response = self.respond({})
        response.read1.side_effect = [b"not json test-key-do-not-leak", b""]
        with self.assertRaises(EmbeddingError) as caught:
            self.client.embed(["text"])
        self.assertNotIn("test-key-do-not-leak", str(caught.exception))

    def test_timeout_and_http_errors_are_sanitized_without_retry(self):
        failures = [TimeoutError("test-key-do-not-leak"),
                    error.URLError("test-key-do-not-leak"),
                    IncompleteRead(b"test-key-do-not-leak"),
                    error.HTTPError("https://api.openai.com/v1/embeddings", 401,
                                    "test-key-do-not-leak", {}, io.BytesIO(b"secret body")),
                    error.HTTPError("https://api.openai.com/v1/embeddings", 429,
                                    "test-key-do-not-leak", {}, io.BytesIO(b"secret body"))]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.opener.open.reset_mock()
                self.opener.open.side_effect = failure
                with self.assertRaises(EmbeddingError) as caught:
                    self.client.embed(["text"])
                self.assertNotIn("test-key-do-not-leak", str(caught.exception))
                self.assertNotIn("secret body", str(caught.exception))
                self.opener.open.assert_called_once()

    def test_redirects_are_refused(self):
        handler = _NoRedirects()
        self.assertIsNone(handler.redirect_request(None, None, 307, "redirect", {}, "https://other.example/"))
        self.opener.open.side_effect = error.HTTPError(
            "https://api.openai.com/v1/embeddings", 307, "redirect", {}, io.BytesIO(b"secret body"))
        with self.assertRaises(EmbeddingError):
            self.client.embed(["text"])
        self.opener.open.assert_called_once()

    def test_unusable_cache_does_not_block_valid_api_result(self):
        self.cache.write_text("this is not sqlite", encoding="utf-8")
        self.respond(api_response([[1, 2]]))
        self.assertEqual(self.client.embed(["text"]), [[1.0, 2.0]])

    def test_corrupt_entry_is_refetched(self):
        self.respond(api_response([[1, 2]]))
        self.client.embed(["text"])
        with closing(sqlite3.connect(self.cache)) as connection:
            connection.execute("UPDATE embeddings SET vector = '[0, 0]'")
            connection.commit()
        self.respond(api_response([[3, 4]]))
        self.assertEqual(self.client.embed(["text"]), [[3, 4]])
        self.assertEqual(self.opener.open.call_count, 2)

    def test_cached_and_new_dimensions_must_match(self):
        self.respond(api_response([[1, 2]]))
        self.client.embed(["old"])
        self.respond(api_response([[1, 2, 3]]))
        with self.assertRaises(EmbeddingError):
            self.client.embed(["old", "new"])

    def test_empty_and_invalid_inputs_never_use_network(self):
        self.assertEqual(self.client.embed([]), [])
        for texts in ["text", [""], ["  "], [None], [3], ["\ud800"]]:
            with self.subTest(texts=repr(texts)):
                with self.assertRaises(EmbeddingError):
                    self.client.embed(texts)
        self.transport.assert_not_called()

    def test_missing_key_and_environment_configuration(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "OPENAI_EMBEDDING_MODEL": "test-model"}):
            client = OpenAIEmbeddings(cache_path=self.cache)
            self.assertEqual(client.model, "test-model")
            with self.assertRaises(EmbeddingError):
                client.embed(["text"])
        self.transport.assert_not_called()

    def test_timed_out_operation_has_no_late_cache_write_or_unbounded_workers(self):
        slots, release, started = BoundedSemaphore(1), Event(), Event()
        client = OpenAIEmbeddings(api_key="test-key", cache_path=self.cache, timeout=0.08)

        def stuck_request(texts, deadline):
            started.set()
            release.wait(2)
            return [[1.0, 0.0]]

        with patch("embeddings._NETWORK_SLOTS", slots), patch.object(
                client, "_fetch_before_deadline", side_effect=stuck_request) as fetch:
            try:
                beginning = monotonic()
                with self.assertRaises(EmbeddingError):
                    client.embed(["first"])
                self.assertLess(monotonic() - beginning, 0.5)
                self.assertTrue(started.is_set())
                # A timed-out OS/network operation still owns its slot. More
                # traffic must fail immediately without queued work or threads.
                beginning = monotonic()
                for index in range(20):
                    with self.assertRaises(EmbeddingError):
                        client._fetch([f"overflow {index}"])
                self.assertLess(monotonic() - beginning, 0.3)
                self.assertEqual(fetch.call_count, 1)
            finally:
                release.set()
                self.assertTrue(slots.acquire(timeout=2))
                slots.release()
        with closing(sqlite3.connect(self.cache)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM embeddings").fetchone()[0], 0)

    def test_response_size_limit_also_applies_to_streamed_chunks(self):
        response = self.respond({})
        response.read1.side_effect = [b"123", b"456"]
        with patch("embeddings._MAX_RESPONSE_BYTES", 5), self.assertRaises(EmbeddingError):
            self.client.embed(["text"])
        self.assertEqual(response.read1.call_count, 2)


class NetworkDeadlineTests(unittest.TestCase):
    def test_header_and_slow_body_share_one_deadline(self):
        closed = Event()

        class SlowResponse(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(api_response([[1, 0]])).encode()
                try:
                    sleep(0.15)
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.flush()
                    # Every individual operation takes less than timeout=0.25,
                    # but headers + body exceed the total allowed time.
                    sleep(0.15)
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    closed.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowResponse)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                    "embeddings.EMBEDDINGS_URL", f"http://127.0.0.1:{server.server_port}/embeddings"):
                client = OpenAIEmbeddings(api_key="test-key", timeout=0.25,
                                          cache_path=Path(directory) / "cache.sqlite3")
                beginning = monotonic()
                with self.assertRaises(EmbeddingError):
                    client.embed(["deadline probe"])
                self.assertLess(monotonic() - beginning, 0.45)
                self.assertTrue(closed.wait(2))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

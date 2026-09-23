"""OpenAI embeddings with a best-effort, persistent local cache.

Only the official embeddings endpoint is used. Cache entries are isolated by model
and the SHA-256 hash of the exact input text. A warm cache works without an API key;
an unavailable cache does not prevent a successful API request.
"""

from __future__ import annotations

import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import sqlite3
from urllib import error, request


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "embeddings.sqlite3"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class EmbeddingError(Exception):
    """A safe, user-displayable embedding failure without credentials or bodies."""


class _NoRedirects(request.HTTPRedirectHandler):
    """Never forward the Authorization header to a redirected destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _vector(value):
    if not isinstance(value, list) or not value:
        raise EmbeddingError("OpenAI вернул некорректный вектор.")
    result = []
    for number in value:
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise EmbeddingError("OpenAI вернул некорректный вектор.")
        try:
            number = float(number)
        except (ValueError, OverflowError):
            raise EmbeddingError("OpenAI вернул некорректный вектор.") from None
        if not math.isfinite(number):
            raise EmbeddingError("OpenAI вернул некорректный вектор.")
        result.append(number)
    if not any(result):
        raise EmbeddingError("OpenAI вернул нулевой вектор.")
    return result


class OpenAIEmbeddings:
    """Embed a batch with one API call for unique texts absent from the cache.

    API configuration comes from OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL unless
    explicitly supplied. No retries are performed. ``timeout`` must be finite and
    between 0 and 60 seconds. Cache reads/writes are best-effort and contain hashes
    and vectors only, never the API key or the original text.
    """

    def __init__(self, api_key=None, model=None, cache_path=None, timeout=6.0):
        self.model = model if model is not None else (os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_MODEL)
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingError("Не задана модель эмбеддингов OpenAI.")
        self.model = self.model.strip()
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        if not isinstance(self._api_key, str):
            raise EmbeddingError("Некорректная настройка OPENAI_API_KEY.")
        self._api_key = self._api_key.strip()
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= 60):
            raise EmbeddingError("Таймаут OpenAI должен быть больше 0 и не больше 60 секунд.")
        self.timeout = float(timeout)
        try:
            self.cache_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
        except (TypeError, ValueError):
            raise EmbeddingError("Некорректный путь кеша эмбеддингов.") from None

    def _connect_cache(self):
        connection = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.cache_path), timeout=0.1)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(model TEXT NOT NULL, text_hash TEXT NOT NULL, vector TEXT NOT NULL, "
                "PRIMARY KEY (model, text_hash))"
            )
            return connection
        except (OSError, ValueError, sqlite3.Error):
            if connection is not None:
                connection.close()
            return None

    def _read_cache(self, hashes):
        cached = {}
        connection = self._connect_cache()
        if connection is None:
            return cached
        try:
            for text_hash in hashes:
                row = connection.execute(
                    "SELECT vector FROM embeddings WHERE model = ? AND text_hash = ?",
                    (self.model, text_hash),
                ).fetchone()
                if row is not None:
                    try:
                        cached[text_hash] = _vector(json.loads(row[0]))
                    except (EmbeddingError, ValueError, TypeError, RecursionError):
                        pass  # A corrupt entry is a cache miss.
        except sqlite3.Error:
            return {}
        finally:
            connection.close()
        if len({len(vector) for vector in cached.values()}) > 1:
            return {}  # Refetch inconsistent entries as one coherent batch.
        return cached

    def _write_cache(self, vectors):
        connection = self._connect_cache()
        if connection is None:
            return
        try:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, vector) VALUES (?, ?, ?)",
                [(self.model, text_hash, json.dumps(vector, allow_nan=False))
                 for text_hash, vector in vectors.items()],
            )
            connection.commit()
        except (OSError, ValueError, sqlite3.Error):
            pass  # Ranking can still use the valid in-memory API response.
        finally:
            connection.close()

    def _fetch(self, texts):
        if not self._api_key:
            raise EmbeddingError("Для AI-подбора задайте OPENAI_API_KEY.")
        payload = json.dumps(
            {"model": self.model, "input": texts, "encoding_format": "float"},
            ensure_ascii=False,
        ).encode("utf-8")
        api_request = request.Request(
            EMBEDDINGS_URL,
            data=payload,
            headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            opener = request.build_opener(_NoRedirects())
            with opener.open(api_request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise EmbeddingError("Сервис эмбеддингов OpenAI временно недоступен.")
                body = response.read(_MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            exc.close()
            if exc.code in (401, 403):
                raise EmbeddingError("OpenAI отклонил API-ключ или доступ к модели.") from None
            if exc.code == 429:
                raise EmbeddingError("Достигнут лимит запросов или квоты OpenAI.") from None
            raise EmbeddingError("Сервис эмбеддингов OpenAI временно недоступен.") from None
        except (error.URLError, HTTPException, OSError, ValueError):
            raise EmbeddingError("Не удалось связаться с OpenAI за отведённое время.") from None
        if len(body) > _MAX_RESPONSE_BYTES:
            raise EmbeddingError("Ответ OpenAI превышает допустимый размер.")
        try:
            result = json.loads(body)
        except (ValueError, UnicodeError, RecursionError):
            raise EmbeddingError("OpenAI вернул некорректный JSON.") from None
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise EmbeddingError("OpenAI вернул некорректный ответ эмбеддингов.")
        if len(result["data"]) != len(texts):
            raise EmbeddingError("OpenAI вернул неполный набор эмбеддингов.")
        vectors = {}
        for item in result["data"]:
            if not isinstance(item, dict):
                raise EmbeddingError("OpenAI вернул некорректный ответ эмбеддингов.")
            index = item.get("index")
            if (isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < len(texts) or index in vectors):
                raise EmbeddingError("OpenAI вернул некорректные индексы эмбеддингов.")
            vectors[index] = _vector(item.get("embedding"))
        if len({len(vector) for vector in vectors.values()}) != 1:
            raise EmbeddingError("Размеры эмбеддингов OpenAI не совпадают.")
        return [vectors[index] for index in range(len(texts))]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return valid vectors in input order, including repeated input texts."""
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Эмбеддинги принимают список непустых строк.")
        if not texts:
            return []
        try:
            hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        except UnicodeError:
            raise EmbeddingError("Текст содержит некорректные символы Unicode.") from None
        unique_texts = dict(zip(hashes, texts))
        cached = self._read_cache(unique_texts)
        missing = {text_hash: text for text_hash, text in unique_texts.items() if text_hash not in cached}
        if missing:
            fetched = self._fetch(list(missing.values()))
            if cached and len(next(iter(cached.values()))) != len(fetched[0]):
                raise EmbeddingError("Размеры кешированных и новых эмбеддингов не совпадают.")
            fresh = dict(zip(missing, fetched))
            self._write_cache(fresh)
            cached.update(fresh)
        return [list(cached[text_hash]) for text_hash in hashes]

"""Deterministic ranking of the complete eligible contractor pool."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any, Iterable, Mapping

from .data_loader import normalize
from .embeddings import EmbeddingError, OpenAIEmbeddings
from .filtering import filter_contractors, validate_request
from .paths import CACHE_DIR

DEFAULT_RANK_CACHE = CACHE_DIR / "ranking.sqlite3"
RANKING_VERSION = "2"
STOP_WORDS = set("для это как что или при без под над все ваш вас вам нас наш мне мой мои мероприятия мероприятие нужен нужна нужно хочу".split())


def text_tokens(text: str) -> list[str]:
    """Small, disclosed lexical fallback; this is not an AI embedding."""
    return [word[:5] for word in re.findall(r"[^\W\d_]+", normalize(text))
            if len(word) > 2 and word not in STOP_WORDS]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise EmbeddingError("Некорректная размерность эмбеддингов.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for vector in (left, right) for x in vector):
        raise EmbeddingError("Некорректные значения эмбеддингов.")
    a, b = math.hypot(*left), math.hypot(*right)
    if not a or not b or not math.isfinite(a) or not math.isfinite(b):
        raise EmbeddingError("Получен нулевой эмбеддинг.")
    return max(-1.0, min(1.0, math.fsum((x / a) * (y / b) for x, y in zip(left, right))))


def lexical_similarity(query: str, description: str) -> float:
    a, b = Counter(text_tokens(query)), Counter(text_tokens(description))
    denominator = math.sqrt(sum(x * x for x in a.values()) * sum(x * x for x in b.values()))
    return sum(value * b[word] for word, value in a.items()) / denominator if denominator else 0.0


def normalize_preferences(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("preferences: ожидается строка")
    value = " ".join(value.split())
    if len(value) > 2000:
        raise ValueError("preferences: не более 2000 символов")
    return value


def build_query(request: Mapping[str, Any], preferences: str = "") -> str:
    # City, price, date, language and hours have already been strictly checked.
    return f"Категория: {request['category']}. Формат: {request['event_type']}." + (
        f" Пожелания: {preferences}" if preferences else ""
    )


class RankingEngine:
    """Reuse one engine in the app; disk cache also survives process restarts.

    auto: OpenAI embeddings, falling back to lexical similarity on API failure.
    offline: lexical similarity only. A saved query keeps its first successful
    ranking mode, including fallback, until the cache/version is changed.
    """

    def __init__(self, *, mode: str = "auto", embedder: Any = None,
                 cache_path: str | Path | None = DEFAULT_RANK_CACHE):
        if mode not in ("auto", "offline"):
            raise ValueError("mode: допустимы auto и offline")
        self.mode = mode
        self.embedder = embedder if embedder is not None else OpenAIEmbeddings()
        self.cache_path = Path(cache_path) if cache_path is not None else None
        # Adding an API key intentionally enables AI for prior no-key queries.
        self.configuration = "custom" if embedder is not None else ("key" if os.getenv("OPENAI_API_KEY") else "no-key")
        self._memory: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self._query_locks: dict[str, Any] = {}

    def _connection(self) -> sqlite3.Connection:
        assert self.cache_path is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.cache_path, timeout=0.25)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS rankings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        except sqlite3.Error:
            db.close()
            raise
        return db

    @staticmethod
    def _valid_snapshot(value: Any, ids: set[str]) -> bool:
        return (isinstance(value, dict) and value.get("mode") in ("openai", "lexical")
                and isinstance(value.get("reason"), str)
                and isinstance(value.get("similarities"), dict)
                and set(value["similarities"]) == ids
                and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
                        for x in value["similarities"].values()))

    def _read(self, key: str, ids: set[str]) -> dict[str, Any] | None:
        if key in self._memory:
            return self._memory[key]
        if self.cache_path is not None:
            try:
                with closing(self._connection()) as db, db:
                    row = db.execute("SELECT value FROM rankings WHERE key = ?", (key,)).fetchone()
                    value = json.loads(row[0]) if row else None
                    if self._valid_snapshot(value, ids):
                        self._memory[key] = value
                        return value
                    if row:
                        db.execute("DELETE FROM rankings WHERE key = ?", (key,))
            except (OSError, sqlite3.Error, ValueError, TypeError):
                pass  # Read-only/unavailable cache must not break a recommendation.
        return None

    def _save(self, key: str, value: dict[str, Any], ids: set[str]) -> dict[str, Any]:
        if self.cache_path is not None:
            try:
                with closing(self._connection()) as db, db:
                    db.execute("INSERT OR IGNORE INTO rankings VALUES (?, ?)",
                               (key, json.dumps(value, ensure_ascii=False, allow_nan=False)))
                    # Concurrent processes agree on whichever result was saved first.
                    row = db.execute("SELECT value FROM rankings WHERE key = ?", (key,)).fetchone()
                    stored = json.loads(row[0])
                    if self._valid_snapshot(stored, ids):
                        value = stored
            except (OSError, sqlite3.Error, ValueError, TypeError):
                pass
        self._memory[key] = value
        return value

    def rank(self, candidates: Iterable[Mapping[str, Any]], request: Mapping[str, Any],
             *, preferences: str | None = None) -> dict[str, Any]:
        """Return all ranked candidates and AI metadata; no top-3 truncation here."""
        query = validate_request(request)
        wishes = normalize_preferences(request.get("preferences") if preferences is None else preferences)
        pool = sorted((deepcopy(dict(c)) for c in candidates), key=lambda c: c["id"])
        # Reuse the strict filter to validate the ranking boundary.
        verified = filter_contractors(pool, query)
        if len(verified["candidates"]) != len(pool):
            raise ValueError("Ранжированию нужно передать только прошедших фильтр candidates.")
        if not pool:
            return {"ranked": [], "ai": {"mode": "not_used", "model": None,
                    "message": "Нет подходящих кандидатов для ранжирования."}}
        model = str(self.embedder.model)
        query_text = build_query(query, wishes)
        key_input = {"version": RANKING_VERSION, "mode": self.mode, "model": model,
                     "configuration": self.configuration, "request": query,
                     "preferences": wishes, "candidates": pool}
        cache_key = hashlib.sha256(json.dumps(key_input, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False).encode("utf-8")).hexdigest()
        ids = {c["id"] for c in pool}
        # Only identical queries share a network wait. Independent requests must
        # not queue behind a slow API call; the cache still fixes one result per key.
        with self._lock:
            query_lock = self._query_locks.setdefault(cache_key, RLock())
        with query_lock:
            snapshot = self._read(cache_key, ids)
            if snapshot is None:
                descriptions = [c.get("description", "") for c in pool]
                used_mode, reason = "lexical", "Выбран офлайн-режим: совпадения по словам."
                similarities = None
                nonempty = [(i, text) for i, text in enumerate(descriptions) if text.strip()]
                if not nonempty:
                    similarities = [0.0] * len(pool)
                    reason = "Описания отсутствуют: смысловая оценка равна нулю, порядок определяется запасом бюджета."
                elif self.mode == "auto":
                    try:
                        # Empty source descriptions receive zero, not invented text.
                        vectors = self.embedder.embed([query_text] + [text for _, text in nonempty])
                        if len(vectors) != len(nonempty) + 1:
                            raise EmbeddingError("Неполный ответ сервиса эмбеддингов.")
                        similarities = [0.0] * len(pool)
                        for (index, _), vector in zip(nonempty, vectors[1:]):
                            similarities[index] = max(0.0, cosine_similarity(vectors[0], vector))
                        used_mode, reason = "openai", "Описания сравниваются с запросом через эмбеддинги OpenAI."
                    except EmbeddingError:
                        similarities = None  # Discard any scores computed before the failure.
                        reason = "AI недоступен: использованы совпадения по словам; результат сохранён для повторного запроса."
                if similarities is None:
                    similarities = [lexical_similarity(query_text, text) for text in descriptions]
                snapshot = self._save(cache_key, {"mode": used_mode, "reason": reason,
                    "similarities": {c["id"]: round(s, 10) for c, s in zip(pool, similarities)}}, ids)

        ranked = []
        for candidate in pool:
            semantic = snapshot["similarities"][candidate["id"]]
            headroom = (1 - candidate["price_from_kzt"] / query["budget"]) if query["budget"] else 1.0
            semantic_points, budget_points = round(70 * semantic, 6), round(30 * headroom, 6)
            details = {"semantic_similarity": semantic, "semantic_points": semantic_points,
                       "budget_headroom": round(headroom, 8), "budget_points": budget_points,
                       "weights": {"semantic": 70, "budget": 30},
                       "preferences": wishes, "mode": snapshot["mode"]}
            ranked.append({"contractor": candidate, "score": round(semantic_points + budget_points, 6),
                           "score_details": details})
        ranked.sort(key=lambda item: (-item["score"], item["contractor"]["id"]))
        return {"ranked": ranked, "ai": {"mode": snapshot["mode"],
                "model": model if snapshot["mode"] == "openai" else None, "message": snapshot["reason"]}}


def rank_contractors(candidates: Iterable[Mapping[str, Any]], request: Mapping[str, Any], *,
                     preferences: str | None = None, engine: RankingEngine | None = None) -> list[dict[str, Any]]:
    """Compatibility entry point; use engine.rank when UI also needs AI status."""
    return (engine or RankingEngine()).rank(candidates, request, preferences=preferences)["ranked"]

"""Ranking and UI integration regressions; no API key or network required."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from threading import Barrier, Event
import unittest

from app.data_loader import load_contractors
from app.embeddings import EmbeddingError
from app.filtering import filter_contractors
from app.recommendations import recommend_from_filtered
from app.scorer import RankingEngine, rank_contractors


ROOT = Path(__file__).resolve().parents[1]


def query(**changes):
    result = dict(city="Алматы", date="2026-09-23", event_type="свадьба",
                  category="Ведущий", budget=2_000_000)
    result.update(changes)
    return result


def profile(**changes):
    result = dict(id="A", anon_name="Тест", categories=["Ведущий"], city="Алматы",
                  price_from_kzt=100_000, event_formats=["свадьба"], languages=["русский"],
                  max_hours=6, busy_dates=[], description="Современные интерактивы для гостей.",
                  synthetic=False, city_imputed=False, price_imputed=False)
    result.update(changes)
    return result


class FakeEmbeddings:
    model = "fake-embedding-v1"

    def __init__(self, matching=(), *, failed=False):
        self.matching = set(matching)
        self.failed = failed
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.failed:
            raise EmbeddingError("Сервис временно недоступен")
        return [[1.0, 0.0]] + [[1.0, 0.0] if text in self.matching else [0.0, 1.0]
                               for text in texts[1:]]


class ForbiddenEmbeddings:
    model = "fake-embedding-v1"

    def embed(self, texts):
        raise AssertionError("Этот сценарий не должен вызывать AI")


def offline_engine():
    return RankingEngine(mode="offline", embedder=ForbiddenEmbeddings(), cache_path=None)


class RankingTests(unittest.TestCase):
    def test_independent_queries_do_not_share_a_network_wait(self):
        barrier = Barrier(2)

        class ConcurrentEmbeddings(FakeEmbeddings):
            def embed(self, texts):
                # A global engine lock would leave the first call alone here
                # until this timeout, instead of admitting the second request.
                barrier.wait(timeout=2)
                return super().embed(texts)

        embedder = ConcurrentEmbeddings()
        engine = RankingEngine(embedder=embedder, cache_path=None)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(engine.rank, [profile()], query(), preferences="вокал")
            second = executor.submit(engine.rank, [profile()], query(), preferences="юмор")
            results = [first.result(timeout=3), second.result(timeout=3)]
        self.assertEqual(len(embedder.calls), 2)
        self.assertTrue(all(result["ai"]["mode"] == "openai" for result in results))

    def test_concurrent_identical_queries_compute_once_and_keep_fallback(self):
        entered, release, second_started = Event(), Event(), Event()

        class BlockingFailure(FakeEmbeddings):
            def embed(self, texts):
                self.calls.append(list(texts))
                entered.set()
                release.wait(2)
                raise EmbeddingError("Тестовый сбой")

        embedder = BlockingFailure()
        engine = RankingEngine(embedder=embedder, cache_path=None)

        def repeat():
            second_started.set()
            return engine.rank([profile()], query())

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(engine.rank, [profile()], query())
            try:
                self.assertTrue(entered.wait(2))
                second = executor.submit(repeat)
                self.assertTrue(second_started.wait(2))
                with self.assertRaises(FutureTimeoutError):
                    second.result(timeout=0.05)
            finally:
                release.set()
            results = [first.result(timeout=3), second.result(timeout=3)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["ai"]["mode"], "lexical")
        self.assertEqual(len(embedder.calls), 1)

    def test_concurrent_engines_keep_first_committed_cache_result(self):
        barrier = Barrier(2)

        class RacingEmbeddings(FakeEmbeddings):
            def embed(self, texts):
                barrier.wait(timeout=2)
                return super().embed(texts)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "rankings.sqlite3"
            engines = [RankingEngine(embedder=RacingEmbeddings(failed=failed), cache_path=cache)
                       for failed in (False, True)]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(engine.rank, [profile()], query()) for engine in engines]
                results = [future.result(timeout=3) for future in futures]
            self.assertEqual(results[0], results[1])
            restarted = RankingEngine(embedder=ForbiddenEmbeddings(), cache_path=cache)
            self.assertEqual(restarted.rank([profile()], query()), results[0])

    def test_semantic_match_beats_cheaper_unrelated_description(self):
        matching = "Живой вокал и джазовые композиции."
        candidates = [profile(id="cheap", price_from_kzt=10_000, description="Фотобудка."),
                      profile(id="relevant", price_from_kzt=200_000, description=matching)]
        embedder = FakeEmbeddings([matching])
        result = RankingEngine(embedder=embedder, cache_path=None).rank(
            candidates, query(budget=200_000), preferences="Джазовый вокал")
        self.assertEqual([item["contractor"]["id"] for item in result["ranked"]],
                         ["relevant", "cheap"])
        self.assertEqual(result["ai"]["mode"], "openai")
        self.assertIn("Джазовый вокал", embedder.calls[0][0])

    def test_equal_scores_use_id_and_order_is_shuffle_invariant(self):
        candidates = [profile(id=key) for key in ("C", "A", "B")]
        expected = rank_contractors(candidates, query(), engine=offline_engine())
        self.assertEqual([item["contractor"]["id"] for item in expected], ["A", "B", "C"])
        self.assertEqual(len({item["score"] for item in expected}), 1)
        random.Random(42).shuffle(candidates)
        self.assertEqual(rank_contractors(candidates, query(), engine=offline_engine()), expected)

    def test_offline_never_calls_embedder(self):
        result = offline_engine().rank([profile()], query())
        self.assertEqual(result["ai"]["mode"], "lexical")
        self.assertIsNone(result["ai"]["model"])

    def test_empty_descriptions_have_no_semantic_match_and_do_not_call_api(self):
        candidates = [profile(id="A", description=""), profile(id="B", description="  \n  ")]
        result = RankingEngine(embedder=ForbiddenEmbeddings(), cache_path=None).rank(candidates, query())
        self.assertEqual(result["ai"]["mode"], "lexical")
        self.assertIsNone(result["ai"]["model"])
        self.assertTrue(result["ai"]["message"])
        self.assertEqual([item["score_details"]["semantic_similarity"] for item in result["ranked"]],
                         [0, 0])

    def test_fallback_stays_identical_after_recovery_and_process_restart(self):
        candidates = filter_contractors(load_contractors(), query())["candidates"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.sqlite3"
            embedder = FakeEmbeddings(failed=True)
            engine = RankingEngine(embedder=embedder, cache_path=path)
            first = engine.rank(candidates, query(), preferences="Живой вокал")
            self.assertEqual(first["ai"]["mode"], "lexical")
            embedder.failed = False
            self.assertEqual(engine.rank(candidates, query(), preferences="Живой вокал"), first)
            self.assertEqual(len(embedder.calls), 1)
            restarted = RankingEngine(embedder=ForbiddenEmbeddings(), cache_path=path)
            self.assertEqual(restarted.rank(candidates, query(), preferences="Живой вокал"), first)
            script = """
import json, sys
from app.data_loader import load_contractors
from app.filtering import filter_contractors
from app.scorer import RankingEngine
from tests.test_ranking import ForbiddenEmbeddings, query
pool = filter_contractors(load_contractors(), query())["candidates"]
result = RankingEngine(embedder=ForbiddenEmbeddings(), cache_path=sys.argv[1]).rank(
    pool, query(), preferences="Живой вокал")
print(json.dumps(result, ensure_ascii=False))
"""
            completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(path)],
                                       cwd=ROOT, capture_output=True, encoding="utf-8", timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), first)

    def test_invalid_later_vector_falls_back_for_entire_pool_and_persists(self):
        class InvalidLaterEmbeddings(FakeEmbeddings):
            def embed(self, texts):
                vectors = super().embed(texts)
                vectors[2] = [1.0]  # Fail after the first similarity was computed.
                return vectors

        candidates = [profile(id="A", description="Фотобудка."),
                      profile(id="B", description="Ведущий на свадьбу."),
                      profile(id="C", description=""),
                      profile(id="D", description="Джазовый вокал.")]
        preferences = "Джазовый вокал"
        expected = offline_engine().rank(candidates, query(), preferences=preferences)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.sqlite3"
            embedder = InvalidLaterEmbeddings([candidates[0]["description"]])
            engine = RankingEngine(embedder=embedder, cache_path=path)
            result = engine.rank(candidates, query(), preferences=preferences)
            self.assertEqual(result["ai"]["mode"], "lexical")
            self.assertIsNone(result["ai"]["model"])
            self.assertEqual(result["ranked"], expected["ranked"])
            self.assertEqual(engine.rank(candidates, query(), preferences=preferences), result)
            self.assertEqual(len(embedder.calls), 1)
            restarted = RankingEngine(embedder=ForbiddenEmbeddings(), cache_path=path)
            self.assertEqual(restarted.rank(candidates, query(), preferences=preferences), result)

    def test_description_and_preferences_changes_invalidate_cached_ranking(self):
        embedder = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            engine = RankingEngine(embedder=embedder, cache_path=Path(directory) / "rankings.db")
            candidate = profile()
            engine.rank([candidate], query(), preferences="интерактивы")
            engine.rank([candidate], query(), preferences="интерактивы")
            self.assertEqual(len(embedder.calls), 1)
            candidate["description"] = "Живой вокал и джаз."
            engine.rank([candidate], query(), preferences="интерактивы")
            self.assertEqual(len(embedder.calls), 2)
            engine.rank([candidate], query(), preferences="джаз")
            self.assertEqual(len(embedder.calls), 3)

    def test_invalid_eligible_pool_is_rejected_before_ai(self):
        changes = [dict(price_from_kzt=2_000_001), dict(busy_dates=["2026-09-23"]),
                   dict(city="Астана"), dict(categories=["Фотограф"]),
                   dict(event_formats=["конференция"]), dict(languages=["казахский"]),
                   dict(max_hours=1)]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                RankingEngine(embedder=ForbiddenEmbeddings(), cache_path=None).rank(
                    [profile(**changed)], query(language="русский", duration_hours=6))

    def test_ranker_does_not_mutate_inputs_or_reuse_returned_mutable_objects(self):
        candidates = [profile()]
        request = query(preferences="Интерактивы")
        before_candidates, before_request = deepcopy(candidates), deepcopy(request)
        engine = offline_engine()
        first = engine.rank(candidates, request)
        expected = deepcopy(first)
        first["ranked"][0]["contractor"]["description"] = "Изменено интерфейсом"
        first["ranked"][0]["score_details"]["weights"]["semantic"] = 0
        self.assertEqual(candidates, before_candidates)
        self.assertEqual(request, before_request)
        self.assertEqual(engine.rank(candidates, request), expected)

    def test_zero_budget_and_exact_budget_are_finite_and_supported(self):
        for price in (0, 100_000):
            with self.subTest(price=price):
                result = offline_engine().rank([profile(price_from_kzt=price)], query(budget=price))
                json.dumps(result, allow_nan=False)
                self.assertEqual(len(result["ranked"]), 1)
                self.assertGreaterEqual(result["ranked"][0]["score"], 0)
                self.assertLessEqual(result["ranked"][0]["score"], 100)


class RecommendationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_contractors()

    def test_top_three_selected_after_ranking_all_four_real_candidates(self):
        filtered = filter_contractors(self.catalog, query())
        self.assertEqual(len(filtered["candidates"]), 4)
        last = filtered["candidates"][-1]
        embedder = FakeEmbeddings([last["description"]])
        result = recommend_from_filtered(filtered, preferences="Творческий ведущий",
                                        engine=RankingEngine(embedder=embedder, cache_path=None))
        self.assertEqual(len(result["cards"]), 3)
        self.assertEqual(result["cards"][0]["id"], last["id"])
        self.assertEqual(len(embedder.calls[0]), 5)
        self.assertEqual(result["counts"]["eligible"], 4)
        self.assertEqual(result["candidates"], filtered["candidates"])

    def test_real_date_change_recomputes_eligible_pool(self):
        results = [recommend_from_filtered(filter_contractors(self.catalog, query(date=day)),
                                          engine=offline_engine())
                   for day in ("2026-10-01", "2026-10-02")]
        self.assertEqual({c["id"] for c in results[0]["cards"]}, {"HK-42352", "HK-44923"})
        self.assertEqual({c["id"] for c in results[1]["cards"]}, {"HK-35215"})
        for result in results:
            cards = {card["id"] for card in result["cards"]}
            self.assertTrue(cards.isdisjoint({item["id"] for item in result["excluded"]}))

    def test_rare_real_category_preserves_shortage_and_source_flags(self):
        # The catalog contains just one florist in Astana.
        florist = next(c for c in self.catalog if c["city"] == "Астана" and c["categories"] == ["Флорист"])
        day = next(f"2026-10-{number:02d}" for number in range(1, 32)
                   if f"2026-10-{number:02d}" not in florist["busy_dates"])
        filtered = filter_contractors(self.catalog, query(city="Астана", category="Флорист", date=day))
        result = recommend_from_filtered(filtered, engine=offline_engine())
        self.assertEqual(len(result["cards"]), 1)
        self.assertIn("Меньше трёх", result["message"])
        card = result["cards"][0]
        self.assertTrue(card["synthetic"])
        for flag in ("synthetic", "city_imputed", "price_imputed"):
            self.assertEqual(card[flag], florist[flag])
        self.assertEqual(card["warnings"], filtered["warnings"][florist["id"]])
        for fact in filtered["evidence"][florist["id"]]:
            self.assertIn(fact, card["evidence"])
        description_facts = [fact for fact in card["evidence"] if fact["field"] == "description"]
        self.assertEqual(len(description_facts), 1)
        self.assertIn(description_facts[0]["value"], florist["description"])

    def test_no_category_and_no_matches_keep_distinct_diagnostics_without_ai(self):
        for request, status in ((query(category="Несуществующая категория"), "no_category_in_city"),
                                (query(budget=0), "no_matches")):
            with self.subTest(status=status):
                filtered = filter_contractors(self.catalog, request)
                result = recommend_from_filtered(filtered, engine=RankingEngine(
                    embedder=ForbiddenEmbeddings(), cache_path=None))
                self.assertEqual(result["status"], status)
                self.assertEqual(result["cards"], [])
                self.assertEqual(result["ai"]["mode"], "not_used")
                for field in ("message", "excluded", "reason_counts", "counts"):
                    self.assertEqual(result[field], filtered[field])

    def test_explicit_preferences_survive_filter_contract_and_inputs_stay_unchanged(self):
        filtered = filter_contractors(self.catalog, query(preferences="Пожелание из формы"))
        self.assertNotIn("preferences", filtered["request"])
        before = deepcopy(filtered)
        embedder = FakeEmbeddings()
        result = recommend_from_filtered(filtered, preferences="  Живой\nвокал  ",
                                        engine=RankingEngine(embedder=embedder, cache_path=None))
        self.assertEqual(result["request"]["preferences"], "Живой вокал")
        self.assertIn("Живой вокал", embedder.calls[0][0])
        self.assertTrue(all(card["score_details"]["preferences"] == "Живой вокал"
                            for card in result["cards"]))
        self.assertEqual(filtered, before)
        result["cards"][0]["warnings"].append("Новое примечание UI")
        result["cards"][0]["evidence"][0]["value"] = "Изменено интерфейсом"
        self.assertEqual(filtered, before)

    def test_offline_cli_json_is_identical_across_processes_and_hash_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            project.mkdir()
            shutil.copy2(ROOT / "main.py", project / "main.py")
            shutil.copytree(ROOT / "app", project / "app", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "data", project / "data")
            command = [sys.executable, "-X", "utf8", str(project / "main.py"), "recommend", "--city", "Алматы",
                       "--date", "2026-09-23", "--event-type", "свадьба", "--category", "Ведущий",
                       "--budget", "2000000", "--preferences", "юмор и импровизация", "--offline"]
            outputs = []
            for seed in ("1", "777"):
                env = {**os.environ, "PYTHONHASHSEED": seed, "OPENAI_API_KEY": ""}
                completed = subprocess.run(command, cwd=directory, env=env, capture_output=True,
                                           encoding="utf-8", timeout=20)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                outputs.append(completed.stdout)
            self.assertEqual(outputs[0], outputs[1])
            result = json.loads(outputs[0])
            self.assertEqual(result["ai"]["mode"], "lexical")
            self.assertEqual(len(result["cards"]), 3)
            self.assertTrue((project / ".cache" / "ranking.sqlite3").is_file())
            self.assertFalse((Path(directory) / ".cache").exists())


if __name__ == "__main__":
    unittest.main()

"""Run: python -m unittest -v test_filtering"""
import csv
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

from data_loader import DEFAULT_CSV, DataValidationError, REQUIRED_COLUMNS, load_contractors
from filtering import RequestValidationError, filter_contractors


def profile(**changes):
    item = dict(id="A", anon_name="Тест", categories=["Ведущий"], city="Алматы",
                price_from_kzt=100000, event_formats=["свадьба"], languages=["русский"],
                max_hours=6, busy_dates=["2026-11-14"], description="Исходный текст.",
                synthetic=False, city_imputed=False, price_imputed=False)
    item.update(changes)
    return item


def query(**changes):
    item = dict(city="Алматы", date="2026-10-01", event_type="свадьба",
                category="Ведущий", budget=100000)
    item.update(changes)
    return item


class FilteringTests(unittest.TestCase):
    def test_exact_budget_and_hours_are_inclusive(self):
        result = filter_contractors([profile()], query(duration_hours=6))
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["counts"]["eligible"], 1)

    def test_all_failures_are_reported(self):
        result = filter_contractors([profile()], query(date="2026-11-14", budget=1,
            event_type="той", language="казахский", duration_hours=7))
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(len(result["excluded"][0]["reasons"]), 5)
        self.assertEqual(sum(result["reason_counts"].values()), 5)
        self.assertIn("несколько причин", result["message"])

    def test_each_rule_excludes_independently(self):
        for changes, code in [({"date": "2026-11-14"}, "busy_date"),
                              ({"budget": 99999}, "over_budget"),
                              ({"event_type": "той"}, "unsupported_format"),
                              ({"language": "казахский"}, "unsupported_language"),
                              ({"duration_hours": 6.5}, "duration_exceeded")]:
            with self.subTest(code=code):
                result = filter_contractors([profile()], query(**changes))
                self.assertEqual(result["candidates"], [])
                self.assertEqual([r["code"] for r in result["excluded"][0]["reasons"]], [code])

    def test_three_outcomes_and_empty_catalog(self):
        self.assertEqual(filter_contractors([profile()], query())["status"], "matched")
        self.assertEqual(filter_contractors([profile()], query(city="Астана"))["status"], "no_category_in_city")
        self.assertEqual(filter_contractors([profile()], query(category="Флорист"))["status"], "no_category_in_city")
        self.assertEqual(filter_contractors([], query())["status"], "no_category_in_city")
        self.assertEqual(filter_contractors([profile()], query(budget=0))["status"], "no_matches")

    def test_null_hours_mean_not_presence_bound(self):
        result = filter_contractors([profile(max_hours=None)], query(duration_hours=24))
        self.assertEqual(result["status"], "matched")
        self.assertIsNone(result["evidence"]["A"][-1]["value"])
        self.assertIn("не привязана", result["evidence"]["A"][-1]["statement"])

    def test_optional_constraints_are_not_applied(self):
        self.assertEqual(filter_contractors([profile(max_hours=1)], query(language=""))["status"], "matched")

    def test_normalization_and_exact_category_membership(self):
        result = filter_contractors([profile(categories=["Банкетный зал", "Ресторан"])],
                                    query(city="  АЛМАТЫ ", category="банкетный   зал", language=" РУССКИЙ "))
        self.assertEqual(result["status"], "matched")
        self.assertEqual(filter_contractors([profile()], query(category="Вед"))["status"], "no_category_in_city")

    def test_venues_obey_same_calendar(self):
        result = filter_contractors([profile(categories=["Банкетный зал"])],
                                    query(category="Банкетный зал", date="2026-11-14"))
        self.assertEqual(result["reason_counts"]["busy_date"], 1)
        self.assertEqual(result["candidates"], [])

    def test_date_changes_availability(self):
        self.assertEqual(filter_contractors([profile()], query())["status"], "matched")
        self.assertEqual(filter_contractors([profile()], query(date="2026-11-14"))["status"], "no_matches")

    def test_calendar_boundaries(self):
        for day in ("2026-09-23", "2026-12-31", date(2026, 10, 1)):
            with self.subTest(day=day):
                self.assertEqual(filter_contractors([profile()], query(date=day))["status"], "matched")
        for day in ("2026-09-22", "2027-01-01", "2026-02-30", "20261001", "2026-1-1", datetime(2026, 10, 1)):
            with self.subTest(day=day), self.assertRaises(RequestValidationError):
                filter_contractors([profile()], query(date=day))

    def test_invalid_requests_are_not_empty_results(self):
        cases = [{"budget": x} for x in (-1, True, "nan", "inf", "-Infinity", None, "oops", "1e999")]
        cases += [{"duration_hours": x} for x in (0, -1, True, "NaN")]
        cases += [{"city": " "}, {"category": None}, {"language": 1}, {"event_type": []}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(RequestValidationError):
                filter_contractors([profile()], query(**changes))

    def test_keeps_all_candidates_for_ranker(self):
        result = filter_contractors([profile(id=str(i)) for i in range(5)], query())
        self.assertEqual(len(result["candidates"]), 5)

    def test_determinism_independent_of_input_order(self):
        rows = [profile(id=str(i)) for i in range(10)]
        expected = filter_contractors(rows, query())
        for seed in range(10):
            random.Random(seed).shuffle(rows)
            self.assertEqual(filter_contractors(rows, query()), expected)

    def test_input_and_returned_data_are_isolated(self):
        rows, request = [profile()], query()
        original = deepcopy((rows, request))
        result = filter_contractors(rows, request)
        result["candidates"][0]["languages"].append("new")
        result["evidence"]["A"][1]["value"].append("new")
        self.assertEqual((rows, request), original)

    def test_flags_and_description_are_preserved(self):
        result = filter_contractors([profile(synthetic=True, city_imputed=True, price_imputed=True)], query())
        self.assertEqual(len(result["warnings"]["A"]), 4)
        self.assertEqual(result["candidates"][0]["description"], "Исходный текст.")
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_rare_result_explanation(self):
        result = filter_contractors([profile()], query())
        self.assertIn("меньше трёх профилей", result["message"])
        result = filter_contractors([profile(), profile(id="B", price_from_kzt=200000)], query())
        self.assertIn("цена «от» выше бюджета: 1", result["message"])


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("sample.csv")
        with DEFAULT_CSV.open(encoding="utf-8-sig", newline="") as source:
            self.sample = next(csv.DictReader(source))

    def load_text(self, text):
        with patch.object(Path, "open", return_value=io.StringIO(text)):
            return load_contractors(self.path)

    def write_csv(self, rows, headers=REQUIRED_COLUMNS):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        self.csv_text = output.getvalue()

    def test_real_dataset(self):
        rows = load_contractors()
        self.assertEqual(len(rows), 66)
        self.assertEqual(sum(r["synthetic"] for r in rows), 13)
        self.assertEqual(sum(r["max_hours"] is None for r in rows), 9)
        self.assertEqual(sum("Ведущий" in r["categories"] for r in rows), 15)
        self.assertEqual(sum("Банкетный зал" in r["categories"] for r in rows), 8)

    def test_quoted_multiline_and_lists(self):
        self.sample.update(description='Текст, "цитата"\nВторая строка', categories=" Флорист |Декоратор|флорист ",
                           busy_dates="2026-10-02|2026-10-01|2026-10-01", max_hours="null")
        self.write_csv([self.sample])
        result = self.load_text(self.csv_text)[0]
        self.assertEqual(result["description"], self.sample["description"])
        self.assertEqual(result["categories"], ["Флорист", "Декоратор"])
        self.assertEqual(result["busy_dates"], ["2026-10-01", "2026-10-02"])
        self.assertIsNone(result["max_hours"])
        self.assertIs(result["synthetic"], False)

    def test_empty_busy_calendar_is_valid(self):
        self.sample["busy_dates"] = ""
        self.write_csv([self.sample])
        self.assertEqual(self.load_text(self.csv_text)[0]["busy_dates"], [])

    def test_header_only_is_empty_catalog(self):
        self.write_csv([])
        self.assertEqual(self.load_text(self.csv_text), [])

    def test_bad_rows_fail_with_line_number(self):
        cases = {"id": "", "price_from_kzt": "NaN", "max_hours": "-1", "synthetic": "yes",
                 "categories": "|", "busy_dates": "2027-01-01"}
        for field, value in cases.items():
            with self.subTest(field=field):
                row = {**self.sample, field: value}
                self.write_csv([row])
                with self.assertRaisesRegex(DataValidationError, "строка 2"):
                    self.load_text(self.csv_text)

    def test_duplicate_id_rejected(self):
        self.write_csv([self.sample, self.sample])
        with self.assertRaisesRegex(DataValidationError, "повторяющийся id"):
            self.load_text(self.csv_text)

    def test_bad_headers_and_row_width(self):
        for text in ("", "id,name\na,b\n", "id,id\na,b\n", ",".join(REQUIRED_COLUMNS) + "\na,b\n"):
            with self.subTest(text=text):
                with self.assertRaises(DataValidationError):
                    self.load_text(text)


class IntegrationTests(unittest.TestCase):
    def test_real_catalog_eligibility_invariants(self):
        rows = load_contractors()
        for category in ("Ведущий", "Фотограф", "Банкетный зал", "Флорист"):
            for day in ("2026-09-23", "2026-10-01", "2026-11-14", "2026-12-31"):
                result = filter_contractors(rows, query(category=category, date=day, budget=500000, language="русский", duration_hours=6))
                self.assertEqual(result["counts"]["city_category"], len(result["candidates"]) + len(result["excluded"]))
                for item in result["candidates"]:
                    self.assertNotIn(day, item["busy_dates"])
                    self.assertLessEqual(item["price_from_kzt"], 500000)
                    self.assertIn("свадьба", item["event_formats"])
                    self.assertIn("русский", item["languages"])
                    self.assertTrue(item["max_hours"] is None or item["max_hours"] >= 6)

    def test_cli_runs_from_another_directory_and_is_deterministic_across_processes(self):
        script = Path(__file__).with_name("demo_filtering.py").resolve()
        command = [sys.executable, str(script), "--city", "Алматы", "--category", "Ведущий",
                   "--date", "2026-10-01", "--event-type", "свадьба", "--budget", "1000000"]
        outputs = []
        for seed in ("1", "7", "42"):
            result = subprocess.run(command, cwd=tempfile.gettempdir(), env={**os.environ, "PYTHONHASHSEED": seed},
                                    capture_output=True, check=True, encoding="utf-8")
            outputs.append(result.stdout)
        self.assertEqual(len(set(outputs)), 1)
        self.assertEqual(json.loads(outputs[0])["status"], "matched")
        bad = subprocess.run(command + ["--date", "2027-01-01"], capture_output=True, encoding="utf-8")
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(json.loads(bad.stderr)["status"], "error")


if __name__ == "__main__":
    unittest.main()

"""Validated, dependency-free loader for the HackAlem contractor CSV."""
from __future__ import annotations

import csv
import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

CALENDAR_START = date(2026, 9, 23)
CALENDAR_END = date(2026, 12, 31)
DEFAULT_CSV = Path(__file__).with_name("contractors.csv")
REQUIRED_COLUMNS = (
    "id", "anon_name", "categories", "city", "city_imputed", "synthetic",
    "price_from_kzt", "price_imputed", "event_formats", "languages",
    "max_hours", "busy_dates", "description",
)


class DataValidationError(ValueError):
    """Invalid catalog; never silently drop malformed contractor records."""


def normalize(value: str) -> str:
    """Case-insensitive equality with Unicode and whitespace normalization."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def parse_date(value: Any, field: str = "date") -> date:
    # datetime is intentionally rejected: the calendar contains dates, not times.
    if type(value) is date:
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        raise ValueError(f"{field}: ожидается дата YYYY-MM-DD")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field}: несуществующая дата {value!r}") from exc


def number(value: Any, field: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field}: ожидается число")
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field}: ожидается число") from exc
    if not parsed.is_finite() or parsed < 0 or (positive and parsed == 0):
        raise ValueError(f"{field}: число должно быть {'положительным' if positive else 'неотрицательным'} и конечным")
    if parsed > Decimal("1000000000000"):
        raise ValueError(f"{field}: слишком большое число")
    return int(parsed) if parsed == parsed.to_integral_value() else float(parsed)


def _list(value: str) -> list[str]:
    # Preserve source spelling for UI/evidence; remove duplicates consistently.
    result: dict[str, str] = {}
    for part in value.split("|"):
        part = " ".join(part.split())
        if part:
            result.setdefault(normalize(part), part)
    return list(result.values())


def _boolean(value: str, field: str) -> bool:
    values = {"true": True, "false": False}
    key = value.strip().casefold()
    if key not in values:
        raise ValueError(f"{field}: ожидается True или False")
    return values[key]


def load_contractors(path: str | Path = DEFAULT_CSV) -> list[dict[str, Any]]:
    """Return JSON-compatible records sorted by id; raise on any invalid row.

    Pipe-separated columns become lists; dates stay ISO strings; flags become
    bool; numbers become int/float; empty max_hours becomes None. Description
    stays verbatim. No dependencies, network, model calls or implicit imputation.
    """
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            headers = reader.fieldnames
            if not headers:
                raise DataValidationError("CSV пуст: отсутствует заголовок")
            if len(headers) != len(set(headers)):
                raise DataValidationError("CSV: повторяющиеся названия колонок")
            missing = sorted(set(REQUIRED_COLUMNS) - set(headers))
            if missing:
                raise DataValidationError("CSV: отсутствуют колонки: " + ", ".join(missing))
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("число ячеек не совпадает с заголовком")
                    record: dict[str, Any] = {key: row[key].strip() for key in REQUIRED_COLUMNS}
                    for field in ("id", "anon_name", "city"):
                        if not record[field]:
                            raise ValueError(f"{field}: обязательное поле пусто")
                    if record["id"] in ids:
                        raise ValueError(f"повторяющийся id: {record['id']}")
                    for field in ("categories", "event_formats", "languages"):
                        record[field] = _list(row[field])
                        if not record[field]:
                            raise ValueError(f"{field}: список пуст")
                    record["busy_dates"] = sorted({parse_date(d, "busy_dates").isoformat() for d in _list(row["busy_dates"])})
                    if any(not CALENDAR_START.isoformat() <= d <= CALENDAR_END.isoformat() for d in record["busy_dates"]):
                        raise ValueError("busy_dates: дата вне окна календаря")
                    for field in ("synthetic", "city_imputed", "price_imputed"):
                        record[field] = _boolean(row[field], field)
                    record["price_from_kzt"] = number(row["price_from_kzt"], "price_from_kzt")
                    hours = row["max_hours"].strip()
                    record["max_hours"] = None if not hours or hours.casefold() == "null" else number(hours, "max_hours", positive=True)
                    record["description"] = row["description"]
                    ids.add(record["id"])
                    records.append(record)
                except ValueError as exc:
                    raise DataValidationError(f"CSV, строка {reader.line_num}: {exc}") from exc
    except (UnicodeError, csv.Error) as exc:
        raise DataValidationError(f"Некорректный CSV UTF-8: {exc}") from exc
    return sorted(records, key=lambda item: item["id"])

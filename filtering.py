"""Strict filtering and evidence for ranking/UI. No ranking or AI here."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from data_loader import CALENDAR_END, CALENDAR_START, normalize, number, parse_date

REASON_LABELS = {
    "busy_date": "заняты на выбранную дату",
    "over_budget": "цена «от» выше бюджета",
    "unsupported_format": "не работают с выбранным форматом",
    "unsupported_language": "не указан нужный язык",
    "duration_exceeded": "превышена максимальная длительность",
}


class RequestValidationError(ValueError):
    """User input error, distinct from a valid request with no matches."""


def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise RequestValidationError("Запрос должен быть словарём")
    try:
        result: dict[str, Any] = {}
        for key in ("city", "category", "event_type"):
            value = request.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key}: обязательная непустая строка")
            result[key] = " ".join(value.split())
        event_date = parse_date(request.get("date"))
        if not CALENDAR_START <= event_date <= CALENDAR_END:
            raise ValueError(f"Доступность известна только с {CALENDAR_START} по {CALENDAR_END}")
        result["date"] = event_date.isoformat()
        result["budget"] = number(request.get("budget"), "budget")
        duration = request.get("duration_hours")
        result["duration_hours"] = None if duration is None or duration == "" else number(duration, "duration_hours", positive=True)
        language = request.get("language")
        if language is not None and not isinstance(language, str):
            raise ValueError("language: ожидается строка")
        result["language"] = " ".join(language.split()) if language and language.strip() else None
        return result
    except ValueError as exc:
        raise RequestValidationError(str(exc)) from exc


def _contains(items: Iterable[str], expected: str) -> bool:
    return any(normalize(item) == normalize(expected) for item in items)


def filter_contractors(contractors: Iterable[Mapping[str, Any]], request: Mapping[str, Any]) -> dict[str, Any]:
    """Filter records from load_contractors; return ALL eligible candidates.

    Input is not mutated. Stable id ordering is the integration baseline, not
    a relevance score. Ranking must run after this function and use id as the
    final tiebreaker. Each rejection retains all failed conditions.
    """
    query = validate_request(request)
    catalog = list(contractors)
    ids = [item["id"] for item in catalog]
    if len(set(ids)) != len(ids):
        raise ValueError("Каталог содержит повторяющиеся id; используйте load_contractors")
    pool = sorted((c for c in catalog if normalize(c["city"]) == normalize(query["city"])
                   and _contains(c["categories"], query["category"])), key=lambda c: c["id"])
    eligible, excluded = [], []
    counts = dict.fromkeys(REASON_LABELS, 0)
    evidence: dict[str, list[dict[str, Any]]] = {}
    warnings: dict[str, list[str]] = {}
    for contractor in pool:
        reasons = []

        def reject(code: str, field: str, actual: Any, expected: Any, message: str) -> None:
            reasons.append({"code": code, "field": field, "actual": deepcopy(actual),
                            "expected": expected, "message": message})
            counts[code] += 1

        if query["date"] in contractor["busy_dates"]:
            reject("busy_date", "busy_dates", query["date"], "free", f"Занят {query['date']} по календарю CSV")
        if contractor["price_from_kzt"] > query["budget"]:
            reject("over_budget", "price_from_kzt", contractor["price_from_kzt"], query["budget"],
                   f"Цена от {contractor['price_from_kzt']:g} ₸ выше бюджета {query['budget']:g} ₸")
        if not _contains(contractor["event_formats"], query["event_type"]):
            reject("unsupported_format", "event_formats", contractor["event_formats"], query["event_type"],
                   f"Формат «{query['event_type']}» отсутствует в профиле")
        if query["language"] and not _contains(contractor["languages"], query["language"]):
            reject("unsupported_language", "languages", contractor["languages"], query["language"],
                   f"Язык «{query['language']}» отсутствует в профиле")
        hours = contractor["max_hours"]
        if query["duration_hours"] is not None and hours is not None and hours < query["duration_hours"]:
            reject("duration_exceeded", "max_hours", hours, query["duration_hours"],
                   f"Максимум {hours:g} ч, запрошено {query['duration_hours']:g} ч")
        if reasons:
            excluded.append({"id": contractor["id"], "anon_name": contractor["anon_name"], "reasons": reasons})
            continue
        eligible.append(deepcopy(dict(contractor)))
        facts = [
            {"field": "city", "value": contractor["city"]},
            {"field": "categories", "value": deepcopy(contractor["categories"])},
            {"field": "busy_dates", "value": query["date"], "statement": "дата отсутствует среди занятых в CSV"},
            {"field": "price_from_kzt", "value": contractor["price_from_kzt"], "budget": query["budget"]},
            {"field": "event_formats", "value": deepcopy(contractor["event_formats"])},
        ]
        if query["language"]:
            facts.append({"field": "languages", "value": deepcopy(contractor["languages"])})
        if query["duration_hours"] is not None:
            facts.append({"field": "max_hours", "value": hours, "requested": query["duration_hours"],
                          "statement": "работа не привязана к присутствию" if hours is None else "длительность в пределах лимита"})
        evidence[contractor["id"]] = facts
        notes = ["Цена указана «от» за мероприятие; итоговая стоимость требует уточнения."]
        for flag, text in (("synthetic", "Синтетический профиль из исходного датасета."),
                           ("city_imputed", "Город проставлен при подготовке датасета."),
                           ("price_imputed", "Цена проставлена при подготовке датасета.")):
            if contractor[flag]:
                notes.append(text)
        warnings[contractor["id"]] = notes
    status = "no_category_in_city" if not pool else "matched" if eligible else "no_matches"
    details = "; ".join(f"{REASON_LABELS[code]}: {count}" for code, count in counts.items() if count)
    if status == "no_category_in_city":
        message = f"В городе «{query['city']}» нет подрядчиков категории «{query['category']}»."
    elif status == "no_matches":
        message = f"В городе есть {len(pool)} кандидатов этой категории, но никто не прошёл условия: {details}."
    elif len(eligible) < 3:
        message = f"Подходят {len(eligible)} из {len(pool)} кандидатов. Меньше трёх: "
        message += f"{details}." if details else "в каталоге этого города и категории меньше трёх профилей."
    else:
        message = f"Подходят {len(eligible)} из {len(pool)} кандидатов; модуль ранжирования выберет до трёх."
        if details:
            message += f" Причины исключения остальных: {details}."
    if details:
        message += " Один кандидат может иметь несколько причин исключения."
    return {"status": status, "message": message, "request": query, "candidates": eligible,
            "excluded": excluded, "reason_counts": counts, "evidence": evidence, "warnings": warnings,
            "counts": {"catalog": len(catalog), "city_category": len(pool), "eligible": len(eligible),
                       "excluded": len(excluded)},
            "calendar_window": {"start": CALENDAR_START.isoformat(), "end": CALENDAR_END.isoformat()}}

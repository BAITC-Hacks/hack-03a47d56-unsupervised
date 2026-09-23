"""Deterministic Russian explanations grounded in catalog fields and quotations.

The source description is data: it is never interpreted as instructions. AI
similarity is not evidence for experience, quality, availability, or a price.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from .data_loader import CALENDAR_END, CALENDAR_START, normalize, number, parse_date

EXCERPT_LIMIT = 180
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = {
    "хочу", "нужен", "нужна", "нужно", "ищем", "ищу", "чтобы", "были",
    "будет", "быть", "очень", "мероприятие", "ведущий", "ведущего", "который",
}
_FEATURES = (
    "стиль", "импровизац", "интерактив", "сценари", "юмор", "квн", "команд",
    "оборудован", "вокал", "танцев", "авторск", "цветочн", "резидент",
    "финалист", "победител", "съем", "съём", "монтаж", "кухн", "вместим",
    "кадр", "момент", "репертуар", "подач", "язык", "атмосфер",
)
_NAME = r"[A-ZА-ЯЁ][\w'’\-]*(?:\s+[A-ZА-ЯЁ][\w'’\-]*)*"
_ROLE = (
    r"(?i:(?:(?:свадебн\w*|профессиональн\w*|опытн\w*)\s+)*"
    r"(?:фотограф|видеограф|ведущ\w*|церемониймейстер|флорист|декоратор|"
    r"музыкант|вокалист\w*|группа|команда|студия))"
)
_BARE_INTRODUCTION = re.compile(
    # Full-fragment matches avoid dropping a useful sentence merely because
    # it starts with a name. Brand/role words alone add no service evidence.
    rf"^(?:(?i:я|мы)\s+(?:[—–-]\s*)?{_NAME}(?:\s*[—–-]\s*{_ROLE})?|"
    rf"(?:(?i:я|мы)\s+(?:[—–-]\s*)?)?{_ROLE}"
    rf"(?:\s+(?i:и)\s+{_ROLE})*\s+{_NAME})[.!?]*$"
)


def _tokens(value: str) -> set[str]:
    return {word for word in _WORDS.findall(value.casefold().replace("ё", "е"))
            if len(word) >= 3 and word not in _STOP_WORDS}


def _matches(word: str, keyword: str) -> bool:
    # A small prefix comparison selects an excerpt only; it makes no semantic
    # claim. Ranking itself belongs to scorer.py.
    return word == keyword or (min(len(word), len(keyword)) >= 4
                               and word[:5] == keyword[:5]) or (
        len(keyword) == 4 and word.startswith(keyword))


def _is_boilerplate(fragment: str) -> bool:
    """Introductions, contacts and instructions are not service evidence."""
    text = " ".join(fragment.split()).lstrip("«\"'—–- ")
    lowered = text.casefold()
    if re.match(r"^(с уважением|с наилучшими пожеланиями)\b", lowered):
        return True
    if _BARE_INTRODUCTION.fullmatch(text):
        return True
    if re.match(r"^(игнорируй\w*|забудь\w*|не учитывай\w*|"
                r"ignore\s+(?:all\s+|previous\s+|the\s+)*instructions)\b", lowered):
        return True
    has_details = any(feature in lowered for feature in _FEATURES) or bool(
        re.search(r"\d+\s*(лет|года|свад|гостей|человек|мест)", lowered))
    if has_details:
        return False
    return bool(re.match(
        r"^(привет\w*|здравствуй\w*|всем привет|добрый день|доброго дня|"
        r"дорогие друзья|меня зовут|связаться со мной|пишите|звоните|"
        r"более подробн\w* информац\w*|подробности|контакты)\b", lowered))


def _clip(fragment: str) -> str:
    fragment = fragment.strip()
    if len(fragment) > EXCERPT_LIMIT:
        shortened = fragment[:EXCERPT_LIMIT + 1]
        boundary = shortened.rfind(" ")
        fragment = shortened[:boundary] if boundary > 0 else ""
    return fragment.rstrip(" \t\r\n.!?;,:…")


def select_description_excerpt(description: str, preferences: str = "") -> str:
    """Return a source substring, at most 180 chars, selected deterministically.

    Whitespace inside the quotation is preserved for verifiable grounding.
    Preference words guide selection without claiming semantic agreement.
    Non-string/empty descriptions produce an empty string.
    """
    if not isinstance(description, str) or not description.strip():
        return ""
    keywords = _tokens(preferences) if isinstance(preferences, str) else set()
    fragments: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|[•\r\n]+", description):
        if _is_boilerplate(sentence):
            continue
        fragment = _clip(sentence)
        if fragment:
            fragments.append(fragment)
        # A long unpunctuated profile may mention the requested detail near its
        # end. Add windows beginning at that word, still exact source slices.
        if len(sentence) > EXCERPT_LIMIT and keywords:
            for match in _WORDS.finditer(sentence):
                word = match.group().casefold().replace("ё", "е")
                if any(_matches(word, keyword) for keyword in keywords):
                    fragment = _clip(sentence[match.start():])
                    if fragment:
                        fragments.append(fragment)
    if not fragments:
        return ""

    def relevance(fragment: str) -> tuple[int, int]:
        lowered = fragment.casefold()
        tokens = _tokens(fragment)
        preference_hits = sum(any(_matches(word, key) for word in tokens)
                              for key in keywords)
        detail_hits = sum(feature in lowered for feature in _FEATURES)
        if re.search(r"\d+\s*(лет|года|свад|гостей|человек|мест)", lowered):
            detail_hits += 2
        if any(phrase in lowered for phrase in (
            "привет", "меня зовут", "с уважением", "самых востребованных",
            "вы можете не сомневаться", "профессиональный ведущий",
        )):
            detail_hits -= 4
        return preference_hits, detail_hits

    # max preserves the first source occurrence on equal scores.
    return max(fragments, key=relevance)


def _display_number(value: Any) -> str | None:
    try:
        parsed = number(value, "value")
    except (ValueError, TypeError):
        return None
    return f"{parsed:,}".replace(",", " ")


def _list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, (list, tuple)) else []


def _contains(values: Any, expected: str) -> bool:
    return any(normalize(value) == normalize(expected) for value in _list(values))


def generate_explanation(
    contractor: dict, request: dict, score_details: dict | None = None,
) -> str:
    """Build two compact sentences from facts, plus an attributed source quote.

    Call with a normalized contractor and request from filtering.py. Optional
    score_details["preferences"] restores the free text omitted by filtering;
    arbitrary generated text/evidence in score_details is intentionally unused.
    Inputs are not modified, and this function makes no API calls.
    """
    facts: list[str] = []
    event_type = request.get("event_type")
    if isinstance(event_type, str) and event_type.strip():
        if _contains(contractor.get("event_formats"), event_type):
            facts.append(f"Формат «{event_type}» указан в профиле")
        else:
            facts.append(f"Формат «{event_type}» не указан в профиле")

    price = _display_number(contractor.get("price_from_kzt"))
    budget = _display_number(request.get("budget"))
    if price is not None:
        price_fact = f"цена от {price} ₸"
        if budget is not None:
            price_fact += f" при бюджете {budget} ₸"
        facts.append(price_fact + " (итог уточняется)")

    try:
        event_date = parse_date(request.get("date"))
    except (ValueError, TypeError):
        event_date = None
    if event_date is not None:
        day = event_date.isoformat()
        if not CALENDAR_START <= event_date <= CALENDAR_END:
            facts.append(f"доступность на {day} вне окна календаря CSV")
        elif not isinstance(contractor.get("busy_dates"), (list, tuple)):
            facts.append(f"доступность на {day} не указана")
        elif day in contractor["busy_dates"]:
            facts.append(f"{day} отмечено занятым в календаре CSV")
        else:
            facts.append(f"{day} не отмечено занятым в календаре CSV")

    language = request.get("language")
    if isinstance(language, str) and language.strip():
        known = _contains(contractor.get("languages"), language)
        facts.append(f"язык «{language}» {'указан' if known else 'не указан'}")
    duration = _display_number(request.get("duration_hours"))
    if duration is not None:
        hours = _display_number(contractor.get("max_hours"))
        if "max_hours" in contractor and contractor["max_hours"] is None:
            facts.append("работа не привязана к присутствию на площадке")
        elif hours is None:
            facts.append(f"для {duration} ч лимит часов не указан, длительность нужно уточнить")
        else:
            facts.append(f"запрошено {duration} ч, лимит — {hours} ч")

    preferences = request.get("preferences", "")
    if isinstance(score_details, Mapping) and isinstance(score_details.get("preferences"), str):
        preferences = score_details["preferences"]
    excerpt = select_description_excerpt(contractor.get("description"), preferences)
    fact_sentence = "; ".join(facts) if facts else "Условия в профиле не указаны"
    if excerpt:
        description = contractor["description"]
        remainder = description[description.find(excerpt) + len(excerpt):].lstrip()
        suffix = "…" if remainder and remainder[0] not in ".!?;,:…" else ""
        source_sentence = f"В описании: «{excerpt}{suffix}»."
    else:
        source_sentence = "Описание с дополнительными деталями не предоставлено."
    return f"{fact_sentence}. {source_sentence}"

"""Adapter from participant 1's FilterResult to participant 3's UI cards."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from explainer import generate_explanation, select_description_excerpt
from scorer import RankingEngine, normalize_preferences


def recommend_from_filtered(filtered: Mapping[str, Any], *, preferences: str | None = None,
                            engine: RankingEngine | None = None) -> dict[str, Any]:
    """Keep filtering diagnostics; rank the whole pool, then return up to 3 cards.

    Pass preferences explicitly: gulya's validate_request intentionally preserves
    only the original seven fields, so filtered['request'] does not retain it.
    """
    result = deepcopy(dict(filtered))
    wishes = normalize_preferences(filtered["request"].get("preferences") if preferences is None else preferences)
    result["request"]["preferences"] = wishes
    result["cards"] = []
    result["ai"] = {"mode": "not_used", "model": None, "message": "Нет кандидатов для ранжирования."}
    if filtered["status"] != "matched":
        return result
    ranking = (engine or RankingEngine()).rank(filtered["candidates"], filtered["request"], preferences=wishes)
    result["ai"] = ranking["ai"]
    for item in ranking["ranked"][:3]:
        contractor = item["contractor"]
        contractor_id = contractor["id"]
        excerpt = select_description_excerpt(contractor.get("description", ""), wishes)
        evidence = deepcopy(filtered.get("evidence", {}).get(contractor_id, []))
        if excerpt:
            evidence.append({"field": "description", "value": excerpt})
        result["cards"].append({
            "id": contractor_id, "name": contractor["anon_name"],
            "category": filtered["request"]["category"], "categories": contractor["categories"],
            "city": contractor["city"], "price_from_kzt": contractor["price_from_kzt"],
            "synthetic": contractor["synthetic"], "city_imputed": contractor["city_imputed"],
            "price_imputed": contractor["price_imputed"], "score": item["score"],
            "score_details": item["score_details"],
            "explanation": generate_explanation(contractor, filtered["request"], item["score_details"]),
            "evidence": evidence, "warnings": deepcopy(filtered.get("warnings", {}).get(contractor_id, [])),
        })
    result["message"] = result["message"].replace(
        "модуль ранжирования выберет до трёх", f"показаны {len(result['cards'])} с наибольшими баллами"
    )
    return result

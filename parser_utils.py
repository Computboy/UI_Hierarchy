from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any, Dict, Optional

from pydantic import ValidationError

from schemas import DIMENSION_LABELS, DIMENSION_ORDER, TASK_NAME, UIHierarchyEvaluation


DIMENSION_KEY_ALIASES = {
    "visual_saliency_difference": "visual_saliency_difference",
    "grouping_compactness_separation": "grouping_compactness_separation",
    "group_compactness_and_separation": "grouping_compactness_separation",
    "grouping_compactness_and_separation": "grouping_compactness_separation",
    "alignment_consistency": "alignment_consistency",
    "reading_flow_continuity": "reading_flow_continuity",
    "visual_noise": "visual_noise",
}


def extract_json_text(raw_text: str) -> str:
    """
    Extract the first JSON object from raw model output.
    Supports pure JSON, fenced JSON, and small amounts of surrounding text.
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    raise ValueError("Could not extract a JSON object from the model output.")


def repair_json_text(json_text: str) -> str:
    """
    Repair common malformed JSON patterns produced by models.
    Current scope:
    - unescaped double quotes inside string values, e.g. 标题"热门推荐"
    """
    repaired: list[str] = []
    in_string = False
    escape_next = False
    length = len(json_text)
    index = 0

    while index < length:
        char = json_text[index]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if escape_next:
            repaired.append(char)
            escape_next = False
            index += 1
            continue

        if char == "\\":
            repaired.append(char)
            escape_next = True
            index += 1
            continue

        if char == '"':
            next_index = index + 1
            while next_index < length and json_text[next_index] in " \t\r\n":
                next_index += 1

            next_char = json_text[next_index] if next_index < length else ""
            if next_char in {",", "}", "]", ":"} or next_char == "":
                repaired.append(char)
                in_string = False
            else:
                repaired.append('\\"')
            index += 1
            continue

        repaired.append(char)
        index += 1

    return "".join(repaired)


def load_json_with_repair(json_text: str) -> Dict[str, Any]:
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as original_error:
        repaired_text = repair_json_text(json_text)
        if repaired_text != json_text:
            try:
                return json.loads(repaired_text)
            except json.JSONDecodeError as repaired_error:
                raise ValueError(
                    "Model output is not valid JSON after lightweight repair. "
                    f"Original error: line {original_error.lineno}, column {original_error.colno}, {original_error.msg}. "
                    f"Repaired error: line {repaired_error.lineno}, column {repaired_error.colno}, {repaired_error.msg}"
                ) from repaired_error

        raise ValueError(
            f"Model output is not valid JSON: line {original_error.lineno}, "
            f"column {original_error.colno}, {original_error.msg}"
        ) from original_error


def normalize_task_value(value: Any) -> str:
    if value is None:
        return TASK_NAME

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        TASK_NAME,
        "ui_hierarchy_evaluate",
        "ui_hierarchy_eval",
        "ui_hierarchy_analysis",
        "hierarchy_evaluation",
        "ui_structure_evaluation",
    }
    if text in aliases or ("ui" in text and "hierarchy" in text):
        return TASK_NAME
    return text


def normalize_confidence(value: Any) -> Any:
    if value is None:
        return None

    text = str(value).strip().lower()
    aliases = {
        "low": "low",
        "medium": "medium",
        "med": "medium",
        "moderate": "medium",
        "high": "high",
    }
    return aliases.get(text, text)


def coerce_score(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None

    try:
        score = round(float(value), 1)
    except Exception:
        return None

    return max(1.0, min(10.0, score))


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def clean_string_list(value: Any, *, max_items: Optional[int] = None) -> list[str]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [value]
    else:
        items = [value]

    cleaned = [clean_string(item) for item in items]
    cleaned = [item for item in cleaned if item]
    if max_items is not None:
        cleaned = cleaned[:max_items]
    return cleaned


def normalize_priority_item(item: Any) -> str:
    if isinstance(item, dict):
        raw_dimension = clean_string(item.get("dimension"))
        dimension_key = normalize_dimension_key(raw_dimension) if raw_dimension else None
        dimension_label = DIMENSION_LABELS.get(dimension_key, raw_dimension)
        action = clean_string(
            first_present(
                item.get("action"),
                item.get("suggestion"),
                item.get("recommendation"),
                item.get("improvement"),
                item.get("content"),
                item.get("text"),
            )
        )

        if dimension_label and action:
            return f"{dimension_label}：{action}"
        if action:
            return action
        return json.dumps(item, ensure_ascii=False)

    return clean_string(item)


def normalize_priority_items(value: Any, *, max_items: Optional[int] = None) -> list[str]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, list):
        items = value
    else:
        items = [value]

    cleaned = [normalize_priority_item(item) for item in items]
    cleaned = [item for item in cleaned if item]
    if max_items is not None:
        cleaned = cleaned[:max_items]
    return cleaned


def first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def normalize_dimension_key(key: str) -> Optional[str]:
    canonical_key = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return DIMENSION_KEY_ALIASES.get(canonical_key)


def get_dimension_source(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data.get("dimensions"), dict):
        return data["dimensions"]
    return data


def pick_dimension_payload(source: Dict[str, Any], target_key: str) -> Dict[str, Any]:
    for raw_key, raw_value in source.items():
        if normalize_dimension_key(raw_key) == target_key and isinstance(raw_value, dict):
            return raw_value
    return {}


def normalize_dimension_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    suggestion_candidates = clean_string_list(obj.get("suggestions"), max_items=1)

    return {
        "score": coerce_score(first_present(obj.get("score"), obj.get("rating"))),
        "judgment": clean_string(
            first_present(obj.get("judgment"), obj.get("reason"), obj.get("analysis"))
        ),
        "evidence": clean_string_list(
            first_present(obj.get("evidence"), obj.get("observations"), obj.get("evidences")),
            max_items=3,
        ),
        "suggestion": clean_string(
            first_present(
                obj.get("suggestion"),
                obj.get("improvement"),
                obj.get("recommendation"),
                suggestion_candidates[0] if suggestion_candidates else None,
            )
        ),
    }


def normalize_dimensions(data: Dict[str, Any]) -> Dict[str, Any]:
    source = get_dimension_source(data)
    return {
        key: normalize_dimension_result(pick_dimension_payload(source, key))
        for key in DIMENSION_ORDER
    }


def compute_overall_score(dimensions: Dict[str, Any]) -> Optional[float]:
    scores = [
        dimension.get("score")
        for dimension in dimensions.values()
        if isinstance(dimension, dict) and dimension.get("score") is not None
    ]
    if len(scores) != len(DIMENSION_ORDER):
        return None
    return round(mean(scores), 1)


def normalize_payload(data: Dict[str, Any], image_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalize slight schema drift from compatible models while keeping validation strict.
    The normalizer only maps aliases and derives overall_score from dimension scores.
    """
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")

    dimensions = normalize_dimensions(data)
    overall = data.get("overall", {}) if isinstance(data.get("overall"), dict) else {}

    overall_score = coerce_score(
        first_present(
            data.get("overall_score"),
            overall.get("score"),
            data.get("score"),
        )
    )
    if overall_score is None:
        overall_score = compute_overall_score(dimensions)

    normalized = {
        "task": normalize_task_value(data.get("task")),
        "image_name": clean_string(first_present(data.get("image_name"), image_name)),
        "overall_score": overall_score,
        "confidence": normalize_confidence(data.get("confidence")),
        "dimensions": dimensions,
        "hierarchy_summary": clean_string(
            first_present(
                data.get("hierarchy_summary"),
                data.get("summary"),
                overall.get("summary"),
            )
        ),
        "priority_improvements": normalize_priority_items(
            first_present(
                data.get("priority_improvements"),
                overall.get("priority_improvements"),
                overall.get("suggestions"),
                data.get("suggestions"),
            ),
            max_items=3,
        ),
    }
    return normalized


def format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)


def parse_and_validate(raw_text: str, image_name: Optional[str] = None) -> UIHierarchyEvaluation:
    json_text = extract_json_text(raw_text)
    data = load_json_with_repair(json_text)

    normalized = normalize_payload(data, image_name=image_name)

    try:
        return UIHierarchyEvaluation.model_validate(normalized)
    except ValidationError as exc:
        normalized_dump = json.dumps(normalized, ensure_ascii=False, indent=2)
        raise ValueError(
            "Model output failed schema validation.\n"
            f"{format_validation_error(exc)}\n"
            "Normalized payload:\n"
            f"{normalized_dump}"
        ) from exc

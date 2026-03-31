from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from schemas import FONT_TASK_NAME, FontHierarchyAssessment


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    raise ValueError("未能从模型输出中提取 JSON 对象。")


def repair_json_text(json_text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escape_next = False

    for index, char in enumerate(json_text):
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            continue

        if escape_next:
            repaired.append(char)
            escape_next = False
            continue

        if char == "\\":
            repaired.append(char)
            escape_next = True
            continue

        if char == '"':
            next_index = index + 1
            while next_index < len(json_text) and json_text[next_index] in " \t\r\n":
                next_index += 1
            next_char = json_text[next_index] if next_index < len(json_text) else ""
            if next_char in {",", "}", "]", ":"} or not next_char:
                repaired.append(char)
                in_string = False
            else:
                repaired.append('\\"')
            continue

        repaired.append(char)

    return "".join(repaired)


def load_json_with_repair(json_text: str) -> dict[str, Any]:
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        repaired = repair_json_text(json_text)
        return json.loads(repaired)


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def clean_string_list(value: Any, *, max_items: int | None = None) -> list[str]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, list):
        items = value
    else:
        items = [value]

    cleaned = [clean_string(item) for item in items]
    cleaned = [item for item in cleaned if item]
    if max_items is not None:
        cleaned = cleaned[:max_items]
    return cleaned


def normalize_confidence(value: Any) -> str:
    aliases = {
        "low": "low",
        "medium": "medium",
        "med": "medium",
        "moderate": "medium",
        "high": "high",
    }
    text = clean_string(value).lower()
    return aliases.get(text, "medium")


def coerce_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = round(float(value), 1)
    except Exception:
        return None
    return max(1.0, min(10.0, score))


def normalize_font_payload(data: dict[str, Any], image_name: str | None = None) -> dict[str, Any]:
    target = data.get("font_hierarchy_delta")
    if not isinstance(target, dict):
        target = data

    normalized = {
        "task": FONT_TASK_NAME,
        "image_name": clean_string(data.get("image_name") or image_name),
        "confidence": normalize_confidence(data.get("confidence")),
        "font_hierarchy_delta": {
            "score": coerce_score(target.get("score") or data.get("score")),
            "judgment": clean_string(target.get("judgment") or target.get("analysis")),
            "evidence": clean_string_list(target.get("evidence") or target.get("observations"), max_items=3),
            "suggestion": clean_string(target.get("suggestion") or target.get("recommendation")),
        },
    }
    return normalized


def format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)


def parse_font_hierarchy_result(raw_text: str, image_name: str | None = None) -> FontHierarchyAssessment:
    json_text = extract_json_text(raw_text)
    data = load_json_with_repair(json_text)
    normalized = normalize_font_payload(data, image_name=image_name)

    try:
        return FontHierarchyAssessment.model_validate(normalized)
    except ValidationError as exc:
        normalized_dump = json.dumps(normalized, ensure_ascii=False, indent=2)
        raise ValueError(
            "字体层级模型输出未通过校验。\n"
            f"{format_validation_error(exc)}\n"
            f"Normalized payload:\n{normalized_dump}"
        ) from exc

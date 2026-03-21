from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional
from schemas import UIHierarchyEvaluation


DIMENSION_KEYS = [
    "visual_saliency_difference",
    "group_compactness_and_separation",
    "alignment_consistency",
    "reading_flow_continuity",
    "visual_noise",
]


def extract_json_text(raw_text: str) -> str:
    """
    尝试从模型返回文本中提取 JSON。
    支持：
    - 纯 JSON
    - ```json ... ```
    - 前后混杂少量解释文本
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    raise ValueError("未能从模型输出中提取 JSON。")


def normalize_task_value(value: Any) -> str:
    """
    无论第三方模型返回什么 task，只要大体表达的是 UI 层次结构评估，
    统一映射成 schema 要求的固定值。
    """
    if not value:
        return "ui_hierarchy_evaluation"

    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    aliases = {
        "ui_hierarchy_evaluation",
        "ui_hierarchy_evaluate",
        "ui_hierarchy_eval",
        "ui_hierarchy",
        "hierarchy_evaluation",
        "ui_structure_evaluation",
        "ui_hierarchy_analysis",
    }

    if text in aliases:
        return "ui_hierarchy_evaluation"

    # 像 "UI hierarchy evaluation" 这种
    if "hierarchy" in text and "ui" in text:
        return "ui_hierarchy_evaluation"

    return "ui_hierarchy_evaluation"


def normalize_dimension_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    防御性补全单个维度字段，避免第三方返回缺字段。
    """
    score = obj.get("score", 0)

    # 防止返回 float / str
    try:
        score = int(round(float(score)))
    except Exception:
        score = 0

    score = max(0, min(10, score))

    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)

    evidence = obj.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []

    evidence = [str(x) for x in evidence]

    return {
        "score": score,
        "reason": reason,
        "evidence": evidence,
    }


def normalize_overall(obj: Dict[str, Any]) -> Dict[str, Any]:
    score = obj.get("score", 0)
    try:
        score = int(round(float(score)))
    except Exception:
        score = 0
    score = max(0, min(10, score))

    summary = obj.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    strengths = obj.get("strengths", [])
    weaknesses = obj.get("weaknesses", [])
    suggestions = obj.get("suggestions", [])

    if isinstance(strengths, str):
        strengths = [strengths]
    if isinstance(weaknesses, str):
        weaknesses = [weaknesses]
    if isinstance(suggestions, str):
        suggestions = [suggestions]

    if not isinstance(strengths, list):
        strengths = []
    if not isinstance(weaknesses, list):
        weaknesses = []
    if not isinstance(suggestions, list):
        suggestions = []

    return {
        "score": score,
        "summary": summary,
        "strengths": [str(x) for x in strengths],
        "weaknesses": [str(x) for x in weaknesses],
        "suggestions": [str(x) for x in suggestions],
    }


def normalize_payload(data: Dict[str, Any], image_name: Optional[str] = None) -> Dict[str, Any]:
    """
    将第三方模型可能返回的不规范 JSON 统一整理成目标结构。
    支持两种情况：

    1) 完整结构：
    {
      "task": "...",
      "image_name": "...",
      "dimensions": {...},
      "overall": {...}
    }

    2) 扁平结构：
    {
      "visual_saliency_difference": {...},
      ...
      "overall": {...}
    }
    """
    # 情况1：已经有 dimensions
    if "dimensions" in data:
        dimensions_raw = data.get("dimensions", {})
        dimensions = {
            k: normalize_dimension_result(dimensions_raw.get(k, {}))
            for k in DIMENSION_KEYS
        }

        normalized = {
            "task": normalize_task_value(data.get("task")),
            "image_name": data.get("image_name", image_name or "unknown_image"),
            "dimensions": dimensions,
            "overall": normalize_overall(data.get("overall", {})),
        }
        return normalized

    # 情况2：5个维度直接铺在顶层
    found_dim_keys = [k for k in DIMENSION_KEYS if k in data]
    if found_dim_keys:
        dimensions = {
            k: normalize_dimension_result(data.get(k, {}))
            for k in DIMENSION_KEYS
        }

        normalized = {
            "task": normalize_task_value(data.get("task")),
            "image_name": data.get("image_name", image_name or "unknown_image"),
            "dimensions": dimensions,
            "overall": normalize_overall(data.get("overall", {})),
        }
        return normalized

    # 原样返回，让后续校验报更明确的错
    return data


def parse_and_validate(raw_text: str, image_name: Optional[str] = None) -> UIHierarchyEvaluation:
    json_text = extract_json_text(raw_text)
    data: Dict[str, Any] = json.loads(json_text)
    normalized = normalize_payload(data, image_name=image_name)
    return UIHierarchyEvaluation.model_validate(normalized)
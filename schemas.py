from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TASK_NAME = "ui_hierarchy_evaluation"

DIMENSION_ORDER = [
    "visual_saliency_difference",
    "grouping_compactness_separation",
    "alignment_consistency",
    "reading_flow_continuity",
    "visual_noise",
]

DIMENSION_LABELS = {
    "visual_saliency_difference": "视觉显著性差异",
    "grouping_compactness_separation": "组内紧密与分离度",
    "alignment_consistency": "对齐一致性",
    "reading_flow_continuity": "阅读流连续性",
    "visual_noise": "视觉干扰度",
}


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DimensionEvaluation(StrictBaseModel):
    score: float = Field(
        ...,
        ge=1,
        le=10,
        description="Hierarchy dimension score on a 1-10 scale.",
    )
    judgment: str = Field(
        ...,
        min_length=8,
        description="One-sentence judgment focused on hierarchy quality.",
    )
    evidence: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="2-3 visible layout evidence points from the screenshot.",
    )
    suggestion: str = Field(
        ...,
        min_length=8,
        description="One actionable hierarchy improvement for this dimension.",
    )

    @model_validator(mode="after")
    def validate_text_fields(self) -> "DimensionEvaluation":
        if any(not item.strip() for item in self.evidence):
            raise ValueError("Evidence items must be non-empty strings.")
        return self


class HierarchyDimensions(StrictBaseModel):
    visual_saliency_difference: DimensionEvaluation
    grouping_compactness_separation: DimensionEvaluation
    alignment_consistency: DimensionEvaluation
    reading_flow_continuity: DimensionEvaluation
    visual_noise: DimensionEvaluation


class UIHierarchyEvaluation(StrictBaseModel):
    task: Literal[TASK_NAME]
    image_name: str = Field(..., min_length=1)
    overall_score: float = Field(
        ...,
        ge=1,
        le=10,
        description="Overall hierarchy score. Prefer the mean of the five dimension scores.",
    )
    confidence: Literal["low", "medium", "high"]
    dimensions: HierarchyDimensions
    hierarchy_summary: str = Field(
        ...,
        min_length=20,
        description="A synthesized summary across the five hierarchy dimensions.",
    )
    priority_improvements: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="1-3 highest-priority improvements tied to low-scoring dimensions.",
    )

    @model_validator(mode="after")
    def validate_priority_items(self) -> "UIHierarchyEvaluation":
        if any(not item.strip() for item in self.priority_improvements):
            raise ValueError("Priority improvements must be non-empty strings.")
        return self


def get_json_schema_dict() -> Dict[str, Any]:
    return {
        "name": TASK_NAME,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task": {
                    "type": "string",
                    "const": TASK_NAME,
                },
                "image_name": {
                    "type": "string",
                    "minLength": 1,
                },
                "overall_score": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "dimensions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "visual_saliency_difference": {
                            "$ref": "#/$defs/dimensionEvaluation"
                        },
                        "grouping_compactness_separation": {
                            "$ref": "#/$defs/dimensionEvaluation"
                        },
                        "alignment_consistency": {
                            "$ref": "#/$defs/dimensionEvaluation"
                        },
                        "reading_flow_continuity": {
                            "$ref": "#/$defs/dimensionEvaluation"
                        },
                        "visual_noise": {
                            "$ref": "#/$defs/dimensionEvaluation"
                        },
                    },
                    "required": DIMENSION_ORDER,
                },
                "hierarchy_summary": {
                    "type": "string",
                    "minLength": 20,
                },
                "priority_improvements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "string",
                        "minLength": 8,
                    },
                },
            },
            "required": [
                "task",
                "image_name",
                "overall_score",
                "confidence",
                "dimensions",
                "hierarchy_summary",
                "priority_improvements",
            ],
            "$defs": {
                "dimensionEvaluation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "judgment": {
                            "type": "string",
                            "minLength": 8,
                        },
                        "evidence": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "minLength": 4,
                            },
                        },
                        "suggestion": {
                            "type": "string",
                            "minLength": 8,
                        },
                    },
                    "required": ["score", "judgment", "evidence", "suggestion"],
                }
            },
        },
        "strict": True,
    }

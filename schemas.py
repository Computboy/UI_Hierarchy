from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TASK_NAME = "ui_hierarchy_evaluation"
FONT_TASK_NAME = "font_hierarchy_delta_assessment"

DIMENSION_ORDER = [
    "visual_saliency_difference",
    "grouping_compactness_separation",
    "alignment_consistency",
]

DIMENSION_LABELS = {
    "visual_saliency_difference": "视觉显著性差异",
    "grouping_compactness_separation": "组内紧密与组间分离度",
    "alignment_consistency": "对齐一致性",
}

METRIC_LABELS = {
    "font_hierarchy_delta": "字体层级差值",
    "visual_weight_delta": "视觉权重差值",
    "region_area_delta": "区域面积差值",
    "foreground_background_contrast_delta": "前景背景对比差值",
    "within_group_distance_mean": "组内距离均值",
    "between_group_distance_mean": "组间距离均值",
    "spatial_cluster_compactness": "空间聚类紧凑度",
    "group_interval_ratio": "分组间隔比",
    "edge_alignment_error": "边缘对齐误差",
    "center_axis_alignment_error": "中轴对齐误差",
    "grid_consistency": "栅格一致性",
    "collinear_element_ratio": "共线元素占比",
}

DIMENSION_METRIC_KEYS = {
    "visual_saliency_difference": [
        "font_hierarchy_delta",
        "visual_weight_delta",
        "region_area_delta",
        "foreground_background_contrast_delta",
    ],
    "grouping_compactness_separation": [
        "within_group_distance_mean",
        "between_group_distance_mean",
        "spatial_cluster_compactness",
        "group_interval_ratio",
    ],
    "alignment_consistency": [
        "edge_alignment_error",
        "center_axis_alignment_error",
        "grid_consistency",
        "collinear_element_ratio",
    ],
}


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MetricEvaluation(StrictBaseModel):
    key: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    method: Literal["opencv", "multimodal_llm", "heuristic_fallback"]
    raw_value: float | None = None
    unit: str = Field(default="")
    normalized_score: float = Field(..., ge=1, le=10)
    formula: str = Field(..., min_length=6)
    interpretation: str = Field(..., min_length=6)


class DimensionEvaluation(StrictBaseModel):
    score: float = Field(..., ge=1, le=10)
    opencv_score: float | None = Field(default=None, ge=1, le=10)
    multimodal_score: float | None = Field(default=None, ge=1, le=10)
    judgment: str = Field(..., min_length=8)
    evidence: list[str] = Field(..., min_length=2, max_length=4)
    suggestion: str = Field(..., min_length=8)
    metrics: list[MetricEvaluation] = Field(..., min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_text_fields(self) -> "DimensionEvaluation":
        if any(not item.strip() for item in self.evidence):
            raise ValueError("Evidence items must be non-empty strings.")
        return self


class HierarchyDimensions(StrictBaseModel):
    visual_saliency_difference: DimensionEvaluation
    grouping_compactness_separation: DimensionEvaluation
    alignment_consistency: DimensionEvaluation


class DetectionSummary(StrictBaseModel):
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    detected_elements: int = Field(..., ge=0)
    detected_groups: int = Field(..., ge=0)
    llm_used: bool
    llm_status: str = Field(..., min_length=4)


class UIHierarchyEvaluation(StrictBaseModel):
    task: Literal[TASK_NAME]
    image_name: str = Field(..., min_length=1)
    overall_score: float = Field(..., ge=1, le=10)
    confidence: Literal["low", "medium", "high"]
    method_summary: str = Field(..., min_length=12)
    detection_summary: DetectionSummary
    dimensions: HierarchyDimensions
    hierarchy_summary: str = Field(..., min_length=20)
    priority_improvements: list[str] = Field(..., min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_priority_items(self) -> "UIHierarchyEvaluation":
        if any(not item.strip() for item in self.priority_improvements):
            raise ValueError("Priority improvements must be non-empty strings.")
        return self


class FontHierarchyDimension(StrictBaseModel):
    score: float = Field(..., ge=1, le=10)
    judgment: str = Field(..., min_length=8)
    evidence: list[str] = Field(..., min_length=2, max_length=3)
    suggestion: str = Field(..., min_length=8)

    @model_validator(mode="after")
    def validate_evidence(self) -> "FontHierarchyDimension":
        if any(not item.strip() for item in self.evidence):
            raise ValueError("Evidence items must be non-empty strings.")
        return self


class FontHierarchyAssessment(StrictBaseModel):
    task: Literal[FONT_TASK_NAME]
    image_name: str = Field(..., min_length=1)
    confidence: Literal["low", "medium", "high"]
    font_hierarchy_delta: FontHierarchyDimension


def get_font_hierarchy_schema_dict() -> Dict[str, Any]:
    return {
        "name": FONT_TASK_NAME,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task": {
                    "type": "string",
                    "const": FONT_TASK_NAME,
                },
                "image_name": {
                    "type": "string",
                    "minLength": 1,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "font_hierarchy_delta": {
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
                },
            },
            "required": ["task", "image_name", "confidence", "font_hierarchy_delta"],
        },
        "strict": True,
    }

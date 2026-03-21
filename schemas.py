from __future__ import annotations

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field, conint


Score = conint(ge=0, le=10)


class DimensionResult(BaseModel):
    score: Score = Field(..., description="0-10分，10分最佳")
    reason: str = Field(..., description="该维度评分理由，简明但具体")
    evidence: List[str] = Field(default_factory=list, description="可观察到的界面证据点")


class OverallResult(BaseModel):
    score: Score = Field(..., description="整体层次结构总分，0-10")
    summary: str = Field(..., description="整体评价摘要")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class UIHierarchyEvaluation(BaseModel):
    task: Literal["ui_hierarchy_evaluation"] = "ui_hierarchy_evaluation"
    image_name: str
    dimensions: Dict[str, DimensionResult]
    overall: OverallResult


def get_json_schema_dict() -> Dict[str, Any]:
    """
    给 Responses API 的 json_schema 使用。
    """
    return {
        "name": "ui_hierarchy_evaluation",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task": {
                    "type": "string",
                    "const": "ui_hierarchy_evaluation"
                },
                "image_name": {"type": "string"},
                "dimensions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "visual_saliency_difference": {
                            "$ref": "#/$defs/dimensionResult"
                        },
                        "group_compactness_and_separation": {
                            "$ref": "#/$defs/dimensionResult"
                        },
                        "alignment_consistency": {
                            "$ref": "#/$defs/dimensionResult"
                        },
                        "reading_flow_continuity": {
                            "$ref": "#/$defs/dimensionResult"
                        },
                        "visual_noise": {
                            "$ref": "#/$defs/dimensionResult"
                        }
                    },
                    "required": [
                        "visual_saliency_difference",
                        "group_compactness_and_separation",
                        "alignment_consistency",
                        "reading_flow_continuity",
                        "visual_noise"
                    ]
                },
                "overall": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "summary": {"type": "string"},
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "weaknesses": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "suggestions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["score", "summary", "strengths", "weaknesses", "suggestions"]
                }
            },
            "required": ["task", "image_name", "dimensions", "overall"],
            "$defs": {
                "dimensionResult": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reason": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["score", "reason", "evidence"]
                }
            }
        },
        "strict": True
    }
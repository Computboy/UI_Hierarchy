from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any

from adapters import build_multimodal_hierarchy_evaluator
from config import Settings
from cv_analysis import CVAnalysisResult, analyze_image_with_opencv
from parser_utils import parse_font_hierarchy_result, parse_grouping_compactness_result
from schemas import (
    DIMENSION_LABELS,
    DIMENSION_METRIC_KEYS,
    DetectionSummary,
    DimensionEvaluation,
    FontHierarchyAssessment,
    GroupingCompactnessAssessment,
    HierarchyDimensions,
    MetricEvaluation,
    UIHierarchyEvaluation,
)


@dataclass
class PipelineArtifacts:
    overlay_image: Any
    llm_raw_text: str | None
    llm_error: str | None


SALIENCY_WEIGHTS = {
    "font_hierarchy_delta": 0.35,
    "visual_weight_delta": 0.30,
    "region_area_delta": 0.20,
    "foreground_background_contrast_delta": 0.15,
}


DIMENSION_EFFECTS = {
    "visual_saliency_difference": "用户很难快速分辨第一层与第二层信息",
    "grouping_compactness_separation": "用户不容易判断哪些元素属于同一信息模块",
    "alignment_consistency": "版面会显得缺少秩序，阅读时更容易被结构噪声打断",
}


METRIC_SUGGESTIONS = {
    "font_hierarchy_delta": "拉开标题、摘要、正文之间的字号和字重差异，让第一层和第二层文本更容易区分。",
    "visual_weight_delta": "减少同时高亮的模块数量，把面积、对比和视觉焦点集中到一个核心区域。",
    "region_area_delta": "适当放大核心模块、压缩次级模块面积，形成更明确的面积层级。",
    "foreground_background_contrast_delta": "增强重点区域与背景的明暗或色彩对比，避免主要信息淹没在周围内容里。",
    "within_group_distance_mean": "缩短同组元素之间的间距，优先保证同一模块内部的聚拢关系。",
    "between_group_distance_mean": "增加不同模块之间的留白或边界，让模块切分更清楚。",
    "spatial_cluster_compactness": "把同一组元素收拢到更稳定的局部空间内，减少组内漂移。",
    "group_interval_ratio": "同时拉大组间距离、压缩组内距离，形成更鲜明的分组间隔比。",
    "edge_alignment_error": "让主要模块沿统一左边线、右边线或顶部边线排列，减少无意义错位。",
    "center_axis_alignment_error": "让核心卡片、标题和按钮共享稳定中轴，避免视觉重心来回跳动。",
    "grid_consistency": "用统一的列宽和行距组织内容，让元素更稳定地回到同一套栅格里。",
    "collinear_element_ratio": "提高共线元素比例，让更多元素挂靠到共同参考线，而不是各自漂移。",
}


DIMENSION_GUIDANCE = {
    "visual_saliency_difference": {
        "strong": "该页面的主次视觉层级较清楚，主要信息能够先被看见。",
        "medium": "该页面有一定视觉层级，但主次差距还没有完全拉开。",
        "weak": "该页面的视觉主次偏混杂，第一层信息不够稳定。",
    },
    "grouping_compactness_separation": {
        "strong": "相关元素的聚合关系较稳定，模块之间也有清楚分隔。",
        "medium": "页面可以感知到分组，但局部模块仍有粘连或松散问题。",
        "weak": "页面分组关系偏弱，元素归属和模块边界都不够明确。",
    },
    "alignment_consistency": {
        "strong": "页面沿共同参考线组织内容，结构秩序较稳定。",
        "medium": "页面存在主要对齐线，但局部偏移仍然明显。",
        "weak": "页面缺少稳定对齐关系，结构秩序感偏弱。",
    },
}


def _image_name(image_path: str) -> str:
    return Path(image_path).name


def _sanitize_llm_error(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***REDACTED***", message)


def _explain_llm_error(message: str, settings: Settings) -> str:
    text = _sanitize_llm_error(message)
    if "Claude-code" in text and "poloai.top" in settings.base_url:
        return (
            f"{text} "
            "This came from the poloai gateway before model execution, which suggests the current API key or account "
            "does not have permission for the routed model group."
        )
    return text


def _weighted_score(metric_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values())
    total = sum(metric_scores[key] * weight for key, weight in weights.items())
    return round(total / total_weight, 1) if total_weight else 1.0


def _simple_average(metric_scores: list[float]) -> float:
    return round(mean(metric_scores), 1) if metric_scores else 1.0


def _transform_grouping_dimension_score(score: float) -> float:
    return round(max(1.0, min(10.0, score)), 1)


def _to_metric_model(measurement) -> MetricEvaluation:
    return MetricEvaluation(
        key=measurement.key,
        label=measurement.label,
        method=measurement.method,
        raw_value=measurement.raw_value,
        unit=measurement.unit,
        normalized_score=measurement.normalized_score,
        formula=measurement.formula,
        interpretation=measurement.interpretation,
    )


def _build_font_metric(
    llm_result: FontHierarchyAssessment | None,
    fallback_measurement,
) -> tuple[MetricEvaluation, list[str], float | None]:
    if llm_result is not None:
        metric = MetricEvaluation(
            key="font_hierarchy_delta",
            label="字体层级差异",
            method="multimodal_llm",
            raw_value=None,
            unit="score",
            normalized_score=llm_result.font_hierarchy_delta.score,
            formula="多模态模型依据截图中的字号、字重与文本层级可见差异直接评估。",
            interpretation=llm_result.font_hierarchy_delta.judgment,
        )
        return metric, llm_result.font_hierarchy_delta.evidence, llm_result.font_hierarchy_delta.score

    return (
        MetricEvaluation(
            key="font_hierarchy_delta",
            label="字体层级差异",
            method="heuristic_fallback",
            raw_value=fallback_measurement.raw_value,
            unit=fallback_measurement.unit,
            normalized_score=fallback_measurement.normalized_score,
            formula=fallback_measurement.formula,
            interpretation=fallback_measurement.interpretation,
        ),
        [fallback_measurement.interpretation],
        None,
    )


def _build_grouping_dimension_from_llm(
    llm_result: GroupingCompactnessAssessment,
    opencv_score: float,
) -> DimensionEvaluation:
    metric = MetricEvaluation(
        key="grouping_compactness_separation",
        label="组内紧密与组间分离度",
        method="multimodal_llm",
        raw_value=None,
        unit="score",
        normalized_score=llm_result.grouping_compactness_separation.score,
        formula="多模态模型依据截图中的空间邻近、留白、模块边界与分区关系直接评估。",
        interpretation=llm_result.grouping_compactness_separation.judgment,
    )
    return DimensionEvaluation(
        score=llm_result.grouping_compactness_separation.score,
        opencv_score=opencv_score,
        multimodal_score=llm_result.grouping_compactness_separation.score,
        judgment=llm_result.grouping_compactness_separation.judgment,
        evidence=llm_result.grouping_compactness_separation.evidence,
        suggestion=llm_result.grouping_compactness_separation.suggestion,
        metrics=[metric],
    )


def _compose_judgment(dimension_key: str, score: float, weakest_metric: MetricEvaluation) -> str:
    guidance = DIMENSION_GUIDANCE[dimension_key]
    if score >= 8:
        base = guidance["strong"]
    elif score >= 5.5:
        base = guidance["medium"]
    else:
        base = guidance["weak"]
    return f"{base}当前最明显的短板是{weakest_metric.label}。"


def _build_dimension_evidence(
    metrics: list[MetricEvaluation],
    extra_evidence: list[str] | None = None,
) -> list[str]:
    weakest_two = sorted(metrics, key=lambda item: item.normalized_score)[:2]
    evidence = [metric.interpretation for metric in weakest_two]
    if extra_evidence:
        for item in extra_evidence:
            if item not in evidence:
                evidence.insert(0, item)

    deduped: list[str] = []
    for item in evidence:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _build_dimension(
    dimension_key: str,
    metrics: list[MetricEvaluation],
    score: float,
    *,
    opencv_score: float | None = None,
    multimodal_score: float | None = None,
    extra_evidence: list[str] | None = None,
) -> DimensionEvaluation:
    weakest_metric = min(metrics, key=lambda item: item.normalized_score)
    return DimensionEvaluation(
        score=score,
        opencv_score=opencv_score,
        multimodal_score=multimodal_score,
        judgment=_compose_judgment(dimension_key, score, weakest_metric),
        evidence=_build_dimension_evidence(metrics, extra_evidence=extra_evidence),
        suggestion=METRIC_SUGGESTIONS[weakest_metric.key],
        metrics=metrics,
    )


def _build_hierarchy_summary(result_dimensions: dict[str, DimensionEvaluation], overall_score: float) -> str:
    ranked = sorted(result_dimensions.items(), key=lambda item: item[1].score, reverse=True)
    best_labels = [DIMENSION_LABELS[item[0]] for item in ranked[:2]]
    weakest_key = ranked[-1][0]
    weakest_label = DIMENSION_LABELS[weakest_key]

    if overall_score >= 8:
        level = "整体层次结构较清晰"
    elif overall_score >= 5.5:
        level = "整体层次结构处于中等水平"
    else:
        level = "整体层次结构偏弱"

    return (
        f"{level}。表现相对更好的两个维度是{best_labels[0]}和{best_labels[1]}，"
        f"说明页面已经建立了一部分稳定的主次和结构规则。"
        f"当前最主要的短板是{weakest_label}，这会让{DIMENSION_EFFECTS[weakest_key]}。"
    )


def _build_priority_improvements(result_dimensions: dict[str, DimensionEvaluation]) -> list[str]:
    ranked = sorted(result_dimensions.items(), key=lambda item: item[1].score)
    improvements: list[str] = []
    for _, dimension in ranked:
        if dimension.suggestion not in improvements:
            improvements.append(dimension.suggestion)
        if len(improvements) == 3:
            break
    return improvements


def _build_confidence(analysis: CVAnalysisResult, llm_used: bool) -> str:
    confidence_score = 0.0
    confidence_score += 0.45 if len(analysis.detected_elements) >= 18 else 0.25 if len(analysis.detected_elements) >= 10 else 0.1
    confidence_score += 0.3 if len(analysis.detected_groups) >= 3 else 0.2 if len(analysis.detected_groups) >= 2 else 0.05
    confidence_score += 0.25 if llm_used else 0.12

    if confidence_score >= 0.85:
        return "high"
    if confidence_score >= 0.55:
        return "medium"
    return "low"


def evaluate_ui_hierarchy(image_path: str, settings: Settings) -> tuple[UIHierarchyEvaluation, PipelineArtifacts]:
    analysis = analyze_image_with_opencv(image_path)

    llm_raw_sections: list[str] = []
    llm_error_sections: list[str] = []
    font_llm_result: FontHierarchyAssessment | None = None
    grouping_llm_result: GroupingCompactnessAssessment | None = None
    llm_used = False
    llm_transport = settings.resolved_provider()

    evaluator = None
    try:
        evaluator = build_multimodal_hierarchy_evaluator(settings)
    except Exception as exc:
        llm_error_sections.append(_explain_llm_error(str(exc), settings))

    if evaluator is not None:
        llm_transport = getattr(evaluator, "transport_name", llm_transport)

        try:
            font_raw_text = evaluator.evaluate_font_hierarchy(image_path)
            font_llm_result = parse_font_hierarchy_result(font_raw_text, image_name=_image_name(image_path))
            llm_raw_sections.append(f"=== font_hierarchy_delta ===\n{font_raw_text}")
            llm_used = True
        except Exception as exc:
            llm_error_sections.append(f"font_hierarchy_delta: {_explain_llm_error(str(exc), settings)}")

        try:
            grouping_raw_text = evaluator.evaluate_grouping_compactness(image_path)
            grouping_llm_result = parse_grouping_compactness_result(
                grouping_raw_text,
                image_name=_image_name(image_path),
            )
            llm_raw_sections.append(f"=== grouping_compactness_separation ===\n{grouping_raw_text}")
            llm_used = True
        except Exception as exc:
            llm_error_sections.append(
                f"grouping_compactness_separation: {_explain_llm_error(str(exc), settings)}"
            )

    llm_raw_text = "\n\n".join(llm_raw_sections) if llm_raw_sections else None
    llm_error = "\n".join(llm_error_sections) if llm_error_sections else None

    font_metric, font_evidence, font_multimodal_score = _build_font_metric(
        font_llm_result,
        analysis.estimated_font_hierarchy,
    )

    saliency_metrics = [
        font_metric,
        _to_metric_model(analysis.metrics["visual_weight_delta"]),
        _to_metric_model(analysis.metrics["region_area_delta"]),
        _to_metric_model(analysis.metrics["foreground_background_contrast_delta"]),
    ]
    saliency_score = _weighted_score(
        {metric.key: metric.normalized_score for metric in saliency_metrics},
        SALIENCY_WEIGHTS,
    )
    saliency_opencv_score = _simple_average(
        [metric.normalized_score for metric in saliency_metrics if metric.method == "opencv"]
    )

    grouping_metrics = [
        _to_metric_model(analysis.metrics[key])
        for key in DIMENSION_METRIC_KEYS["grouping_compactness_separation"]
    ]
    grouping_opencv_raw_score = _simple_average([metric.normalized_score for metric in grouping_metrics])
    grouping_opencv_score = _transform_grouping_dimension_score(grouping_opencv_raw_score)

    alignment_metrics = [
        _to_metric_model(analysis.metrics[key])
        for key in DIMENSION_METRIC_KEYS["alignment_consistency"]
    ]
    alignment_score = _simple_average([metric.normalized_score for metric in alignment_metrics])

    result_dimensions = {
        "visual_saliency_difference": _build_dimension(
            "visual_saliency_difference",
            saliency_metrics,
            saliency_score,
            opencv_score=saliency_opencv_score,
            multimodal_score=font_multimodal_score,
            extra_evidence=font_evidence,
        ),
        "grouping_compactness_separation": (
            _build_grouping_dimension_from_llm(grouping_llm_result, grouping_opencv_score)
            if grouping_llm_result is not None
            else _build_dimension(
                "grouping_compactness_separation",
                grouping_metrics,
                grouping_opencv_score,
                opencv_score=grouping_opencv_score,
            )
        ),
        "alignment_consistency": _build_dimension(
            "alignment_consistency",
            alignment_metrics,
            alignment_score,
            opencv_score=alignment_score,
        ),
    }

    overall_score = round(mean(dimension.score for dimension in result_dimensions.values()), 1)

    llm_runtime = (
        f"provider={settings.resolved_provider()}, "
        f"transport={llm_transport}, "
        f"model={settings.model}, "
        f"base_url={settings.base_url}"
    )
    if font_llm_result is not None and grouping_llm_result is not None:
        llm_status = f"Multimodal scoring succeeded for font hierarchy and grouping ({llm_runtime})."
    elif font_llm_result is not None and grouping_llm_result is None:
        llm_status = (
            f"Multimodal font scoring succeeded, but grouping used fallback ({llm_runtime})."
            + (f" {llm_error}" if llm_error else "")
        )
    elif font_llm_result is None and grouping_llm_result is not None:
        llm_status = (
            f"Multimodal grouping scoring succeeded, but font hierarchy used fallback ({llm_runtime})."
            + (f" {llm_error}" if llm_error else "")
        )
    elif not settings.enable_mllm:
        llm_status = "Multimodal scoring is disabled. Font hierarchy and grouping used fallback logic."
    elif not settings.api_key:
        llm_status = f"OPENAI_API_KEY not found ({llm_runtime}). Font hierarchy and grouping used fallback logic."
    elif llm_error:
        llm_status = f"Multimodal scoring failed ({llm_runtime}): {llm_error}"
    else:
        llm_status = f"Multimodal scoring was not used ({llm_runtime}). Font hierarchy and grouping used fallback logic."

    result = UIHierarchyEvaluation(
        task="ui_hierarchy_evaluation",
        image_name=_image_name(image_path),
        overall_score=overall_score,
        confidence=_build_confidence(analysis, llm_used),
        method_summary=(
            "本版本采用 OpenCV 规则特征与多模态模型互补结合的混合路线："
            "面积、对比、对齐等可解释指标由 OpenCV 提取，"
            "字体层级差异与组内紧密及组间分离度优先由多模态模型判断，"
            "在模型不可用时再回退到启发式或 OpenCV 评分。"
        ),
        detection_summary=DetectionSummary(
            image_width=analysis.image_width,
            image_height=analysis.image_height,
            detected_elements=len(analysis.detected_elements),
            detected_groups=len(analysis.detected_groups),
            llm_used=llm_used,
            llm_provider=settings.resolved_provider(),
            llm_transport=llm_transport,
            llm_model=settings.model,
            llm_base_url=settings.base_url,
            llm_status=llm_status,
        ),
        dimensions=HierarchyDimensions(
            visual_saliency_difference=result_dimensions["visual_saliency_difference"],
            grouping_compactness_separation=result_dimensions["grouping_compactness_separation"],
            alignment_consistency=result_dimensions["alignment_consistency"],
        ),
        hierarchy_summary=_build_hierarchy_summary(result_dimensions, overall_score),
        priority_improvements=_build_priority_improvements(result_dimensions),
    )

    return result, PipelineArtifacts(
        overlay_image=analysis.overlay_image,
        llm_raw_text=llm_raw_text,
        llm_error=llm_error,
    )

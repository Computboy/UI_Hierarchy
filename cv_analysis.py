from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from config import LayoutGroupingConfig

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True)
class ElementGroup:
    box: BoundingBox
    member_indices: list[int]
    group_id: str = ""
    column: str = "unknown"
    group_type: str = "unknown"
    confidence: float = 0.5


@dataclass(frozen=True)
class SemanticGroupHint:
    label: str
    x: float
    y: float
    w: float
    h: float
    role: str = "unknown"


@dataclass(frozen=True)
class LayoutColumn:
    column_id: str
    name: str
    box: BoundingBox
    member_indices: list[int]
    confidence: float = 0.5


@dataclass(frozen=True)
class LocalMetricMeasurement:
    key: str
    label: str
    method: str
    raw_value: float | None
    unit: str
    normalized_score: float
    formula: str
    interpretation: str


@dataclass
class CVAnalysisResult:
    image_width: int
    image_height: int
    detected_elements: list[BoundingBox]
    detected_groups: list[ElementGroup]
    detected_columns: list[LayoutColumn]
    grouping_strategy: str
    metrics: dict[str, LocalMetricMeasurement]
    overlay_image: Any
    estimated_font_hierarchy: LocalMetricMeasurement


def _require_cv_stack() -> None:
    if cv2 is None or np is None:
        raise ModuleNotFoundError(
            "缺少 OpenCV 运行依赖，请先执行 `pip install -r requirements.txt` 安装 "
            "`opencv-python-headless` 和 `numpy`。"
        )


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _round_score(value: float) -> float:
    return round(max(1.0, min(10.0, value)), 1)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_higher_better(value: float, worst: float, best: float) -> float:
    if best <= worst:
        return 5.5
    ratio = _clamp01((value - worst) / (best - worst))
    return _round_score(1 + ratio * 9)


def _score_lower_better(value: float, best: float, worst: float) -> float:
    if worst <= best:
        return 5.5
    ratio = _clamp01((worst - value) / (worst - best))
    return _round_score(1 + ratio * 9)


def _robust_center(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    if len(ordered) < 5:
        return float(median(ordered))
    trim = max(1, int(len(ordered) * 0.1))
    trimmed = ordered[trim:-trim] or ordered
    return float(median(trimmed))


def _describe_score(score: float, good: str, medium: str, weak: str) -> str:
    if score >= 8:
        return good
    if score >= 5.5:
        return medium
    return weak


def _load_image_bgr(image_path: str):
    _require_cv_stack()
    image_bytes = np.fromfile(str(Path(image_path)), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像文件：{image_path}")
    return image


def save_image_bgr(image, save_path: str) -> None:
    _require_cv_stack()
    suffix = Path(save_path).suffix.lower() or ".png"
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"无法将图像编码为 {ext}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(Path(save_path)))


def _bbox_distance(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(0, max(a.x - b.right, b.x - a.right))
    dy = max(0, max(a.y - b.bottom, b.y - a.bottom))
    return math.hypot(dx, dy)


def _center_in_box(box: BoundingBox, target: BoundingBox) -> bool:
    return (
        target.x <= box.center_x <= target.right
        and target.y <= box.center_y <= target.bottom
    )


def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.right for box in boxes)
    y2 = max(box.bottom for box in boxes)
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def _expand_box(box: BoundingBox, pad_x: int, pad_y: int, image_shape: tuple[int, int]) -> BoundingBox:
    h, w = image_shape
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w, box.right + pad_x)
    y2 = min(h, box.bottom + pad_y)
    return BoundingBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _box_to_list(box: BoundingBox) -> list[int]:
    return [box.x, box.y, box.right, box.bottom]


def _percentile(values: list[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.percentile(np.array(values, dtype=float), q))


def _intersection_over_union(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def _smaller_box_overlap(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / max(1, min(a.area, b.area))


def _remove_nested_boxes(boxes: list[BoundingBox], overlap_threshold: float = 0.92) -> list[BoundingBox]:
    kept: list[BoundingBox] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        is_nested = False
        for existing in kept:
            x1 = max(box.x, existing.x)
            y1 = max(box.y, existing.y)
            x2 = min(box.right, existing.right)
            y2 = min(box.bottom, existing.bottom)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if box.area and inter / box.area >= overlap_threshold:
                is_nested = True
                break
        if not is_nested:
            kept.append(box)
    return sorted(kept, key=lambda item: (item.y, item.x))


def _merge_close_overlaps(boxes: list[BoundingBox], iou_threshold: float = 0.78) -> list[BoundingBox]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[BoundingBox] = []
        used = [False] * len(merged)
        for index, box in enumerate(merged):
            if used[index]:
                continue
            current = box
            used[index] = True
            for other_index in range(index + 1, len(merged)):
                if used[other_index]:
                    continue
                other = merged[other_index]
                if _intersection_over_union(current, other) >= iou_threshold:
                    current = _union_boxes([current, other])
                    used[other_index] = True
                    changed = True
            next_boxes.append(current)
        merged = next_boxes
    return sorted(merged, key=lambda item: (item.y, item.x))


def _find_boxes_from_mask(mask, image_area: int, min_area_ratio: float, max_area_ratio: float) -> list[BoundingBox]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BoundingBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * min_area_ratio or area > image_area * max_area_ratio:
            continue
        if w < 4 or h < 4:
            continue
        aspect_ratio = w / max(h, 1)
        if aspect_ratio > 35 and h < 12:
            continue
        boxes.append(BoundingBox(x, y, w, h))
    return _merge_close_overlaps(_remove_nested_boxes(boxes))


def _build_foreground_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block_size = _ensure_odd(max(31, min(image.shape[:2]) // 18))
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        12,
    )
    gradient = cv2.morphologyEx(
        blur,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    _, gradient_mask = cv2.threshold(
        gradient,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    edges = cv2.Canny(blur, 50, 150)
    mask = cv2.bitwise_or(adaptive, gradient_mask)
    mask = cv2.bitwise_or(mask, edges)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, clean_kernel)
    return gray, edges, mask


def _detect_element_boxes(mask, image_shape: tuple[int, int]) -> list[BoundingBox]:
    h, w = image_shape
    image_area = h * w

    primary = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 70), max(5, h // 120))),
        iterations=1,
    )
    primary = cv2.morphologyEx(
        primary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, w // 120), max(6, h // 120))),
    )
    boxes = _find_boxes_from_mask(primary, image_area, 0.00008, 0.28)

    if len(boxes) >= 12:
        return boxes

    fallback = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 110), max(3, h // 160))),
        iterations=1,
    )
    fallback_boxes = _find_boxes_from_mask(fallback, image_area, 0.00004, 0.22)
    return _merge_close_overlaps(_remove_nested_boxes(boxes + fallback_boxes))


def _fallback_groups_by_rows(element_boxes: list[BoundingBox], image_height: int) -> list[ElementGroup]:
    if not element_boxes:
        return []

    sorted_pairs = sorted(enumerate(element_boxes), key=lambda item: (item[1].y, item[1].x))
    threshold = max(18, int(image_height * 0.035))
    current_indices: list[int] = [sorted_pairs[0][0]]
    current_bottom = sorted_pairs[0][1].bottom
    groups: list[ElementGroup] = []

    for index, box in sorted_pairs[1:]:
        vertical_gap = max(0, box.y - current_bottom)
        if vertical_gap > threshold:
            member_boxes = [element_boxes[item] for item in current_indices]
            groups.append(ElementGroup(box=_union_boxes(member_boxes), member_indices=current_indices.copy()))
            current_indices = [index]
        else:
            current_indices.append(index)
        current_bottom = max(current_bottom, box.bottom)

    member_boxes = [element_boxes[item] for item in current_indices]
    groups.append(ElementGroup(box=_union_boxes(member_boxes), member_indices=current_indices.copy()))
    return groups


def _split_indices_by_y_gaps(
    indexed_boxes: list[tuple[int, BoundingBox]],
    image_height: int,
) -> list[list[int]]:
    if not indexed_boxes:
        return []

    heights = [box.h for _, box in indexed_boxes]
    gap_threshold = max(24, int(image_height * 0.025), int(_robust_center([float(item) for item in heights], 24.0) * 2.2))
    sorted_pairs = sorted(indexed_boxes, key=lambda item: (item[1].y, item[1].x))
    groups: list[list[int]] = []
    current_indices: list[int] = [sorted_pairs[0][0]]
    current_bottom = sorted_pairs[0][1].bottom

    for index, box in sorted_pairs[1:]:
        vertical_gap = max(0, box.y - current_bottom)
        if vertical_gap > gap_threshold:
            groups.append(current_indices.copy())
            current_indices = [index]
        else:
            current_indices.append(index)
        current_bottom = max(current_bottom, box.bottom)

    groups.append(current_indices.copy())
    return groups


def _fallback_groups_by_columns_and_rows(
    element_boxes: list[BoundingBox],
    image_shape: tuple[int, int],
) -> list[ElementGroup]:
    if not element_boxes:
        return []

    h, w = image_shape
    image_area = h * w
    indexed_boxes = list(enumerate(element_boxes))
    content_items = [
        (index, box)
        for index, box in indexed_boxes
        if box.area < image_area * 0.12 and not (box.w >= w * 0.65 and box.h >= h * 0.22)
    ]
    if len(content_items) < max(4, len(indexed_boxes) // 3):
        content_items = indexed_boxes

    widths = [float(box.w) for _, box in content_items]
    center_threshold = max(56.0, min(w * 0.14, _robust_center(widths, 56.0) * 1.6))
    wide_items = [(index, box) for index, box in content_items if box.w >= w * 0.68]
    regular_items = [(index, box) for index, box in content_items if box.w < w * 0.68]

    columns: list[list[tuple[int, BoundingBox]]] = []
    for item in sorted(regular_items, key=lambda pair: pair[1].center_x):
        if not columns:
            columns.append([item])
            continue
        column_center = _robust_center([box.center_x for _, box in columns[-1]], item[1].center_x)
        if abs(item[1].center_x - column_center) > center_threshold:
            columns.append([item])
        else:
            columns[-1].append(item)

    member_groups: list[list[int]] = []
    member_groups.extend(_split_indices_by_y_gaps(wide_items, h))
    for column in columns:
        member_groups.extend(_split_indices_by_y_gaps(column, h))

    output: list[ElementGroup] = []
    seen: set[tuple[int, ...]] = set()
    for member_indices in member_groups:
        if not member_indices:
            continue
        key = tuple(sorted(member_indices))
        if key in seen:
            continue
        seen.add(key)
        member_boxes = [element_boxes[index] for index in key]
        output.append(ElementGroup(box=_union_boxes(member_boxes), member_indices=list(key)))

    return sorted(output, key=lambda group: (group.box.y, group.box.x))


def _estimate_grouping_score(
    element_boxes: list[BoundingBox],
    groups: list[ElementGroup],
    image_shape: tuple[int, int],
) -> float:
    if not groups:
        return 0.0
    h, w = image_shape
    metrics = _build_grouping_metrics(element_boxes, groups, w, h)
    return mean(metric.normalized_score for metric in metrics.values())


def _semantic_hint_to_box(hint: SemanticGroupHint, image_shape: tuple[int, int]) -> BoundingBox:
    h, w = image_shape
    x1 = int(round(_clamp01(hint.x) * w))
    y1 = int(round(_clamp01(hint.y) * h))
    x2 = int(round(_clamp01(hint.x + hint.w) * w))
    y2 = int(round(_clamp01(hint.y + hint.h) * h))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def _assign_elements_to_semantic_groups(
    element_boxes: list[BoundingBox],
    semantic_groups: list[SemanticGroupHint] | None,
    image_shape: tuple[int, int],
) -> list[ElementGroup]:
    if not semantic_groups:
        return []

    group_boxes = [_semantic_hint_to_box(hint, image_shape) for hint in semantic_groups]
    assigned: dict[int, list[int]] = {index: [] for index in range(len(group_boxes))}

    for element_index, element_box in enumerate(element_boxes):
        best_group_index: int | None = None
        best_score = 0.0
        for group_index, group_box in enumerate(group_boxes):
            center_bonus = 1.0 if _center_in_box(element_box, group_box) else 0.0
            overlap = _smaller_box_overlap(element_box, group_box)
            score = center_bonus + overlap
            if score > best_score:
                best_score = score
                best_group_index = group_index
        if best_group_index is not None and best_score >= 0.18:
            assigned[best_group_index].append(element_index)

    groups: list[ElementGroup] = []
    for group_index, member_indices in assigned.items():
        if not member_indices:
            continue
        hint = semantic_groups[group_index]
        groups.append(
            ElementGroup(
                box=group_boxes[group_index],
                member_indices=member_indices,
                group_id=f"semantic_{group_index + 1:02d}",
                column="semantic",
                group_type=hint.role or "semantic",
                confidence=0.82,
            )
        )

    return sorted(groups, key=lambda group: (group.box.y, group.box.x))


def _make_group(
    element_boxes: list[BoundingBox],
    member_indices: list[int],
    *,
    group_id: str,
    column: str,
    group_type: str,
    confidence: float,
    image_shape: tuple[int, int],
    padding: int = 6,
) -> ElementGroup | None:
    if not member_indices:
        return None
    member_boxes = [element_boxes[index] for index in member_indices]
    box = _expand_box(_union_boxes(member_boxes), padding, padding, image_shape)
    return ElementGroup(
        box=box,
        member_indices=sorted(member_indices),
        group_id=group_id,
        column=column,
        group_type=group_type,
        confidence=max(0.0, min(1.0, confidence)),
    )


def _split_by_x_gaps(
    indexed_boxes: list[tuple[int, BoundingBox]],
    image_width: int,
    *,
    max_splits: int,
    min_gap_ratio: float,
) -> list[list[tuple[int, BoundingBox]]]:
    if not indexed_boxes:
        return []
    if len(indexed_boxes) <= 2 or max_splits <= 0:
        return [sorted(indexed_boxes, key=lambda item: item[1].center_x)]

    sorted_items = sorted(indexed_boxes, key=lambda item: item[1].center_x)
    center_gaps = [
        sorted_items[index + 1][1].center_x - sorted_items[index][1].center_x
        for index in range(len(sorted_items) - 1)
    ]
    positive_gaps = [gap for gap in center_gaps if gap > 0]
    median_gap = _robust_center(positive_gaps, default=image_width * min_gap_ratio)
    threshold = max(image_width * min_gap_ratio, median_gap * 1.75)
    candidate_splits = [
        (index, gap)
        for index, gap in enumerate(center_gaps)
        if gap >= threshold
    ]
    candidate_splits = sorted(candidate_splits, key=lambda item: item[1], reverse=True)[:max_splits]
    split_after = sorted(index for index, _ in candidate_splits)
    if not split_after:
        return [sorted_items]

    clusters: list[list[tuple[int, BoundingBox]]] = []
    start = 0
    for split_index in split_after:
        clusters.append(sorted_items[start : split_index + 1])
        start = split_index + 1
    clusters.append(sorted_items[start:])
    return [cluster for cluster in clusters if cluster]


def _split_by_y_gaps(
    indexed_boxes: list[tuple[int, BoundingBox]],
    image_shape: tuple[int, int],
    config: LayoutGroupingConfig,
    gray=None,
    column_box: BoundingBox | None = None,
) -> list[list[int]]:
    if not indexed_boxes:
        return []

    h, _ = image_shape
    sorted_items = sorted(indexed_boxes, key=lambda item: (item[1].y, item[1].x))
    progressive_gaps: list[float] = []
    current_bottom = sorted_items[0][1].bottom
    for _, box in sorted_items[1:]:
        progressive_gaps.append(float(max(0, box.y - current_bottom)))
        current_bottom = max(current_bottom, box.bottom)

    positive_gaps = [gap for gap in progressive_gaps if gap > 1]
    median_gap = _robust_center(positive_gaps, default=max(12.0, h * 0.012))
    q1 = _percentile(positive_gaps, 25, default=median_gap)
    q3 = _percentile(positive_gaps, 75, default=median_gap)
    iqr = max(0.0, q3 - q1)
    gap_threshold = max(
        14.0,
        median_gap * config.median_gap_multiplier,
        median_gap + config.iqr_gap_multiplier * iqr,
        h * 0.012,
    )

    groups: list[list[int]] = []
    current_indices: list[int] = [sorted_items[0][0]]
    current_box = sorted_items[0][1]
    current_bottom = sorted_items[0][1].bottom
    max_group_height = (column_box.h if column_box else h) * config.max_group_height_ratio

    for index, box in sorted_items[1:]:
        vertical_gap = max(0, box.y - current_bottom)
        separator = has_horizontal_separator_between(current_box, box, gray, column_box, config)
        candidate_box = _union_boxes([current_box, box])
        too_tall = (
            len(current_indices) >= 2
            and candidate_box.h > max_group_height
            and box.y > current_box.y + current_box.h * 0.45
        )
        should_split = vertical_gap > gap_threshold or separator or too_tall
        if should_split:
            groups.append(current_indices.copy())
            current_indices = [index]
            current_box = box
        else:
            current_indices.append(index)
            current_box = candidate_box
        current_bottom = max(current_bottom, box.bottom)

    groups.append(current_indices.copy())
    return groups


def has_horizontal_separator_between(
    box_a: BoundingBox,
    box_b: BoundingBox,
    gray,
    column_box: BoundingBox | None,
    config: LayoutGroupingConfig,
) -> bool:
    if gray is None or box_b.y <= box_a.bottom:
        return False

    gap_top = max(0, box_a.bottom)
    gap_bottom = min(gray.shape[0], box_b.y)
    if gap_bottom - gap_top < 4:
        return False

    x1 = column_box.x if column_box else min(box_a.x, box_b.x)
    x2 = column_box.right if column_box else max(box_a.right, box_b.right)
    x1 = max(0, min(gray.shape[1] - 1, x1))
    x2 = max(x1 + 1, min(gray.shape[1], x2))
    band = gray[gap_top:gap_bottom, x1:x2]
    if band.size == 0:
        return False

    blank_ratio = float((band > 245).mean())
    row_std = band.std(axis=1) if band.shape[1] > 4 else np.array([])
    row_mean = band.mean(axis=1) if band.shape[1] > 4 else np.array([])
    has_line = bool(np.any((row_std < 8) & (row_mean > 170) & (row_mean < 245))) if row_std.size else False
    return blank_ratio >= config.separator_blank_ratio or has_line


def _detect_top_groups(
    element_boxes: list[BoundingBox],
    image_shape: tuple[int, int],
    config: LayoutGroupingConfig,
) -> tuple[list[ElementGroup], set[int], BoundingBox | None]:
    h, w = image_shape
    top_limit = int(h * max(0.08, min(0.24, config.top_region_ratio)))
    top_items = [
        (index, box)
        for index, box in enumerate(element_boxes)
        if box.center_y <= top_limit or (box.y <= top_limit * 0.72 and box.h <= h * 0.16)
    ]
    if not top_items:
        return [], set(), None

    clusters = _split_by_x_gaps(top_items, w, max_splits=2, min_gap_ratio=0.035)
    groups: list[ElementGroup] = []
    used: set[int] = set()
    labels = ["header", "category_tabs", "search"]
    for order, cluster in enumerate(clusters, start=1):
        indices = [index for index, _ in cluster]
        used.update(indices)
        cluster_box = _union_boxes([box for _, box in cluster])
        if cluster_box.right > w * 0.68:
            group_type = "search"
        elif cluster_box.w > w * 0.35:
            group_type = "category_tabs"
        else:
            group_type = labels[min(order - 1, len(labels) - 1)]
        group = _make_group(
            element_boxes,
            indices,
            group_id=f"top_{order:02d}",
            column="top",
            group_type=group_type,
            confidence=0.78,
            image_shape=image_shape,
        )
        if group:
            groups.append(group)

    return groups, used, _union_boxes([group.box for group in groups]) if groups else None


def _detect_layout_columns(
    element_boxes: list[BoundingBox],
    candidate_indices: list[int],
    image_shape: tuple[int, int],
    config: LayoutGroupingConfig,
) -> list[LayoutColumn]:
    h, w = image_shape
    if not candidate_indices:
        return []

    indexed = [(index, element_boxes[index]) for index in candidate_indices]
    # Very wide bands are usually section containers or banners; keep them assignable later
    # but do not let them collapse the x-axis column split.
    split_basis = [
        (index, box)
        for index, box in indexed
        if box.w < w * 0.72 and box.area < h * w * 0.20
    ] or indexed
    clusters = _split_by_x_gaps(
        split_basis,
        w,
        max_splits=max(0, config.max_columns - 1),
        min_gap_ratio=config.min_column_gap_ratio,
    )
    if not clusters:
        clusters = [indexed]

    clusters = sorted(clusters, key=lambda cluster: _robust_center([box.center_x for _, box in cluster], 0.0))
    boundaries: list[float] = []
    for left, right in zip(clusters, clusters[1:]):
        left_center = _robust_center([box.center_x for _, box in left], 0.0)
        right_center = _robust_center([box.center_x for _, box in right], w)
        boundaries.append((left_center + right_center) / 2.0)

    assigned: list[list[int]] = [[] for _ in clusters]
    for index, box in indexed:
        column_index = 0
        while column_index < len(boundaries) and box.center_x > boundaries[column_index]:
            column_index += 1
        assigned[min(column_index, len(assigned) - 1)].append(index)

    names_by_count = {
        1: ["main"],
        2: ["main", "right"],
        3: ["left", "middle", "right"],
    }
    names = names_by_count.get(len(assigned), [f"column_{i + 1}" for i in range(len(assigned))])
    columns: list[LayoutColumn] = []
    for order, indices in enumerate(assigned, start=1):
        if not indices:
            continue
        box = _expand_box(_union_boxes([element_boxes[index] for index in indices]), 10, 10, image_shape)
        columns.append(
            LayoutColumn(
                column_id=f"col_{order:02d}",
                name=names[min(order - 1, len(names) - 1)],
                box=box,
                member_indices=sorted(indices),
                confidence=0.72 if len(columns) > 1 else 0.62,
            )
        )
    return columns


def _infer_group_type(column_name: str, group_box: BoundingBox, member_count: int, image_shape: tuple[int, int]) -> str:
    h, w = image_shape
    if group_box.y < h * 0.18:
        return "header"
    if column_name == "left":
        return "left_nav" if member_count >= 2 else "side_tool_group"
    if column_name == "right":
        return "right_recommend_card" if member_count <= 6 else "ad_or_popup"
    if group_box.bottom > h * 0.88:
        return "footer_or_contact"
    if group_box.w > w * 0.65 and group_box.h < h * 0.16:
        return "category_tabs"
    return "content_card" if member_count >= 1 else "unknown"


def _build_column_groups(
    element_boxes: list[BoundingBox],
    columns: list[LayoutColumn],
    image_shape: tuple[int, int],
    config: LayoutGroupingConfig,
    gray=None,
) -> list[ElementGroup]:
    output: list[ElementGroup] = []
    counters: dict[str, int] = {}
    for column in columns:
        groups = _split_by_y_gaps(
            [(index, element_boxes[index]) for index in column.member_indices],
            image_shape,
            config,
            gray=gray,
            column_box=column.box,
        )
        for member_indices in groups:
            if len(member_indices) < config.min_group_elements:
                continue
            group_box = _union_boxes([element_boxes[index] for index in member_indices])
            group_type = _infer_group_type(column.name, group_box, len(member_indices), image_shape)
            counters[column.name] = counters.get(column.name, 0) + 1
            group = _make_group(
                element_boxes,
                member_indices,
                group_id=f"{column.name}_{counters[column.name]:02d}",
                column=column.name,
                group_type=group_type,
                confidence=0.70,
                image_shape=image_shape,
            )
            if group:
                output.append(group)
    return sorted(output, key=lambda group: (group.box.y, group.box.x))


def _is_overmerged_single_group(groups: list[ElementGroup], element_count: int, image_shape: tuple[int, int]) -> bool:
    if len(groups) != 1 or element_count < 8:
        return False
    h, w = image_shape
    group = groups[0]
    member_ratio = len(group.member_indices) / max(element_count, 1)
    area_ratio = group.box.area / max(h * w, 1)
    return member_ratio >= 0.65 or area_ratio >= 0.45


def _detect_groups(
    element_boxes: list[BoundingBox],
    image_shape: tuple[int, int],
    *,
    gray=None,
    config: LayoutGroupingConfig | None = None,
) -> tuple[list[ElementGroup], list[LayoutColumn], str]:
    if not element_boxes:
        return [], [], "empty"

    config = config or LayoutGroupingConfig()

    # Main path: users perceive desktop pages by global header first, then columns,
    # then repeated cards/modules inside each column. This avoids single-linkage
    # chaining where nearby cards recursively merge into one giant region.
    top_groups, top_indices, top_box = _detect_top_groups(element_boxes, image_shape, config)
    body_indices = [index for index in range(len(element_boxes)) if index not in top_indices]
    columns = _detect_layout_columns(element_boxes, body_indices, image_shape, config)
    column_groups = _build_column_groups(element_boxes, columns, image_shape, config, gray=gray)
    heuristic_groups = sorted(top_groups + column_groups, key=lambda group: (group.box.y, group.box.x))
    if len(heuristic_groups) >= 2:
        strategy = "column_module_split"
        if top_groups:
            strategy += "+top_region"
        return heuristic_groups, columns, strategy

    h, w = image_shape
    image_area = h * w
    group_mask = np.zeros((h, w), dtype=np.uint8)
    for box in element_boxes:
        cv2.rectangle(group_mask, (box.x, box.y), (box.right, box.bottom), 255, thickness=-1)

    merged = cv2.dilate(
        group_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 28), max(14, h // 42))),
        iterations=1,
    )
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, w // 35), max(14, h // 48))),
    )
    group_boxes = _find_boxes_from_mask(merged, image_area, 0.003, 0.85)

    groups: list[ElementGroup] = []
    for group_box in group_boxes:
        member_indices = [
            index for index, element_box in enumerate(element_boxes) if _center_in_box(element_box, group_box)
        ]
        if not member_indices:
            continue
        groups.append(ElementGroup(box=group_box, member_indices=member_indices))

    if len(groups) >= 2:
        fallback_columns = [LayoutColumn("col_01", "fallback", _union_boxes([group.box for group in groups]), list(range(len(element_boxes))), 0.35)]
        return groups, fallback_columns, "morphological_fallback"

    alt_merged = cv2.dilate(
        group_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(14, w // 42), max(10, h // 60))),
        iterations=1,
    )
    alt_merged = cv2.morphologyEx(
        alt_merged,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 60), max(8, h // 70))),
    )
    alt_group_boxes = _find_boxes_from_mask(alt_merged, image_area, 0.0016, 0.5)
    alt_groups: list[ElementGroup] = []
    for group_box in alt_group_boxes:
        member_indices = [
            index for index, element_box in enumerate(element_boxes) if _center_in_box(element_box, group_box)
        ]
        if len(member_indices) >= 2:
            alt_groups.append(ElementGroup(box=group_box, member_indices=member_indices))
    if len(alt_groups) >= 2:
        fallback_columns = [LayoutColumn("col_01", "fallback", _union_boxes([group.box for group in alt_groups]), list(range(len(element_boxes))), 0.35)]
        return alt_groups, fallback_columns, "morphological_alt_fallback"

    row_groups = _fallback_groups_by_rows(element_boxes, h)
    layout_groups = _fallback_groups_by_columns_and_rows(element_boxes, image_shape)
    candidates = [candidate for candidate in (groups, alt_groups, row_groups, layout_groups) if candidate]
    multi_group_candidates = [candidate for candidate in candidates if len(candidate) >= 2]
    if multi_group_candidates and any(_is_overmerged_single_group(candidate, len(element_boxes), image_shape) for candidate in candidates):
        selected = max(multi_group_candidates, key=lambda candidate: _estimate_grouping_score(element_boxes, candidate, image_shape))
    else:
        selected = max(candidates, key=lambda candidate: _estimate_grouping_score(element_boxes, candidate, image_shape))
    fallback_columns = [LayoutColumn("col_01", "fallback", _union_boxes([group.box for group in selected]), list(range(len(element_boxes))), 0.3)]
    return selected, fallback_columns, "legacy_fallback"


def _sample_top_ratio_mean(values: list[float], ratio: float = 0.18) -> float:
    if not values:
        return 0.0
    count = max(1, int(len(values) * ratio))
    return float(mean(sorted(values, reverse=True)[:count]))


def _safe_median(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.median(np.array(values, dtype=float)))


def _compute_element_contrast(gray, box: BoundingBox) -> float:
    inner = gray[box.y : box.bottom, box.x : box.right]
    inner_mean = float(inner.mean()) if inner.size else float(gray.mean())

    margin = max(6, int(min(box.w, box.h) * 0.35))
    x1 = max(0, box.x - margin)
    y1 = max(0, box.y - margin)
    x2 = min(gray.shape[1], box.right + margin)
    y2 = min(gray.shape[0], box.bottom + margin)
    outer = gray[y1:y2, x1:x2]

    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[box.y - y1 : box.bottom - y1, box.x - x1 : box.right - x1] = False
    ring_pixels = outer[ring_mask]
    surround_mean = float(ring_pixels.mean()) if ring_pixels.size else float(gray.mean())
    return abs(inner_mean - surround_mean) / 255.0


def _compute_element_visual_weight(elements: list[BoundingBox], gray, edges) -> list[float]:
    image_area = gray.shape[0] * gray.shape[1]
    weights: list[float] = []
    for box in elements:
        contrast = _compute_element_contrast(gray, box)
        edge_patch = edges[box.y : box.bottom, box.x : box.right]
        edge_density = float(edge_patch.mean()) / 255.0 if edge_patch.size else 0.0
        area_ratio = box.area / max(image_area, 1)
        weight = math.sqrt(max(area_ratio, 1e-6)) * (0.6 * contrast + 0.4 * min(1.0, edge_density * 3.0))
        weights.append(weight)
    return weights


def _cluster_positions(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    clusters: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.median(cluster)) for cluster in clusters if len(cluster) >= 2]


def _nearest_distance(value: float, anchors: list[float]) -> float:
    if not anchors:
        return float("inf")
    return min(abs(value - anchor) for anchor in anchors)


def _build_visual_saliency_metrics(
    elements: list[BoundingBox],
    gray,
    edges,
    image_width: int,
    image_height: int,
) -> tuple[dict[str, LocalMetricMeasurement], LocalMetricMeasurement]:
    image_area = image_width * image_height
    areas = [box.area / max(image_area, 1) for box in elements]
    contrasts = [_compute_element_contrast(gray, box) for box in elements]
    weights = _compute_element_visual_weight(elements, gray, edges)

    area_ratio = _sample_top_ratio_mean(areas) / max(_safe_median(areas, 1e-6), 1e-6)
    weight_ratio = _sample_top_ratio_mean(weights) / max(_safe_median(weights, 1e-6), 1e-6)
    contrast_ratio = _sample_top_ratio_mean(contrasts) / max(_safe_median(contrasts, 1e-6), 1e-6)

    area_score = _score_higher_better(area_ratio, worst=1.15, best=5.0)
    weight_score = _score_higher_better(weight_ratio, worst=1.1, best=4.2)
    contrast_score = _score_higher_better(contrast_ratio, worst=1.03, best=2.5)

    area_metric = LocalMetricMeasurement(
        key="region_area_delta",
        label="区域面积差值",
        method="opencv",
        raw_value=round(area_ratio, 3),
        unit="ratio",
        normalized_score=area_score,
        formula="高面积元素均值 / 全部元素面积中位数",
        interpretation=(
            f"核心区域与普通区域的面积比约为 {area_ratio:.2f}，"
            + _describe_score(area_score, "面积主次对比明显。", "面积层级有一定拉开，但还不够稳定。", "大区域与普通区域差别偏弱。")
        ),
    )
    weight_metric = LocalMetricMeasurement(
        key="visual_weight_delta",
        label="视觉权重差值",
        method="opencv",
        raw_value=round(weight_ratio, 3),
        unit="ratio",
        normalized_score=weight_score,
        formula="高视觉权重元素均值 / 全部元素视觉权重中位数",
        interpretation=(
            f"高视觉权重元素与普通元素的权重比约为 {weight_ratio:.2f}，"
            + _describe_score(weight_score, "页面主焦点较突出。", "主焦点存在，但竞争注意力的元素偏多。", "页面缺少稳定的主焦点。")
        ),
    )
    contrast_metric = LocalMetricMeasurement(
        key="foreground_background_contrast_delta",
        label="前景背景对比差值",
        method="opencv",
        raw_value=round(contrast_ratio, 3),
        unit="ratio",
        normalized_score=contrast_score,
        formula="高显著元素局部对比均值 / 全部元素局部对比中位数",
        interpretation=(
            f"高显著元素与周围背景的对比差值比约为 {contrast_ratio:.2f}，"
            + _describe_score(
                contrast_score,
                "重点区域与背景分离较清楚。",
                "重点区域有一定对比，但仍会与周围内容相互稀释。",
                "重点区域和背景容易混在一起。",
            )
        ),
    )

    text_like_heights = [
        box.h
        for box in elements
        if box.h <= image_height * 0.12 and box.w >= box.h * 1.4 and box.area <= image_area * 0.03
    ]
    if len(text_like_heights) < 6:
        text_like_heights = [box.h for box in elements if box.h <= image_height * 0.16]

    if len(text_like_heights) >= 4:
        top_height = float(np.percentile(text_like_heights, 85))
        mid_height = max(float(np.percentile(text_like_heights, 45)), 1.0)
        text_ratio = top_height / mid_height
    else:
        text_ratio = 1.0

    text_score = _score_higher_better(text_ratio, worst=1.05, best=2.8)
    estimated_font_metric = LocalMetricMeasurement(
        key="font_hierarchy_delta",
        label="字体层级差值",
        method="heuristic_fallback",
        raw_value=round(text_ratio, 3),
        unit="ratio",
        normalized_score=text_score,
        formula="候选文本框高位高度 / 候选文本框中位高度",
        interpretation=(
            f"候选文本框高度比约为 {text_ratio:.2f}，"
            + _describe_score(
                text_score,
                "字号层级差异较明显。",
                "字号层级有一定差异，但标题与正文仍可能接近。",
                "字号层级差异偏弱，文本主次不易快速识别。",
            )
        ),
    )

    return {
        area_metric.key: area_metric,
        weight_metric.key: weight_metric,
        contrast_metric.key: contrast_metric,
    }, estimated_font_metric


def _build_grouping_metrics(
    elements: list[BoundingBox],
    groups: list[ElementGroup],
    image_width: int,
    image_height: int,
) -> dict[str, LocalMetricMeasurement]:
    diag = math.hypot(image_width, image_height)
    within_distances: list[float] = []
    within_edge_distances: list[float] = []
    compactness_values: list[float] = []

    for group in groups:
        member_boxes = [elements[index] for index in group.member_indices]
        if len(member_boxes) >= 2:
            content_box = _union_boxes(member_boxes)
            group_diag = math.hypot(content_box.w, content_box.h)
            for box in member_boxes:
                center_distances = [
                    math.hypot(box.center_x - other.center_x, box.center_y - other.center_y)
                    for other in member_boxes
                    if other is not box
                ]
                edge_distances = [
                    _bbox_distance(box, other) / max(diag, 1.0)
                    for other in member_boxes
                    if other is not box
                ]
                if center_distances:
                    within_distances.append(min(center_distances) / max(group_diag, 1.0))
                if edge_distances:
                    within_edge_distances.append(min(edge_distances))

            center_distances = [
                math.hypot(box.center_x - content_box.center_x, box.center_y - content_box.center_y)
                for box in member_boxes
            ]
            compactness_values.append(
                max(0.0, 1.0 - min(1.0, _robust_center(center_distances) / max(group_diag * 0.6, 1.0)))
            )

    between_distances: list[float] = []
    groups_by_column: dict[str, list[ElementGroup]] = {}
    for group in groups:
        groups_by_column.setdefault(group.column or "unknown", []).append(group)
    for column_groups in groups_by_column.values():
        ordered = sorted(column_groups, key=lambda item: (item.box.y, item.box.x))
        for current, next_group in zip(ordered, ordered[1:]):
            if _smaller_box_overlap(current.box, next_group.box) >= 0.45:
                continue
            between_distances.append(_bbox_distance(current.box, next_group.box) / max(diag, 1.0))

    within_mean = _robust_center(within_distances, default=0.24)
    between_mean = _robust_center(between_distances, default=0.0)
    compactness = _robust_center(compactness_values, default=0.0)
    within_edge = _robust_center(within_edge_distances, default=0.004)
    interval_ratio = between_mean / max(within_edge, 0.004)

    within_score = _score_lower_better(within_mean, best=0.04, worst=0.24)
    between_score = _score_higher_better(between_mean, worst=0.0, best=0.04)
    compactness_score = _score_higher_better(compactness, worst=0.15, best=0.70)
    interval_ratio_score = _score_higher_better(interval_ratio, worst=0.70, best=4.0)

    return {
        "within_group_distance_mean": LocalMetricMeasurement(
            key="within_group_distance_mean",
            label="组内距离稳健值",
            method="opencv",
            raw_value=round(within_mean, 4),
            unit="group_diag_ratio",
            normalized_score=within_score,
            formula="同组元素最近邻中心距离的稳健中位值 / 所在组真实包围框对角线（单元素组不参与）",
            interpretation=(
                f"组内最近邻中心距离稳健值约为所在组对角线的 {within_mean:.3f}，"
                + _describe_score(
                    within_score,
                    "同组元素彼此靠近。",
                    "同组元素基本聚在一起，但仍有局部松散。",
                    "同组元素之间距离偏大，归属关系不够紧。",
                )
            ),
        ),
        "between_group_distance_mean": LocalMetricMeasurement(
            key="between_group_distance_mean",
            label="组间距离稳健值",
            method="opencv",
            raw_value=round(between_mean, 4),
            unit="diag_ratio",
            normalized_score=between_score,
            formula="同一 column 内相邻分组真实边界距离的稳健中位值 / 图像对角线",
            interpretation=(
                f"组间最近边界距离稳健值约为画面对角线的 {between_mean:.3f}，"
                + _describe_score(
                    between_score,
                    "不同分组之间留白较清楚。",
                    "分组之间有分隔，但边界还不够稳定。",
                    "不同分组之间距离偏近，容易互相粘连。",
                )
            ),
        ),
        "spatial_cluster_compactness": LocalMetricMeasurement(
            key="spatial_cluster_compactness",
            label="空间聚类紧凑度",
            method="opencv",
            raw_value=round(compactness, 4),
            unit="0_1",
            normalized_score=compactness_score,
            formula="1 - 组内元素到真实组中心距离的稳健中位值 / 组包围框对角线",
            interpretation=(
                f"空间聚类紧凑度约为 {compactness:.2f}，"
                + _describe_score(
                    compactness_score,
                    "组内聚类较紧凑。",
                    "组内聚类基本可辨，但仍有局部漂移。",
                    "组内元素分布松散，分组形态不稳定。",
                )
            ),
        ),
        "group_interval_ratio": LocalMetricMeasurement(
            key="group_interval_ratio",
            label="分组间隔比",
            method="opencv",
            raw_value=round(interval_ratio, 4),
            unit="ratio",
            normalized_score=interval_ratio_score,
            formula="同栏相邻组间边界距离稳健值 / 组内元素边界距离稳健值",
            interpretation=(
                f"分组间隔比约为 {interval_ratio:.2f}，"
                + _describe_score(
                    interval_ratio_score,
                    "组间远、组内近的关系较明显。",
                    "组内外距离有差别，但还不够拉开。",
                    "组内外距离接近，信息分组不够鲜明。",
                )
            ),
        ),
    }


def _build_alignment_metrics(
    elements: list[BoundingBox],
    image_width: int,
    image_height: int,
) -> dict[str, LocalMetricMeasurement]:
    alignment_boxes = sorted(elements, key=lambda item: item.area, reverse=True)[:120] or elements
    tolerance_x = max(8, int(image_width * 0.012))
    tolerance_y = max(8, int(image_height * 0.012))

    x_edge_axes = _cluster_positions([box.x for box in alignment_boxes] + [box.right for box in alignment_boxes], tolerance_x)
    y_edge_axes = _cluster_positions([box.y for box in alignment_boxes] + [box.bottom for box in alignment_boxes], tolerance_y)
    x_center_axes = _cluster_positions([box.center_x for box in alignment_boxes], tolerance_x)
    y_center_axes = _cluster_positions([box.center_y for box in alignment_boxes], tolerance_y)

    if x_edge_axes:
        edge_x_error = mean(
            min(_nearest_distance(box.x, x_edge_axes), _nearest_distance(box.right, x_edge_axes)) / max(image_width, 1)
            for box in alignment_boxes
        )
    else:
        edge_x_error = 0.12

    if y_edge_axes:
        edge_y_error = mean(
            min(_nearest_distance(box.y, y_edge_axes), _nearest_distance(box.bottom, y_edge_axes)) / max(image_height, 1)
            for box in alignment_boxes
        )
    else:
        edge_y_error = 0.12

    if x_center_axes:
        center_x_error = mean(
            _nearest_distance(box.center_x, x_center_axes) / max(image_width, 1)
            for box in alignment_boxes
        )
    else:
        center_x_error = 0.12

    if y_center_axes:
        center_y_error = mean(
            _nearest_distance(box.center_y, y_center_axes) / max(image_height, 1)
            for box in alignment_boxes
        )
    else:
        center_y_error = 0.12

    edge_alignment_error = float((edge_x_error + edge_y_error) / 2.0)
    center_alignment_error = float((center_x_error + center_y_error) / 2.0)

    collinear_hits = 0
    grid_hits = 0
    for box in alignment_boxes:
        x_ok = (
            _nearest_distance(box.x, x_edge_axes) <= tolerance_x
            or _nearest_distance(box.right, x_edge_axes) <= tolerance_x
            or _nearest_distance(box.center_x, x_center_axes) <= tolerance_x
        )
        y_ok = (
            _nearest_distance(box.y, y_edge_axes) <= tolerance_y
            or _nearest_distance(box.bottom, y_edge_axes) <= tolerance_y
            or _nearest_distance(box.center_y, y_center_axes) <= tolerance_y
        )
        if x_ok or y_ok:
            collinear_hits += 1
        if x_ok and y_ok:
            grid_hits += 1

    collinear_ratio = collinear_hits / max(len(alignment_boxes), 1)
    grid_consistency = grid_hits / max(len(alignment_boxes), 1)

    edge_score = _score_lower_better(edge_alignment_error, best=0.006, worst=0.06)
    center_score = _score_lower_better(center_alignment_error, best=0.006, worst=0.08)
    grid_score = _score_higher_better(grid_consistency, worst=0.15, best=0.9)
    collinear_score = _score_higher_better(collinear_ratio, worst=0.15, best=0.85)

    return {
        "edge_alignment_error": LocalMetricMeasurement(
            key="edge_alignment_error",
            label="边缘对齐误差",
            method="opencv",
            raw_value=round(edge_alignment_error, 4),
            unit="canvas_ratio",
            normalized_score=edge_score,
            formula="元素边缘到最近共享对齐线的平均偏差 / 画布尺寸",
            interpretation=(
                f"边缘对齐误差约为 {edge_alignment_error:.3f}，"
                + _describe_score(
                    edge_score,
                    "左右或上下边缘较稳定地贴合参考线。",
                    "存在主要参考线，但局部仍有偏移。",
                    "边缘偏差较大，版面显得零散。",
                )
            ),
        ),
        "center_axis_alignment_error": LocalMetricMeasurement(
            key="center_axis_alignment_error",
            label="中轴对齐误差",
            method="opencv",
            raw_value=round(center_alignment_error, 4),
            unit="canvas_ratio",
            normalized_score=center_score,
            formula="元素中心点到最近共享中轴的平均偏差 / 画布尺寸",
            interpretation=(
                f"中轴对齐误差约为 {center_alignment_error:.3f}，"
                + _describe_score(
                    center_score,
                    "元素中心轴较统一。",
                    "存在可辨识中轴，但偏移仍然可见。",
                    "中心轴关系较弱，结构不够稳。",
                )
            ),
        ),
        "grid_consistency": LocalMetricMeasurement(
            key="grid_consistency",
            label="栅格一致性",
            method="opencv",
            raw_value=round(grid_consistency, 4),
            unit="ratio",
            normalized_score=grid_score,
            formula="同时落入共享横纵参考线的元素占比",
            interpretation=(
                f"栅格一致性约为 {grid_consistency:.2f}，"
                + _describe_score(
                    grid_score,
                    "版面基本能被统一列网格解释。",
                    "局部存在栅格，但跨模块一致性一般。",
                    "元素难以落回同一套栅格体系。",
                )
            ),
        ),
        "collinear_element_ratio": LocalMetricMeasurement(
            key="collinear_element_ratio",
            label="共线元素占比",
            method="opencv",
            raw_value=round(collinear_ratio, 4),
            unit="ratio",
            normalized_score=collinear_score,
            formula="至少与一条共享边线或中轴对齐的元素占比",
            interpretation=(
                f"共线元素占比约为 {collinear_ratio:.2f}，"
                + _describe_score(
                    collinear_score,
                    "大多数元素都能挂靠到共同参考线。",
                    "部分元素可以共线，但一致性仍不足。",
                    "很多元素脱离共同参考线，结构显得漂移。",
                )
            ),
        ),
    }


def _draw_overlay(image, elements: list[BoundingBox], groups: list[ElementGroup], columns: list[LayoutColumn]):
    overlay = image.copy()
    column_colors = [(255, 245, 210), (225, 245, 255), (235, 255, 225), (245, 230, 255)]
    for index, column in enumerate(columns):
        color = column_colors[index % len(column_colors)]
        column_layer = overlay.copy()
        cv2.rectangle(column_layer, (column.box.x, column.box.y), (column.box.right, column.box.bottom), color, -1)
        overlay = cv2.addWeighted(column_layer, 0.12, overlay, 0.88, 0)
        cv2.putText(
            overlay,
            column.name,
            (column.box.x + 6, max(18, column.box.y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (90, 90, 90),
            2,
            cv2.LINE_AA,
        )

    for group in groups:
        cv2.rectangle(overlay, (group.box.x, group.box.y), (group.box.right, group.box.bottom), (42, 130, 255), 3)
        label = f"{group.group_id or 'group'}:{group.group_type}"
        cv2.putText(
            overlay,
            label[:34],
            (group.box.x + 4, max(16, group.box.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 80, 200),
            2,
            cv2.LINE_AA,
        )
    for box in elements:
        cv2.rectangle(overlay, (box.x, box.y), (box.right, box.bottom), (52, 199, 89), 1)
    cv2.putText(
        overlay,
        f"elements={len(elements)} groups={len(groups)} columns={len(columns)}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"elements={len(elements)} groups={len(groups)} columns={len(columns)}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def analyze_image_with_opencv(
    image_path: str,
    semantic_groups: list[SemanticGroupHint] | None = None,
    grouping_config: LayoutGroupingConfig | None = None,
) -> CVAnalysisResult:
    image = _load_image_bgr(image_path)
    image_height, image_width = image.shape[:2]
    grouping_config = grouping_config or LayoutGroupingConfig()

    gray, edges, mask = _build_foreground_mask(image)
    elements = _detect_element_boxes(mask, (image_height, image_width))
    semantic_detected_groups = _assign_elements_to_semantic_groups(
        elements,
        semantic_groups,
        (image_height, image_width),
    )
    if len(semantic_detected_groups) >= 2:
        groups = semantic_detected_groups
        columns = [
            LayoutColumn(
                column_id=f"semantic_col_{index + 1:02d}",
                name=group.column,
                box=group.box,
                member_indices=group.member_indices,
                confidence=group.confidence,
            )
            for index, group in enumerate(groups)
        ]
        grouping_strategy = "multimodal_semantic_groups"
    else:
        groups, columns, grouping_strategy = _detect_groups(
            elements,
            (image_height, image_width),
            gray=gray,
            config=grouping_config,
        )

    saliency_metrics, font_fallback_metric = _build_visual_saliency_metrics(
        elements,
        gray,
        edges,
        image_width,
        image_height,
    )
    grouping_metrics = _build_grouping_metrics(elements, groups, image_width, image_height)
    alignment_metrics = _build_alignment_metrics(elements, image_width, image_height)

    metrics = {}
    metrics.update(saliency_metrics)
    metrics.update(grouping_metrics)
    metrics.update(alignment_metrics)

    return CVAnalysisResult(
        image_width=image_width,
        image_height=image_height,
        detected_elements=elements,
        detected_groups=groups,
        detected_columns=columns,
        grouping_strategy=grouping_strategy,
        metrics=metrics,
        overlay_image=_draw_overlay(image, elements, groups, columns),
        estimated_font_hierarchy=font_fallback_metric,
    )

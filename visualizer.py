from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from schemas import DIMENSION_LABELS, DIMENSION_ORDER


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False


PAGE_BG = "#f3f6fa"
CARD_BG = "#ffffff"
CARD_BORDER = "#d9e2ec"
CARD_SUB_BORDER = "#cfd9e6"
TITLE_COLOR = "#243b53"
TEXT_COLOR = "#334e68"
MUTED_COLOR = "#627d98"
ACCENT_COLOR = "#c17c17"


def save_json_result(result, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(result.model_dump(mode="json"), file, ensure_ascii=False, indent=2)


def get_dimension_items(result):
    return [(key, getattr(result.dimensions, key)) for key in DIMENSION_ORDER]


def format_score(score: float) -> str:
    return str(int(score)) if int(score) == score else f"{score:.1f}"


def wrap_text_lines(text: str, width: int) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def wrap_labeled_text(label: str, text: str, width: int) -> List[str]:
    prefix = f"{label}: "
    return textwrap.wrap(
        f"{prefix}{text}",
        width=width,
        initial_indent="",
        subsequent_indent=" " * len(prefix),
    ) or [prefix]


def wrap_bullet_text(text: str, width: int) -> List[str]:
    return textwrap.wrap(
        str(text),
        width=width,
        initial_indent="- ",
        subsequent_indent="  ",
    ) or ["- "]


def build_metric_lines(metrics, width: int) -> List[str]:
    lines: List[str] = []
    for metric in metrics:
        method = {
            "opencv": "OpenCV",
            "multimodal_llm": "MLLM",
            "heuristic_fallback": "Fallback",
        }.get(metric.method, metric.method)
        lines.extend(
            wrap_bullet_text(
                f"{metric.label} [{method}] {format_score(metric.normalized_score)}/10",
                width,
            )
        )
    return lines


def build_dimension_detail_lines(dim, width: int) -> List[str]:
    lines: List[str] = []
    lines.extend(wrap_labeled_text("判断", dim.judgment, width))
    lines.extend(wrap_labeled_text("子指标", "", width))
    lines.extend(build_metric_lines(dim.metrics, width))
    lines.extend(wrap_labeled_text("证据", "", width))
    for evidence in dim.evidence:
        lines.extend(wrap_bullet_text(evidence, width))
    lines.extend(wrap_labeled_text("建议", dim.suggestion, width))
    return lines


def make_round_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = CARD_BG,
    edgecolor: str = CARD_BORDER,
    linewidth: float = 1.2,
    rounding: float = 0.02,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    return patch


def plot_bar_chart(result, save_path: str) -> None:
    dim_items = get_dimension_items(result)
    labels = [DIMENSION_LABELS.get(key, key) for key, _ in dim_items]
    scores = [dim.score for _, dim in dim_items]

    plt.figure(figsize=(9, 5), facecolor="white")
    bars = plt.bar(labels, scores, color="#8eb8d1", edgecolor="#537895", linewidth=1.2)
    plt.ylim(0, 10)
    plt.ylabel("Score (0-10)")
    plt.title(f"UI Hierarchy Evaluation - {result.image_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.15,
            format_score(score),
            ha="center",
            va="bottom",
            fontsize=10,
            color=TITLE_COLOR,
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_summary_card(result, original_image_path: str, save_path: str) -> None:
    image = Image.open(original_image_path).convert("RGB")

    overview_lines: List[str] = []
    overview_lines.extend(wrap_labeled_text("Image", result.image_name, 54))
    overview_lines.extend(wrap_labeled_text("Overall", f"{format_score(result.overall_score)} / 10", 54))
    overview_lines.extend(wrap_labeled_text("Confidence", result.confidence, 54))
    overview_lines.extend(
        wrap_labeled_text(
            "Detection",
            (
                f"{result.detection_summary.detected_elements} elements / "
                f"{result.detection_summary.detected_groups} groups / "
                f"{result.detection_summary.detected_columns} columns"
            ),
            54,
        )
    )
    overview_lines.extend(wrap_labeled_text("Grouping", result.detection_summary.grouping_strategy, 54))
    overview_lines.extend(wrap_labeled_text("LLM", result.detection_summary.llm_status, 54))
    if result.detection_summary.llm_provider:
        overview_lines.extend(wrap_labeled_text("LLM Provider", result.detection_summary.llm_provider, 54))
    if result.detection_summary.llm_transport:
        overview_lines.extend(wrap_labeled_text("LLM Transport", result.detection_summary.llm_transport, 54))
    if result.detection_summary.llm_model:
        overview_lines.extend(wrap_labeled_text("LLM Model", result.detection_summary.llm_model, 54))
    if result.detection_summary.llm_base_url:
        overview_lines.extend(wrap_labeled_text("LLM Base URL", result.detection_summary.llm_base_url, 54))
    overview_lines.extend(wrap_labeled_text("Method", result.method_summary, 54))

    dimension_blocks = []
    for key, dim in get_dimension_items(result):
        dimension_blocks.append(
            {
                "label": DIMENSION_LABELS.get(key, key),
                "score": format_score(dim.score),
                "lines": build_dimension_detail_lines(dim, width=48),
            }
        )

    summary_lines = wrap_text_lines(result.hierarchy_summary, width=54)
    priority_lines: List[str] = []
    for item in result.priority_improvements:
        priority_lines.extend(wrap_bullet_text(item, width=54))

    section_title_height = 0.04
    section_line_height = 0.028
    section_header_gap = 0.038
    section_top_pad = 0.02
    section_bottom_pad = 0.02
    section_gap = 0.026

    dimension_title_height = 0.026
    dimension_line_height = 0.022
    dimension_top_pad = 0.016
    dimension_bottom_pad = 0.016
    dimension_gap = 0.012
    dimension_header_gap = 0.042

    overview_height = (
        section_top_pad
        + section_title_height
        + section_header_gap
        + len(overview_lines) * section_line_height
        + section_bottom_pad
    )

    dimension_heights = []
    for block in dimension_blocks:
        dimension_heights.append(
            dimension_top_pad
            + dimension_title_height
            + dimension_header_gap
            + len(block["lines"]) * dimension_line_height
            + dimension_bottom_pad
        )

    dimensions_height = (
        section_top_pad
        + section_title_height
        + 0.03
        + sum(dimension_heights)
        + dimension_gap * max(0, len(dimension_heights) - 1)
        + section_bottom_pad
    )

    summary_height = (
        section_top_pad
        + section_title_height
        + section_header_gap
        + len(summary_lines) * section_line_height
        + section_bottom_pad
    )
    priority_height = (
        section_top_pad
        + section_title_height
        + section_header_gap
        + len(priority_lines) * section_line_height
        + section_bottom_pad
    )

    panel_height = (
        0.02
        + overview_height
        + section_gap
        + dimensions_height
        + section_gap
        + summary_height
        + section_gap
        + priority_height
        + 0.02
    )

    fig = plt.figure(figsize=(16, max(11.5, 8.0 * panel_height)), facecolor=PAGE_BG)
    fig.text(
        0.5,
        0.975,
        "UI Hierarchy Hybrid Evaluation Report",
        ha="center",
        va="top",
        fontsize=26,
        fontweight="bold",
        color=TITLE_COLOR,
    )

    ax_img = fig.add_axes([0.04, 0.12, 0.43, 0.78])
    ax_panel = fig.add_axes([0.49, 0.05, 0.47, 0.88])

    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(
        "Input UI Screenshot",
        fontsize=18,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=12,
    )

    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, panel_height)
    ax_panel.axis("off")

    def draw_section_card(title: str, lines: List[str], y_top: float, height: float) -> float:
        y_bottom = y_top - height
        make_round_box(ax_panel, 0.02, y_bottom, 0.96, height)
        ax_panel.text(
            0.04,
            y_top - 0.035,
            title,
            fontsize=16,
            fontweight="bold",
            color=TITLE_COLOR,
            ha="left",
            va="top",
        )
        ax_panel.text(
            0.04,
            y_top - 0.075,
            "\n".join(lines),
            fontsize=11.2,
            color=TEXT_COLOR,
            ha="left",
            va="top",
            linespacing=1.45,
        )
        return y_bottom

    current_y = panel_height - 0.02
    current_y = draw_section_card("Overview", overview_lines, current_y, overview_height) - section_gap

    dimensions_top = current_y
    dimensions_bottom = dimensions_top - dimensions_height
    make_round_box(ax_panel, 0.02, dimensions_bottom, 0.96, dimensions_height)
    ax_panel.text(
        0.04,
        dimensions_top - 0.035,
        "Three Dimension Scores",
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
        ha="left",
        va="top",
    )

    dimension_y = dimensions_top - 0.085
    for block, block_height in zip(dimension_blocks, dimension_heights):
        block_bottom = dimension_y - block_height
        make_round_box(
            ax_panel,
            0.05,
            block_bottom,
            0.90,
            block_height,
            edgecolor=CARD_SUB_BORDER,
            rounding=0.016,
        )
        ax_panel.text(
            0.07,
            dimension_y - 0.02,
            block["label"],
            fontsize=13.5,
            fontweight="bold",
            color=TITLE_COLOR,
            ha="left",
            va="top",
        )
        ax_panel.text(
            0.93,
            dimension_y - 0.02,
            block["score"],
            fontsize=17,
            fontweight="bold",
            color=ACCENT_COLOR,
            ha="right",
            va="top",
        )
        ax_panel.text(
            0.07,
            dimension_y - 0.06,
            "\n".join(block["lines"]),
            fontsize=10.2,
            color=TEXT_COLOR,
            ha="left",
            va="top",
            linespacing=1.4,
        )
        dimension_y = block_bottom - dimension_gap

    current_y = dimensions_bottom - section_gap
    current_y = draw_section_card("Hierarchy Summary", summary_lines, current_y, summary_height) - section_gap
    draw_section_card("Priority Improvements", priority_lines, current_y, priority_height)

    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor=PAGE_BG)
    plt.close()

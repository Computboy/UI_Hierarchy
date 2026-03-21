from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False


DIMENSION_LABELS = {
    "visual_saliency_difference": "Saliency",
    "group_compactness_and_separation": "Grouping",
    "alignment_consistency": "Alignment",
    "reading_flow_continuity": "Reading Flow",
    "visual_noise": "Low Noise",
}


def save_json_result(result, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)


def plot_bar_chart(result, save_path: str) -> None:
    dims = result.dimensions
    labels = [DIMENSION_LABELS[k] for k in dims.keys()]
    scores = [v.score for v in dims.values()]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores)
    plt.ylim(0, 10)
    plt.ylabel("Score (0-10)")
    plt.title(f"UI Hierarchy Evaluation - {result.image_name}")
    plt.xticks(rotation=15)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def wrap_text(text: str, width: int = 28) -> str:
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width))


def wrap_list(items: List[str], width: int = 26, max_items: int = 3) -> str:
    if not items:
        return "- None"
    out = []
    for item in items[:max_items]:
        wrapped = textwrap.wrap(str(item), width=width)
        if not wrapped:
            continue
        out.append(f"• {wrapped[0]}")
        for line in wrapped[1:]:
            out.append(f"  {line}")
    return "\n".join(out)


def create_summary_card(result, original_image_path: str, save_path: str) -> None:
    """
    自动计算区块高度，避免文字溢出导致错位。
    """
    from pathlib import Path
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    img = Image.open(original_image_path).convert("RGB")

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.06)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])

    # ===== 左侧图片 =====
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Input UI Screenshot", fontsize=16, pad=12)

    # ===== 右侧面板 =====
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.axis("off")

    def wrap_text(text: str, width: int = 34) -> list[str]:
        if not text:
            return [""]
        lines = []
        for paragraph in str(text).split("\n"):
            wrapped = textwrap.wrap(paragraph, width=width) or [""]
            lines.extend(wrapped)
        return lines

    def wrap_bullets(items, width: int = 32, max_items: int = 3) -> list[str]:
        if not items:
            return ["• None"]
        lines = []
        for item in items[:max_items]:
            wrapped = textwrap.wrap(str(item), width=width) or [""]
            if wrapped:
                lines.append("• " + wrapped[0])
                for line in wrapped[1:]:
                    lines.append("  " + line)
        return lines

    def block_height(num_lines: int, title_lines: int = 1) -> float:
        """
        根据文本行数自动给高度。
        """
        top_padding = 0.025
        bottom_padding = 0.02
        title_h = 0.045 * title_lines
        line_h = 0.032
        return top_padding + title_h + num_lines * line_h + bottom_padding

    def draw_block(y_top: float, title: str, content_lines: list[str],
                   title_size: int = 14, content_size: int = 11) -> float:
        """
        在 y_top 位置画一个自动高度区块，返回新区块底部 y 值。
        """
        h = block_height(len(content_lines))
        y_bottom = y_top - h

        rect = Rectangle(
            (0.02, y_bottom), 0.96, h,
            fill=False, linewidth=1.2, edgecolor="black"
        )
        ax_panel.add_patch(rect)

        ax_panel.text(
            0.04, y_top - 0.03, title,
            fontsize=title_size, fontweight="bold",
            va="top", ha="left"
        )

        ax_panel.text(
            0.04, y_top - 0.075,
            "\n".join(content_lines),
            fontsize=content_size,
            va="top", ha="left",
            linespacing=1.45
        )

        return y_bottom

    # ===== 内容准备 =====
    overview_lines = [
        f"Image: {result.image_name}",
        f"Overall Score: {result.overall.score}/10",
    ]

    dim_lines = []
    for key, dim in result.dimensions.items():
        label = DIMENSION_LABELS.get(key, key)
        dim_lines.append(f"{label}: {dim.score}/10")

    summary_lines = wrap_text(result.overall.summary, width=34)
    strengths_lines = wrap_bullets(result.overall.strengths, width=32, max_items=2)
    weaknesses_lines = wrap_bullets(result.overall.weaknesses, width=32, max_items=2)
    suggestions_lines = wrap_bullets(result.overall.suggestions, width=32, max_items=2)

    # ===== 自上而下排版 =====
    y = 0.98
    gap = 0.015

    y = draw_block(y, "Overview", overview_lines, title_size=15, content_size=12) - gap
    y = draw_block(y, "Dimension Scores", dim_lines, title_size=14, content_size=12) - gap
    y = draw_block(y, "Summary", summary_lines, title_size=14, content_size=11) - gap
    y = draw_block(y, "Strengths", strengths_lines, title_size=14, content_size=11) - gap
    y = draw_block(y, "Weaknesses", weaknesses_lines, title_size=14, content_size=11) - gap
    y = draw_block(y, "Suggestions", suggestions_lines, title_size=14, content_size=11) - gap
    
    if y < 0:
        print(f"[WARN] Summary panel overflow detected: y={y:.3f}")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
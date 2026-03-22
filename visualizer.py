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
    修复：当内容过多时，动态缩放行间距，确保所有区块完整显示。
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

    # ----- 辅助函数：文字换行 -----
    def wrap_text_block(text: str, width: int = 34) -> List[str]:
        if not text:
            return [""]
        lines = []
        for paragraph in str(text).split("\n"):
            wrapped = textwrap.wrap(paragraph, width=width) or [""]
            lines.extend(wrapped)
        return lines

    def wrap_bullets_block(items, width: int = 32, max_items: int = 3) -> List[str]:
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

    # ----- 准备所有区块内容 -----
    # 区块定义：(标题, 内容行列表)
    blocks_raw = []

    overview_lines = [
        f"Image: {result.image_name}",
        f"Overall Score: {result.overall.score}/10",
    ]
    blocks_raw.append(("Overview", overview_lines))

    dim_lines = []
    for key, dim in result.dimensions.items():
        label = DIMENSION_LABELS.get(key, key)
        dim_lines.append(f"{label}: {dim.score}/10")
    blocks_raw.append(("Dimension Scores", dim_lines))

    summary_lines = wrap_text_block(result.overall.summary, width=34)
    blocks_raw.append(("Summary", summary_lines))

    strengths_lines = wrap_bullets_block(result.overall.strengths, width=32, max_items=2)
    blocks_raw.append(("Strengths", strengths_lines))

    weaknesses_lines = wrap_bullets_block(result.overall.weaknesses, width=32, max_items=2)
    blocks_raw.append(("Weaknesses", weaknesses_lines))

    suggestions_lines = wrap_bullets_block(result.overall.suggestions, width=32, max_items=2)
    blocks_raw.append(("Suggestions", suggestions_lines))

    # ----- 布局参数（归一化坐标）-----
    # 基础参数
    base_title_height = 0.045      # 标题占用高度
    base_line_height = 0.032       # 每行正文高度
    base_title_offset = 0.03       # 标题相对于区块顶部的偏移
    base_text_offset = 0.075       # 第一行正文相对于区块顶部的偏移
    block_padding_top = 0.015      # 区块内顶部留白
    block_padding_bottom = 0.01    # 区块内底部留白
    gap_between_blocks = 0.012     # 区块间间隙
    top_margin = 0.02              # 面板顶部留白
    bottom_margin = 0.02           # 面板底部留白

    # 计算每个区块的理想高度（归一化）
    block_heights = []
    for title, content_lines in blocks_raw:
        num_lines = len(content_lines)
        # 总高度 = 顶部留白 + 标题高度 + 正文行数*行高 + 底部留白
        h = (block_padding_top +
             base_title_height +
             num_lines * base_line_height +
             block_padding_bottom)
        block_heights.append(h)

    total_ideal = sum(block_heights) + (len(blocks_raw) - 1) * gap_between_blocks
    total_available = 1.0 - top_margin - bottom_margin

    # 如果理想高度超出可用区域，按比例压缩行高和标题高度
    scale = 1.0
    if total_ideal > total_available:
        scale = total_available / total_ideal
        # 压缩后的参数
        title_height = base_title_height * scale
        line_height = base_line_height * scale
        title_offset = base_title_offset * scale
        text_offset = base_text_offset * scale
        pad_top = block_padding_top * scale
        pad_bottom = block_padding_bottom * scale
        gap = gap_between_blocks * scale
    else:
        title_height = base_title_height
        line_height = base_line_height
        title_offset = base_title_offset
        text_offset = base_text_offset
        pad_top = block_padding_top
        pad_bottom = block_padding_bottom
        gap = gap_between_blocks

    # 重新计算实际区块高度（基于压缩后的参数）
    actual_heights = []
    for title, content_lines in blocks_raw:
        num_lines = len(content_lines)
        h = pad_top + title_height + num_lines * line_height + pad_bottom
        actual_heights.append(h)

    # 计算每个区块的 y_top 和 y_bottom（归一化坐标，自上而下）
    y_blocks = []  # 元素为 (y_top, y_bottom, title, content)
    current_y = 1.0 - top_margin
    for idx, ((title, content), h) in enumerate(zip(blocks_raw, actual_heights)):
        y_top = current_y
        y_bottom = y_top - h
        y_blocks.append((y_top, y_bottom, title, content))
        current_y = y_bottom - gap

    # 如果最底部超出面板，整体向上微调（安全兜底）
    min_bottom = min(yb for _, yb, _, _ in y_blocks)
    if min_bottom < bottom_margin:
        shift = bottom_margin - min_bottom
        y_blocks = [(yt + shift, yb + shift, t, c) for yt, yb, t, c in y_blocks]

    # ----- 绘制所有区块 -----
    for y_top, y_bottom, title, content_lines in y_blocks:
        h = y_top - y_bottom
        # 绘制边框矩形
        rect = Rectangle(
            (0.02, y_bottom), 0.96, h,
            fill=False, linewidth=1.2, edgecolor="black"
        )
        ax_panel.add_patch(rect)

        # 绘制标题
        ax_panel.text(
            0.04, y_top - title_offset, title,
            fontsize=14, fontweight="bold",
            va="top", ha="left"
        )

        # 绘制正文
        if content_lines:
            ax_panel.text(
                0.04, y_top - text_offset,
                "\n".join(content_lines),
                fontsize=11,
                va="top", ha="left",
                linespacing=1.45
            )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
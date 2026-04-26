import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# 设置中文字体，避免中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def extract_display_name(image_name: str) -> str:
    """
    从 image_name 中提取用于展示的平台名称。
    例如：
    '01_CSDN.png' -> 'CSDN'
    '02_知乎.png' -> '知乎'
    """
    base = Path(image_name).stem
    parts = base.split("_")

    if len(parts) >= 2:
        return "_".join(parts[1:])

    return parts[0]


def load_batch_summary(json_file_path: Path):
    """
    读取 batch_summary.json。
    """
    try:
        with json_file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误：文件未找到：{json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"错误：文件不是有效 JSON 格式：{json_file_path}")

    if not isinstance(data, list):
        raise ValueError("错误：batch_summary.json 顶层结构应为 list。")

    return data


def plot_scores(
    json_file_path: str | Path,
    output_image: str | Path = "web_platform_scores.png",
    top_n: int | None = None,
    dpi: int = 300,
    show: bool = False,
):
    """
    读取 batch_summary.json，生成横向整体评分条形图。

    参数:
        json_file_path: batch_summary.json 路径
        output_image: 输出图片路径或文件名
        top_n: 仅展示前 N 个样本；None 表示全部展示
        dpi: 图片分辨率
        show: 是否弹出显示图片窗口
    """
    json_file_path = Path(json_file_path)
    data = load_batch_summary(json_file_path)

    platforms = []
    scores = []
    confidences = []

    for item in data:
        if "image_name" not in item or "overall_score" not in item:
            continue

        platforms.append(extract_display_name(item["image_name"]))
        scores.append(float(item["overall_score"]))
        confidences.append(item.get("confidence", "unknown"))

    if not platforms:
        raise ValueError("错误：未从 batch_summary.json 中读取到有效评分数据。")

    sorted_indices = np.argsort(scores)[::-1]

    platforms_sorted = [platforms[i] for i in sorted_indices]
    scores_sorted = [scores[i] for i in sorted_indices]
    confidences_sorted = [confidences[i] for i in sorted_indices]

    if top_n is not None and top_n > 0:
        platforms_sorted = platforms_sorted[:top_n]
        scores_sorted = scores_sorted[:top_n]
        confidences_sorted = confidences_sorted[:top_n]

    color_map = {
        "high": "#68b2eb",
        "medium": "#ff7f0e",
        "low": "#d62728",
        "unknown": "#8E9AAF",
    }

    bar_colors = [color_map.get(conf, color_map["unknown"]) for conf in confidences_sorted]

    # 根据样本数量动态调整高度
    fig_h = max(8, len(platforms_sorted) * 0.38 + 2.5)
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor="white")

    y = np.arange(len(platforms_sorted))

    bars = ax.barh(
        y,
        scores_sorted,
        color=bar_colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.45,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(platforms_sorted, fontsize=9)
    ax.invert_yaxis()

    # 数值标签
    for bar, score in zip(bars, scores_sorted):
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1F2937",
        )

    avg_score = float(np.mean(scores_sorted))
    ax.axvline(
        x=avg_score,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"平均分: {avg_score:.2f}",
    )

    legend_elements = [
        Patch(facecolor=color_map["high"], edgecolor="black", label="高置信度"),
        Patch(facecolor=color_map["medium"], edgecolor="black", label="中置信度"),
    ]

    if any(conf == "low" for conf in confidences_sorted):
        legend_elements.append(
            Patch(facecolor=color_map["low"], edgecolor="black", label="低置信度")
        )

    if any(conf not in {"high", "medium", "low"} for conf in confidences_sorted):
        legend_elements.append(
            Patch(facecolor=color_map["unknown"], edgecolor="black", label="未知置信度")
        )

    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=False)

    ax.set_title("各平台 UI/UX 整体评分对比", fontsize=17, fontweight="bold", pad=18)
    ax.set_xlabel("整体评分（overall_score）", fontsize=12, labelpad=10)
    ax.set_ylabel("平台名称", fontsize=12, labelpad=10)

    ax.set_xlim(0, 10.8)
    ax.set_xticks(np.arange(0, 10.5, 0.5))
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.text(
        0.99,
        0.01,
        "数据来源：批量评估结果（batch_summary.json）",
        ha="right",
        va="bottom",
        fontsize=8,
        style="italic",
        alpha=0.6,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 1])

    output_path = Path(output_image)

    # 如果 output_image 只是文件名，则默认保存到 JSON 所在目录
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = json_file_path.parent / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"图表已保存为: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取 batch_summary.json，生成整体评分横向条形图。"
    )

    parser.add_argument(
        "--batch-dir",
        required=True,
        help="批次输出目录，例如 outputs/batch_20260426_203835",
    )

    parser.add_argument(
        "--output",
        default="web_platform_scores.png",
        help="输出图片文件名或完整路径。若只写文件名，则默认保存到 batch 目录下。",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="仅展示评分最高的前 N 个样本。默认展示全部样本。",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="导出图片分辨率，默认 300。",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="生成后弹出显示图片窗口。默认不弹出。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    batch_dir = Path(args.batch_dir)
    json_file = batch_dir / "batch_summary.json"

    if not batch_dir.exists():
        raise FileNotFoundError(f"批次目录不存在：{batch_dir}")

    if not json_file.exists():
        raise FileNotFoundError(f"未找到 batch_summary.json：{json_file}")

    output_path = Path(args.output)

    # 如果只传文件名，就自动保存到 batch 文件夹下
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = batch_dir / output_path

    plot_scores(
        json_file_path=json_file,
        output_image=output_path,
        top_n=args.top_n,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
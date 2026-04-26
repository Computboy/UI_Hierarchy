import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DIR_PATH = 'outputs/batch_20260412_142235'

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DIMENSIONS = [
    ('visual_saliency_difference', '视觉显著性差异'),
    ('grouping_compactness_separation', '组内紧密与分离'),
    ('alignment_consistency', '对齐一致性'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='读取批次目录下的 result.json，生成更清晰的二维评分图。'
    )
    parser.add_argument('--batch-dir', default=DIR_PATH, help='包含多个 result.json 子目录的批次目录。')
    parser.add_argument('--output', default=None, help='输出图片路径，默认保存到批次目录下的 dimension_scores_clean.png。')
    parser.add_argument('--sort-by', choices=['overall', 'visual', 'grouping', 'alignment'], default='overall', help='样本排序方式。')
    parser.add_argument('--top-n', type=int, default=30, help='仅展示前 N 个样本，默认 30。')
    parser.add_argument('--style', choices=['heatmap', 'barh'], default='heatmap', help='图表样式。')
    parser.add_argument('--dpi', type=int, default=300, help='导出图片分辨率。')
    return parser.parse_args()


def clean_platform_name(image_name: str) -> str:
    stem = Path(image_name).stem
    return re.sub(r'^\d+_', '', stem)


def load_results(batch_dir: Path):
    records = []
    for result_file in sorted(batch_dir.glob('*/result.json')):
        with result_file.open('r', encoding='utf-8') as f:
            data = json.load(f)

        dimensions = data['dimensions']
        records.append(
            {
                'platform': clean_platform_name(data['image_name']),
                'overall': float(data.get('overall_score', 0)),
                'visual': float(dimensions['visual_saliency_difference']['score']),
                'grouping': float(dimensions['grouping_compactness_separation']['score']),
                'alignment': float(dimensions['alignment_consistency']['score']),
            }
        )

    if not records:
        raise FileNotFoundError(f'未在 {batch_dir} 下找到任何 result.json。')

    return records


def sort_records(records, sort_key):
    return sorted(records, key=lambda item: item[sort_key], reverse=True)


def build_score_matrix(records):
    return np.array(
        [[item['visual'], item['grouping'], item['alignment']] for item in records],
        dtype=float,
    )


def add_summary_box(fig, records, score_matrix, batch_dir: Path):
    avg_scores = score_matrix.mean(axis=0)
    summary_lines = [
        f'批次目录：{batch_dir.name}',
        f'样本数量：{len(records)}',
        f'视觉显著性差异均值：{avg_scores[0]:.2f}',
        f'组内紧密与分离均值：{avg_scores[1]:.2f}',
        f'对齐一致性均值：{avg_scores[2]:.2f}',
    ]
    fig.text(
        0.02,
        0.965,
        '\n'.join(summary_lines),
        ha='left',
        va='top',
        fontsize=10.5,
        color='#344054',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#F8FAFC',
            'edgecolor': '#D0D5DD',
            'linewidth': 1.0,
        },
    )

def to_percentile_matrix(raw_matrix):
    percentile_matrix = np.zeros_like(raw_matrix, dtype=float)

    for j in range(raw_matrix.shape[1]):
        values = raw_matrix[:, j]
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=float)

        # rank 从 1 到 n
        ranks[order] = np.arange(1, len(values) + 1)

        percentile_matrix[:, j] = ranks / len(values) * 100

    return percentile_matrix

def plot_heatmap(records, batch_dir: Path, output_path: Path, dpi: int):
    raw_matrix = build_score_matrix(records)
    labels = [item['platform'] for item in records]
    dim_labels = [label for _, label in DIMENSIONS]

    percentile_matrix = to_percentile_matrix(raw_matrix)

    fig_h = max(8, len(records) * 0.34 + 2.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), facecolor='white')

    im = ax.imshow(
        percentile_matrix,
        aspect='auto',
        vmin=0,
        vmax=100,
        cmap='YlGnBu'
    )

    ax.set_xticks(np.arange(len(dim_labels)))
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, len(dim_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(percentile_matrix.shape[0]):
        for j in range(percentile_matrix.shape[1]):
            value = percentile_matrix[i, j]
            text_color = 'white' if value >= 60 else '#243B53'

            ax.text(
                j,
                i,
                f'{value:.0f}',
                ha='center',
                va='center',
                fontsize=8.5,
                color=text_color
            )

    ax.set_title('界面层次结构三维度样本内百分位热力图', fontsize=18, fontweight='bold', pad=18)
    ax.set_xlabel('评分维度', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('界面样本', fontsize=12, fontweight='bold', labelpad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('样本内百分位', fontsize=11)

    add_summary_box(fig, records, raw_matrix, batch_dir)

    fig.text(
        0.50,
        0.025,
        '注：图中数值表示页面在对应维度中的样本内百分位。数值越高，表示该页面在该维度上相较同批次样本表现越突出。',
        ha='center',
        va='bottom',
        fontsize=9.5,
        color='#475467'
    )

    fig.subplots_adjust(left=0.28, right=0.92, top=0.90, bottom=0.08)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)



def plot_barh(records, batch_dir: Path, output_path: Path, dpi: int):
    score_matrix = build_score_matrix(records)
    labels = [item['platform'] for item in records]
    y = np.arange(len(records))
    bar_h = 0.22

    fig_h = max(8, len(records) * 0.36 + 2.2)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), facecolor='white')

    visual = score_matrix[:, 0]
    grouping = score_matrix[:, 1]
    alignment = score_matrix[:, 2]

    ax.barh(y - bar_h, visual, height=bar_h, label='视觉显著性差异')
    ax.barh(y, grouping, height=bar_h, label='组内紧密与分离')
    ax.barh(y + bar_h, alignment, height=bar_h, label='对齐一致性')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)

    ax.set_title('界面层次结构三维度评分对比图', fontsize=18, fontweight='bold', pad=18)
    ax.set_xlabel('得分', fontsize=12, fontweight='bold')
    ax.set_ylabel('界面样本', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=10)

    add_summary_box(fig, records, score_matrix, batch_dir)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.06)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise FileNotFoundError(f'批次目录不存在：{batch_dir}')

    records = load_results(batch_dir)
    records = sort_records(records, args.sort_by)

    if args.top_n is not None:
        records = records[:args.top_n]

    default_name = 'dimension_scores_clean.png' if args.style == 'heatmap' else 'dimension_scores_barh.png'
    output_path = Path(args.output) if args.output else batch_dir / default_name

    if args.style == 'heatmap':
        plot_heatmap(records, batch_dir, output_path, dpi=args.dpi)
    else:
        plot_barh(records, batch_dir, output_path, dpi=args.dpi)

    print(f'已生成图表：{output_path}')


if __name__ == '__main__':
    main()

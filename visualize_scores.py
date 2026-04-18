import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，避免中文显示为方框
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def extract_display_name(image_name):
    """
    从 image_name 中提取用于展示的平台名称。
    例如：'01_CSDN.png' -> 'CSDN', '02_知乎.png' -> '知乎'
    """
    # 移除 .png 后缀
    base = image_name.replace('.png', '')
    # 按下划线拆分，取第二部分（索引1），若没有下划线则取整个名称
    parts = base.split('_')
    if len(parts) >= 2:
        return parts[1]
    else:
        return parts[0]

def plot_scores(json_file_path, output_image='scores_bar_chart.png'):
    """
    读取 JSON 文件，生成评分柱状图并保存/显示。
    
    参数:
        json_file_path (str): JSON 文件路径
        output_image (str): 输出图片文件名，默认 'scores_bar_chart.png'
    """
    # 1. 加载 JSON 数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {json_file_path} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的 JSON 格式。")
        return

    # 2. 提取数据：平台名称、评分、置信度
    platforms = []
    scores = []
    confidences = []
    for item in data:
        name = extract_display_name(item['image_name'])
        platforms.append(name)
        scores.append(item['overall_score'])
        confidences.append(item['confidence'])  # 'high' 或 'medium'

    # 3. 按评分降序排序（从高到低），便于观察优劣
    sorted_indices = np.argsort(scores)[::-1]  # 降序索引
    platforms_sorted = [platforms[i] for i in sorted_indices]
    scores_sorted = [scores[i] for i in sorted_indices]
    confidences_sorted = [confidences[i] for i in sorted_indices]

    # 4. 设置柱状图颜色：根据置信度（high -> 绿色, medium -> 橙色, low -> 红色）
    color_map = {
        'high': "#68b2eb",   # 鲜绿色
        'medium': '#ff7f0e', # 橙色
        'low': '#d62728'     # 红色
    }
    bar_colors = [color_map.get(conf, '#1f77b4') for conf in confidences_sorted]

    # 5. 创建图形
    plt.figure(figsize=(12, 7))
    bars = plt.bar(platforms_sorted, scores_sorted, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # 6. 在柱顶添加数值标签
    for bar, score in zip(bars, scores_sorted):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{score:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 7. 添加平均分参考线
    avg_score = np.mean(scores_sorted)
    plt.axhline(y=avg_score, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label=f'平均分: {avg_score:.2f}')

    # 8. 添加图例（置信度颜色说明）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['high'], edgecolor='black', label='高置信度'),
        Patch(facecolor=color_map['medium'], edgecolor='black', label='中置信度')
    ]
    # 如果数据中有低置信度，可以取消下面注释
    # if any(c == 'low' for c in confidences_sorted):
    #     legend_elements.append(Patch(facecolor=color_map['low'], edgecolor='black', label='低置信度'))

    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 9. 图表装饰
    plt.title('各平台 UI/UX 整体评分对比', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('平台名称', fontsize=12, labelpad=10)
    plt.ylabel('整体评分 (overall_score)', fontsize=12, labelpad=10)
    plt.ylim(0, 10)  # 评分范围 0-10，可根据实际调整
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(np.arange(0, 10.5, 0.5), fontsize=9)
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    # 10. 添加数据来源或备注
    plt.figtext(0.99, 0.01, '数据来源：批量评估结果 (batch_summary.json)', 
                ha='right', va='bottom', fontsize=8, style='italic', alpha=0.6)

    # 11. 调整布局，避免标签被截断
    plt.tight_layout()

    # 12. 保存图片（可选）并显示
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {output_image}")
    plt.show()

# 使用示例（请根据实际文件路径修改）
if __name__ == "__main__":
    # 假设 JSON 文件名为 'batch_summary.json'，位于当前目录
    json_file = "outputs/batch_20260412_142235/batch_summary.json"
    plot_scores(json_file, output_image="web_platform_scores.png")
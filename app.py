from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from adapters import build_evaluator
from config import Settings
from parser_utils import parse_and_validate
from visualizer import create_summary_card, plot_bar_chart, save_json_result


def main() -> None:
    parser = argparse.ArgumentParser(description="UI Hierarchy Evaluation")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入 UI 截图路径，例如 input/sample_ui.png",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"未找到输入图片: {image_path}")

    settings = Settings.from_env()
    output_root = Path(settings.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    evaluator = build_evaluator(settings)

    print(f"[1/4] 调用多模态模型进行层级评估: {image_path.name}")
    raw_text = evaluator.evaluate_image(str(image_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{image_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "raw.txt"
    with open(raw_path, "w", encoding="utf-8") as file:
        file.write(raw_text)

    print("[2/4] 解析并校验结构化 JSON")
    result = parse_and_validate(raw_text, image_name=image_path.name)

    json_path = run_dir / "result.json"
    chart_path = run_dir / "scores.png"
    card_path = run_dir / "summary.png"

    print("[3/4] 保存 JSON 结果")
    save_json_result(result, str(json_path))

    print("[4/4] 生成可视化报告")
    plot_bar_chart(result, str(chart_path))
    create_summary_card(result, str(image_path), str(card_path))

    print("\n评估完成")
    print(f"- 输出目录: {run_dir}")
    print(f"- RAW: {raw_path}")
    print(f"- JSON: {json_path}")
    print(f"- Score Chart: {chart_path}")
    print(f"- Summary Card: {card_path}")
    print(f"- Overall Score: {result.overall_score:.1f}/10")
    print(f"- Confidence: {result.confidence}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

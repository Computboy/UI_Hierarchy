from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import Settings
from cv_analysis import save_image_bgr
from pipeline import evaluate_ui_hierarchy
from visualizer import create_summary_card, plot_bar_chart, save_json_result


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI Hierarchy Hybrid Evaluation")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Evaluate one UI screenshot.")
    source_group.add_argument("--input-dir", type=str, help="Evaluate every supported image in a directory.")
    source_group.add_argument("--all-input", action="store_true", help="Evaluate every image inside ./input.")
    parser.add_argument("--limit", type=int, default=None, help="Limit batch mode to the first N images.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Disable the multimodal model and use OpenCV plus heuristic fallback only.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override UI_EVAL_MODEL for this run.")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override UI_EVAL_PROVIDER for this run. Use auto, openai_responses, or openai_compatible_chat.",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Override OPENAI_BASE_URL for this run.")
    return parser.parse_args()


def collect_image_paths(directory: Path, limit: int | None = None) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    image_paths = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES],
        key=lambda path: path.name,
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0.")
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No supported images were found in: {directory}")
    return image_paths


def build_run_dir(parent_dir: Path, image_path: Path, timestamp: str) -> Path:
    run_dir = parent_dir / f"{image_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def evaluate_single_image(
    image_path: Path,
    settings: Settings,
    parent_output_dir: Path,
    *,
    progress_prefix: str = "",
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = build_run_dir(parent_output_dir, image_path, timestamp)

    print(f"{progress_prefix}[1/5] Extracting OpenCV features: {image_path.name}")
    result, artifacts = evaluate_ui_hierarchy(str(image_path), settings)

    json_path = run_dir / "result.json"
    overlay_path = run_dir / "layout_overlay.png"
    chart_path = run_dir / "scores.png"
    card_path = run_dir / "summary.png"

    print(f"{progress_prefix}[2/5] Saving JSON result")
    save_json_result(result, str(json_path))

    print(f"{progress_prefix}[3/5] Saving layout overlay")
    save_image_bgr(artifacts.overlay_image, str(overlay_path))

    print(f"{progress_prefix}[4/5] Rendering score chart")
    plot_bar_chart(result, str(chart_path))

    print(f"{progress_prefix}[5/5] Rendering summary card")
    create_summary_card(result, str(image_path), str(card_path))

    if artifacts.llm_raw_text:
        (run_dir / "llm_raw.txt").write_text(artifacts.llm_raw_text, encoding="utf-8")
    if artifacts.llm_error:
        (run_dir / "llm_error.txt").write_text(artifacts.llm_error, encoding="utf-8")

    print(f"\n{progress_prefix}Evaluation finished")
    print(f"{progress_prefix}- Output Dir: {run_dir}")
    print(f"{progress_prefix}- JSON: {json_path}")
    print(f"{progress_prefix}- Layout Overlay: {overlay_path}")
    print(f"{progress_prefix}- Score Chart: {chart_path}")
    print(f"{progress_prefix}- Summary Card: {card_path}")
    print(f"{progress_prefix}- Overall Score: {result.overall_score:.1f}/10")
    print(f"{progress_prefix}- Confidence: {result.confidence}")
    print(f"{progress_prefix}- LLM Used: {result.detection_summary.llm_used}")

    return {
        "image_name": result.image_name,
        "overall_score": result.overall_score,
        "confidence": result.confidence,
        "llm_used": result.detection_summary.llm_used,
        "llm_status": result.detection_summary.llm_status,
        "output_dir": str(run_dir),
    }


def save_batch_summary(batch_dir: Path, batch_records: Iterable[dict]) -> Path:
    summary_path = batch_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(list(batch_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def run_batch(image_paths: list[Path], settings: Settings, output_root: Path) -> None:
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = output_root / f"batch_{batch_timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        print(f"\n========== [{index}/{total}] {image_path.name} ==========")
        record = evaluate_single_image(
            image_path,
            settings,
            batch_dir,
            progress_prefix=f"[{index}/{total}] ",
        )
        records.append(record)

    summary_path = save_batch_summary(batch_dir, records)
    print("\nBatch evaluation finished")
    print(f"- Batch Output Dir: {batch_dir}")
    print(f"- Batch Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    settings = Settings.from_env().with_cli_overrides(
        skip_llm=args.skip_llm,
        base_url=args.base_url,
        provider=args.provider,
        model=args.model,
    )
    output_root = Path(settings.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if settings.enable_mllm:
        print(f"[LLM] {settings.llm_runtime_summary()}")
        if not settings.api_key:
            print("[LLM] OPENAI_API_KEY not found. The run will fall back to heuristics.")
    else:
        print("[LLM] disabled. The run will use OpenCV plus heuristic fallback only.")

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        evaluate_single_image(image_path, settings, output_root)
        return

    input_dir = Path("input") if args.all_input else Path(args.input_dir)
    image_paths = collect_image_paths(input_dir, limit=args.limit)
    run_batch(image_paths, settings, output_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

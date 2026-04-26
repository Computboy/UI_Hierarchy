# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **UI Hierarchy Hybrid Evaluation System** — a Python-based pipeline that evaluates the visual hierarchy quality of UI screenshots using a combination of **OpenCV computer vision** and **multimodal LLM** analysis.

It was developed for a university course (信息与交互设计技术, 大二春夏学期, 1-8周个人作业) to assess how well a UI screenshot establishes visual hierarchy across three dimensions:

1. **视觉显著性差异** (Visual Saliency Difference) — Can users quickly distinguish primary vs. secondary information?
2. **组内紧密与组间分离度** (Grouping Compactness & Separation) — Are related elements grouped, and groups separated?
3. **对齐一致性** (Alignment Consistency) — Is the layout organized along stable reference lines?

## Architecture

```
app.py                Entry point, CLI argument parsing, single/batch evaluation orchestration
pipeline.py           Core evaluation pipeline: orchestrates OpenCV + LLM, assembles dimensions
cv_analysis.py        OpenCV-based image analysis: element detection, grouping, metrics computation
adapters.py           LLM adapter layer: OpenAI Responses API and Chat Completions API
prompts.py            System and user prompts for the font hierarchy MLLM evaluation
schemas.py            Pydantic models: evaluation output schema, dimension/metric definitions
parser_utils.py       JSON parsing with repair logic for robust LLM output handling
config.py             Settings dataclass: env vars, provider resolution, CLI overrides
visualizer.py         Matplotlib visualization: bar charts, summary cards, JSON export
plot_dimension_scores.py  Batch-level heatmap/horizontal bar chart for dimension scores
visualize_scores.py   Batch-level overall score bar chart (from batch_summary.json)
```

### Data Flow

1. `app.py` parses CLI args → loads `Settings` from `.env`
2. Calls `pipeline.evaluate_ui_hierarchy(image_path, settings)`
3. Pipeline runs `cv_analysis.analyze_image_with_opencv()` for element detection, grouping, and 12 metric computations
4. Pipeline optionally calls LLM via `adapters.FontHierarchyEvaluator` for font hierarchy delta scoring
5. Pipeline assembles 3 dimension scores from 12 metrics (4 per dimension)
6. Returns `UIHierarchyEvaluation` result + `PipelineArtifacts`
7. `app.py` saves: `result.json`, `layout_overlay.png`, `scores.png`, `summary.png`, optional `llm_raw.txt`/`llm_error.txt`

### Key Data Structures

- **BoundingBox** / **ElementGroup** — detected UI elements and their groupings
- **LocalMetricMeasurement** — a single metric with raw value, normalized score (1-10), formula, interpretation
- **MetricEvaluation** / **DimensionEvaluation** — Pydantic models for JSON output
- **UIHierarchyEvaluation** — top-level result with overall score, confidence, 3 dimensions, suggestions

### Dimension-Metric Mapping (schemas.py: DIMENSION_METRIC_KEYS)

| Dimension | Metrics |
|---|---|
| visual_saliency_difference | font_hierarchy_delta, visual_weight_delta, region_area_delta, foreground_background_contrast_delta |
| grouping_compactness_separation | within_group_distance_mean, between_group_distance_mean, spatial_cluster_compactness, group_interval_ratio |
| alignment_consistency | edge_alignment_error, center_axis_alignment_error, grid_consistency, collinear_element_ratio |

### Scoring

- All metrics normalize to 1-10 scale using `_score_higher_better` / `_score_lower_better` in `cv_analysis.py`
- Saliency dimension uses **weighted average** (font 0.35, weight 0.30, area 0.20, contrast 0.15)
- Grouping and alignment use **simple average** of their 4 metrics
- Overall score = simple average of 3 dimension scores

## Running the Project

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires `.env` file in project root (see `.env.example`).

### Single Image Evaluation

```bash
python app.py --image input/01_CSDN.png
```

### Batch Evaluation

```bash
# Evaluate all images in ./input/
python app.py --all-input

# Evaluate all images in a specific directory
python app.py --input-dir ./input

# Limit to first N images
python app.py --all-input --limit 10
```

### CLI Options

| Flag | Description |
|---|---|
| `--image PATH` | Evaluate one UI screenshot |
| `--input-dir PATH` | Evaluate every image in a directory |
| `--all-input` | Evaluate every image in `./input/` |
| `--limit N` | Limit batch mode to first N images |
| `--skip-llm` | Disable MLLM, use OpenCV + heuristic fallback only |
| `--model NAME` | Override `UI_EVAL_MODEL` |
| `--provider NAME` | Override `UI_EVAL_PROVIDER` (auto, openai_responses, openai_compatible_chat) |
| `--base-url URL` | Override `OPENAI_BASE_URL` |

### Post-Processing Scripts

```bash
# Generate dimension heatmap/barh chart from batch results
python plot_dimension_scores.py --batch-dir outputs/batch_XXX --style heatmap --top-n 30

# Generate overall score bar chart from batch_summary.json
python visualize_scores.py
# (edit the hardcoded path inside visualize_scores.py)
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required for LLM) | API key for multimodal model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `UI_EVAL_PROVIDER` | `auto` | `auto`, `openai_responses`, `openai_compatible_chat` |
| `UI_EVAL_MODEL` | `gpt-4.1-mini` | Model name |
| `UI_EVAL_OUTPUT_DIR` | `outputs` | Output directory |
| `UI_HIERARCHY_ENABLE_MLLM` | `true` | Enable/disable MLLM font scoring |

## LLM Provider Resolution

The `resolved_provider()` method in `config.py` determines which API transport to use:
- `auto` + official OpenAI endpoint → `openai_responses` (Responses API)
- `auto` + third-party endpoint → `openai_compatible_chat` (Chat Completions API)
- Falls back to Chat Completions if the SDK doesn't support Responses API

## Input Data

The `input/` directory contains **50 Chinese and international web platform screenshots** (CSDN, 知乎, 掘金, Hacker News, TechCrunch, BBC News, etc.) with naming convention `NN_平台名称.png`.

## Output Structure

Each evaluation creates a timestamped run directory under `outputs/`:
```
outputs/{image_name}_{timestamp}/
  result.json           # Full evaluation result (Pydantic model dump)
  layout_overlay.png    # OpenCV-detected elements and groups overlay
  scores.png            # Dimension score bar chart
  summary.png           # Full summary card with image + details
  llm_raw.txt           # Raw LLM JSON response (if LLM succeeded)
  llm_error.txt         # LLM error message (if LLM failed)
```

Batch evaluations create `outputs/batch_{timestamp}/` with per-image subdirectories and `batch_summary.json`.

## No Existing Rules

There are no Cursor rules (`.cursor/`), Cursor rules file (`.cursorrules`), or Copilot rules (`.github/copilot-instructions.md`) in this project.

# UI Hierarchy-Clarity 混合评估系统

基于 **OpenCV 计算机视觉 + 多模态大模型** 的 UI 截图**视觉层级清晰度**质量评估工具。

从面积、对比、间距、聚类、对齐等可解释指标出发，结合多模态模型对字体层级的判断，将 UI 截图的视觉层级量化为三个维度的 1–10 分评分，并附带证据与改进建议。

---

## 评估维度

| 维度 | 含义 | 子指标 |
|---|---|---|
| **视觉显著性差异** | 用户能否快速分辨第一层与第二层信息 | 字体层级差值、视觉权重差值、区域面积差值、前景背景对比差值 |
| **组内紧密与组间分离度** | 用户是否容易判断哪些元素属于同一模块 | 组内距离均值、组间距离均值、空间聚类紧凑度、分组间隔比 |
| **对齐一致性** | 版面是否沿统一参考线组织 | 边缘对齐误差、中轴对齐误差、栅格一致性、共线元素占比 |

## 技术路线

```
截图输入
  │
  ├─ OpenCV 特征提取 ──── 元素检测 → 分组 → 12 项指标计算 → 归一化评分 (1-10)
  │
  ├─ 多模态模型 (可选) ── 字体层级差值评估 → JSON 结构化输出
  │
  └─ 汇总 ─────────────── 三维分数 → 总体评分 → 置信度 → 改进建议
```

- **OpenCV 路线**：自适应阈值 + 形态学操作检测前景元素 → 包围框分组 → 面积/对比/间距/对齐等可解释指标
- **多模态模型路线**：调用 GPT-4o 等视觉模型，仅评估字体层级差值（字号、字重、行高差异）
- **回退机制**：模型调用失败或未配置时，字体层级差值自动回退为基于文本框高度的启发式估算

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：`opencv-python-headless`、`numpy`、`openai`、`pydantic`、`pillow`、`matplotlib`、`python-dotenv`。

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
UI_EVAL_PROVIDER=auto
UI_EVAL_MODEL=gpt-4o
UI_EVAL_OUTPUT_DIR=outputs
UI_HIERARCHY_ENABLE_MLLM=true
```

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | API 密钥（启用多模态模型时必填） |
| `OPENAI_BASE_URL` | API 端点地址，支持 OpenAI 官方或兼容接口 |
| `UI_EVAL_PROVIDER` | 提供商模式：`auto` / `openai_responses` / `openai_compatible_chat` |
| `UI_EVAL_MODEL` | 模型名称 |
| `UI_EVAL_OUTPUT_DIR` | 输出目录 |
| `UI_HIERARCHY_ENABLE_MLLM` | 是否启用多模态模型 |

### 3. 运行评估

**单张截图：**

```bash
python app.py --image input/01_CSDN.png
```

**批量评估 `input/` 目录下的所有截图：**

```bash
python app.py --all-input
```

**限制前 10 张：**

```bash
python app.py --all-input --limit 10
```

**跳过模型，纯 OpenCV + 启发式回退：**

```bash
python app.py --image input/01_CSDN.png --skip-llm
```

## 输出说明

每张截图会在 `outputs/` 下生成一个带时间戳的子目录：

```
outputs/01_CSDN_20260418_161724/
├── result.json           # 完整评估结果（JSON）
├── layout_overlay.png    # OpenCV 元素检测与分组的叠加图
├── scores.png            # 三维度评分柱状图
├── summary.png           # 综合报告卡片（含截图、评分、证据、建议）
├── llm_raw.txt           # 多模态模型原始 JSON 响应（如调用成功）
└── llm_error.txt         # 模型调用错误信息（如调用失败）
```

批量评估额外生成 `batch_summary.json`，汇总所有样本的评分。

## 批处理可视化

### 维度评分热力图 / 横向柱状图

```bash
python plot_dimension_scores.py --batch-dir outputs/batch_20260412_142235 --style heatmap
python plot_dimension_scores.py --batch-dir outputs/batch_20260412_142235 --style barh
```

支持 `--sort-by`（overall / visual / grouping / alignment）、`--top-n`、`--dpi` 等参数。

### 总体评分对比柱状图

编辑 `visualize_scores.py` 中的 JSON 路径后运行：

```bash
python visualize_scores.py
```

## 评分规则

- 每项子指标先归一化到 **1–10 分**（线性映射 + 截断）
- **视觉显著性差异**：加权平均（字体 0.35、权重 0.30、面积 0.20、对比 0.15）
- **组内紧密与组间分离度**：四项简单平均
- **对齐一致性**：四项简单平均
- **总体评分**：三维度加权求和（视觉显著性差异 0.35、组内紧密与组间分离度 0.35、对齐一致性 0.30）

分数档位：

| 分数区间 | 等级 |
|---|---|
| ≥ 8 | 较好 |
| 5.5 – 8 | 中等 |
| < 5.5 | 偏弱 |

## 项目结构

```
├── app.py                     # CLI 入口，单张/批量评估编排
├── pipeline.py                # 核心评估管线（OpenCV + LLM 融合）
├── cv_analysis.py             # OpenCV 图像分析（元素检测、分组、12 项指标）
├── adapters.py                # 多模态模型适配层（Responses API / Chat Completions）
├── prompts.py                 # 字体层级评估的 System / User Prompt
├── schemas.py                 # Pydantic 数据模型与维度定义
├── parser_utils.py            # LLM JSON 解析与容错修复
├── config.py                  # 环境变量读取与配置管理
├── visualizer.py              # 柱状图 & 综合报告卡片生成
├── plot_dimension_scores.py   # 批次维度评分热力图/横向柱状图
├── visualize_scores.py        # 批次总体评分对比柱状图
├── input/                     # 输入截图（50 个中文/国际网站）
├── outputs/                   # 评估输出目录（git-ignored）
└── requirements.txt           # Python 依赖
```

## License

本项目为课程学习用途，仅供学术参考。

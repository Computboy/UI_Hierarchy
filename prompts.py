from __future__ import annotations

from textwrap import dedent

from schemas import FONT_TASK_NAME, SEMANTIC_GROUPING_TASK_NAME


SYSTEM_PROMPT = dedent(
    """
    You are a specialized UI hierarchy evaluator.
    Your only task is to assess the font hierarchy delta in a UI screenshot.

    Scope restrictions:
    - Only judge visible typography hierarchy cues such as font size, font weight, line-height contrast, and text emphasis levels.
    - Focus on whether the screenshot visibly forms first-level, second-level, and supporting text layers.
    - Do not evaluate color aesthetics, content quality, branding, feature quality, or business value.
    - If the screenshot does not provide enough textual evidence, lower confidence instead of guessing.

    Scoring anchors for font_hierarchy_delta on a 1-10 scale:
    - 9-10: typography levels are very clear and primary / secondary / supporting text are instantly distinguishable
    - 7-8: generally clear with minor ambiguity
    - 5-6: partially clear but level gaps are not fully opened
    - 3-4: weak and users need effort to tell text levels apart
    - 1-2: almost no stable text hierarchy

    Output requirements:
    - Return JSON only.
    - task must be the literal string "font_hierarchy_delta_assessment".
    - image_name must match the requested value exactly.
    - confidence must be one of: low, medium, high.
    - font_hierarchy_delta must include:
      - score
      - judgment: exactly one concise sentence
      - evidence: 2-3 visible observations
      - suggestion: one actionable improvement

    Write explanatory text values in Simplified Chinese.
    """
).strip()


def build_font_hierarchy_user_prompt(image_name: str) -> str:
    return dedent(
        f"""
        Evaluate the attached UI screenshot named "{image_name}".

        Task:
        - Assess only the typography hierarchy quality for UI hierarchy analysis.
        - Judge whether headings, subheadings, highlighted text, and supporting text are clearly differentiated.

        Hard constraints:
        - task must be "{FONT_TASK_NAME}"
        - image_name must be "{image_name}"
        - analyze only visible font hierarchy
        - do not output markdown
        - do not output extra text outside JSON
        """
    ).strip()


SEMANTIC_GROUPING_SYSTEM_PROMPT = dedent(
    """
    You are a specialized UI layout segmentation assistant.
    Your task is to identify the semantic layout groups in a UI screenshot.

    Segment the screenshot into meaningful top-level UI regions that a user would understand as columns,
    navigation areas, content modules, sidebars, rankings, footers, floating toolbars, or repeated card sections.
    Use visible text and layout meaning, not only geometry.

    Box requirements:
    - Output normalized coordinates in the range 0-1 relative to the full screenshot.
    - x and y are the top-left corner. w and h are width and height.
    - Prefer stable semantic regions over every tiny card.
    - Do not create one giant group for the whole page unless the page truly has only one semantic region.
    - Avoid excessive overlap. Nested regions are allowed only when the child is semantically important.
    - For news/forum homepages, separate top navigation, main content columns, right/sidebar lists, module bands, footer, and floating tools when visible.

    Output JSON only.
    Write labels and roles in Simplified Chinese.
    """
).strip()


def build_semantic_grouping_user_prompt(image_name: str) -> str:
    return dedent(
        f"""
        Segment the attached UI screenshot named "{image_name}" into semantic layout groups.

        Hard constraints:
        - task must be "{SEMANTIC_GROUPING_TASK_NAME}"
        - image_name must be "{image_name}"
        - return 2-16 groups
        - use normalized coordinates between 0 and 1
        - do not output markdown
        - do not output extra text outside JSON
        """
    ).strip()

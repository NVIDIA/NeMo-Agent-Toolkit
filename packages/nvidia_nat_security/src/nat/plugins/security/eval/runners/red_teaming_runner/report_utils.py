# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Report utilities for red teaming evaluation results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)


def _validate_columns(df: pd.DataFrame, required_columns: list[str], context: str = "") -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must exist.
        context: Optional context string for error message (e.g., function name).

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        available = list(df.columns)
        ctx = f" in {context}" if context else ""
        raise ValueError(f"Missing required column(s){ctx}: {missing}. Available columns: {available}")


def plot_score_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str = "score",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str = "Risk Score",
    y_range: tuple[float, float] = (-0.05, 1.05),  # Start below 0 to show full box when min=0
    box_color: str = "rgb(55, 126, 184)",
    box_fill_opacity: float = 0.3,
    point_color: str = "rgba(255, 50, 0, 0.5)",
    point_size: int = 6,
    jitter: float = 0.3,
) -> go.Figure:
    """Create a box plot with data points overlaid inside the boxes.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (grouping variable).
        y: Column name for y-axis (score values).
        title: Plot title. Defaults to "Score Distribution by {x}".
        x_label: X-axis label. Defaults to the column name.
        y_label: Y-axis label.
        y_range: Tuple of (min, max) for y-axis range.
        box_color: RGB color for box outline.
        box_fill_opacity: Opacity for box fill (0-1).
        point_color: RGBA color for data points.
        point_size: Size of data points.
        jitter: Horizontal jitter for points (0-1).

    Returns:
        The Plotly Figure object.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    _validate_columns(df, [x, y], "plot_score_boxplot")

    if title is None:
        title = f"Score Distribution by {x}"
    if x_label is None:
        x_label = x

    # Parse box_color to create fill color with opacity
    # Parse box_color to create fill color with opacity
    if box_color.startswith("rgb(") and not box_color.startswith("rgba("):
        box_fill_color = box_color.replace("rgb(", "rgba(").replace(")", f", {box_fill_opacity})")
    else:
        box_fill_color = box_color  # Use as-is if already rgba or different format

    # Use go.Box directly for explicit control over data
    fig = go.Figure()

    # Get unique x values
    unique_x_values = df[x].unique()
    n_categories = len(unique_x_values)

    # Calculate box width dynamically based on number of categories
    # Wider boxes for fewer categories, narrower for more
    box_width = max(0.2, min(0.67, 1.5 / n_categories))
    half_width = box_width / 2
    means: list[tuple[int, float, str]] = []  # (x_position, mean_value, label)

    for i, x_val in enumerate(unique_x_values):
        mask = df[x] == x_val
        subset = df.loc[mask]
        y_values = subset[y].tolist()  # Explicitly convert to list
        # Use index (uid) for hover text
        hover_text = subset.index.tolist()

        # Calculate mean for this group
        mean_val = sum(y_values) / len(y_values) if y_values else 0
        means.append((i, mean_val, str(x_val)))

        fig.add_trace(
            go.Box(
                y=y_values,
                name=str(x_val),
                boxpoints="all",
                pointpos=0,
                jitter=jitter,
                width=box_width,
                marker=dict(color=point_color, size=point_size),
                line=dict(color=box_color),
                fillcolor=box_fill_color,
                text=hover_text,
                hovertemplate="uid: %{text}<br>score: %{y}<extra></extra>",
            ))

    # Add mean lines and annotations for each box
    for x_pos, mean_val, _label in means:
        # Add dotted horizontal line at the mean (spanning exact box width)
        fig.add_shape(
            type="line",
            x0=x_pos - half_width,
            x1=x_pos + half_width,
            y0=mean_val,
            y1=mean_val,
            line=dict(color="red", width=2, dash="dot"),
        )
        # Add mean label annotation right next to the box (at left edge)
        fig.add_annotation(
            x=x_pos - half_width,
            y=mean_val,
            text=f"mean: {mean_val:.2f}",
            showarrow=False,
            font=dict(size=12, color="red", weight="bold"),
            xanchor="right",
            yanchor="middle",
            xshift=-5,  # Small gap from the box edge
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=list(y_range),
        showlegend=False,
    )

    return fig


def plot_success_rate_bar(
    df: pd.DataFrame,
    x: str,
    y: str = "score",
    threshold: float = 0.5,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str = "Attack Success Rate (%)",
    bar_color: str = "rgb(55, 126, 184)",
) -> go.Figure:
    """Create a bar chart showing attack success rate (percentage of scores above threshold).

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (grouping variable).
        y: Column name for the score values.
        threshold: Score threshold to consider as "success" (default 0.5).
        title: Plot title. Defaults to "Attack Success Rate by {x}".
        x_label: X-axis label. Defaults to the column name.
        y_label: Y-axis label.
        bar_color: Color for the bars.

    Returns:
        The Plotly Figure object.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    _validate_columns(df, [x, y], "plot_success_rate_bar")

    if title is None:
        title = f"Attack Success Rate by {x} (threshold={threshold})"
    if x_label is None:
        x_label = x

    # Calculate success rate for each group
    unique_x_values = df[x].unique()
    x_labels = []
    success_rates = []
    counts = []

    for x_val in unique_x_values:
        mask = df[x] == x_val
        subset = df.loc[mask]
        total = len(subset)
        successes = (subset[y] > threshold).sum()
        rate = (successes / total * 100) if total > 0 else 0

        x_labels.append(str(x_val))
        success_rates.append(rate)
        counts.append(f"{successes}/{total}")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=success_rates,
            text=[f"{rate:.1f}%<br>({count})" for rate, count in zip(success_rates, counts, strict=True)],
            textposition="outside",
            marker_color=bar_color,
            hovertemplate="%{x}<br>Attack Success Rate: %{y:.1f}%<br>Count: %{text}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_range=[-10, 125],  # Leave room for labels above bars
        showlegend=False,
    )

    return fig


def generate_standard_plots(df: pd.DataFrame) -> list[tuple[str, str, go.Figure | None]]:
    """Generate standard plots for red teaming results, grouped by category.

    Args:
        df: DataFrame with columns: scenario_id, condition_name, tags, scenario_group, score.

    Returns:
        List of tuples (filename, title, figure) for each plot.
        Section headers have figure=None and are rendered as section titles.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    # Validate required columns upfront
    _validate_columns(df, ["scenario_id", "score", "condition_name"], "generate_standard_plots")

    plots: list[tuple[str, str, go.Figure | None]] = []

    # ==================== RESULTS BY SCENARIO ID ====================
    plots.append(("_section", "Results by group: Scenario ID", None))

    fig_scenario = plot_score_boxplot(
        df,
        x="scenario_id",
        y="score",
        title="Score Distribution by Scenario",
        x_label="Scenario",
    )
    plots.append(("scenario_id_boxplot", "Score Distribution", fig_scenario))

    fig_scenario_bar = plot_success_rate_bar(
        df,
        x="scenario_id",
        y="score",
        title="Attack Success Rate by Scenario",
        x_label="Scenario",
    )
    plots.append(("scenario_id_success_rate", "Attack Success Rate", fig_scenario_bar))

    # ==================== RESULTS BY SCENARIO GROUP ====================
    if "scenario_group" in df.columns:
        plots.append(("_section", "Results by group: Scenario Group", None))

        fig_group = plot_score_boxplot(
            df,
            x="scenario_group",
            y="score",
            title="Risk Score Distribution by Scenario Group",
            x_label="Scenario Group",
        )
        plots.append(("scenario_group_boxplot", "Score Distribution", fig_group))

        fig_group_bar = plot_success_rate_bar(
            df,
            x="scenario_group",
            y="score",
            title="Attack Success Rate by Scenario Group",
            x_label="Scenario Group",
        )
        plots.append(("scenario_group_success_rate", "Attack Success Rate", fig_group_bar))

    # ==================== RESULTS BY CONDITION ====================
    plots.append(("_section", "Results by group: Output Filtering Condition", None))

    fig_condition = plot_score_boxplot(
        df,
        x="condition_name",
        y="score",
        title="Score Distribution by Output Filtering Condition",
        x_label="Condition",
    )
    plots.append(("condition_name_boxplot", "Score Distribution", fig_condition))

    fig_condition_bar = plot_success_rate_bar(
        df,
        x="condition_name",
        y="score",
        title="Attack Success Rate by Output Filtering Condition",
        x_label="Condition",
    )
    plots.append(("condition_name_success_rate", "Attack Success Rate", fig_condition_bar))

    # ==================== RESULTS BY TAGS ====================
    if "tags" in df.columns:
        df_tags = df.explode("tags")
        df_tags = df_tags.dropna(subset=["tags"])
        if not df_tags.empty:
            plots.append(("_section", "Results by group: Tags", None))

            fig_tags = plot_score_boxplot(
                df_tags,
                x="tags",
                y="score",
                title="Risk Score Distribution by Tag",
                x_label="Tag",
            )
            plots.append(("tags_boxplot", "Score Distribution", fig_tags))

            fig_tags_bar = plot_success_rate_bar(
                df_tags,
                x="tags",
                y="score",
                title="Attack Success Rate by Tag",
                x_label="Tag",
            )
            plots.append(("tags_success_rate", "Attack Success Rate", fig_tags_bar))

    return plots


def _get_risk_color(value: float, max_value: float = 1.0) -> str:
    """Get a color that transitions from low to high risk based on value.

    The color transitions and opacity increases as risk increases:
    - Opacity: 0.3 (at 0) -> 1.0 (at max)
    - Color: muted -> intense red

    Args:
        value: The risk value (0 to max_value).
        max_value: The maximum value (1.0 for scores, 100.0 for percentages).

    Returns:
        RGBA color string.
    """
    # Normalize to 0-1 range
    normalized = min(max(value / max_value, 0.0), 1.0)

    # Interpolate color
    normalized = normalized if normalized >= 0.5 else normalized**2
    r = int(30 + (255 - 30) * normalized)
    g = int(10 + (0 - 10) * normalized)
    b = int(10 + (0 - 10) * normalized)

    # Interpolate opacity from 0.5 to 1.0
    opacity = 0.3 + 0.7 * normalized

    return f"rgba({r}, {g}, {b}, {opacity})"


def _render_summary_html(summary: dict[str, Any] | None) -> str:
    """Render the summary section as HTML.

    Args:
        summary: The summary dictionary from _compute_result_summary.

    Returns:
        HTML string for the summary section.
    """
    if not summary:
        return ""

    overall_score = summary.get("overall_score", 0.0)
    attack_success_rate = summary.get("attack_success_rate", 0.0)
    num_scenarios = summary.get("num_scenarios", 0)
    total_workflow_runs = summary.get("total_workflow_runs", 0)
    total_evaluations = summary.get("total_evaluations", 0)
    evaluation_successes = summary.get("evaluation_successes", 0)
    evaluation_failures = summary.get("evaluation_failures", 0)
    per_scenario = summary.get("per_scenario_summary", {})

    # Get dynamic colors based on risk values
    score_color = _get_risk_color(overall_score, 1.0)
    asr_color = _get_risk_color(attack_success_rate, 1.0)

    # Build per-scenario rows with ASR as first data column
    scenario_rows = ""
    for scenario_id, stats in per_scenario.items():
        scenario_asr = stats.get("attack_success_rate", 0.0)
        mean_score = stats.get("mean_score", 0.0)
        min_score = stats.get("min_score", 0.0)
        max_score = stats.get("max_score", 0.0)
        row_asr_color = _get_risk_color(scenario_asr, 1.0)
        scenario_rows += f"""
            <tr>
                <td>{scenario_id}</td>
                <td style="background-color: {row_asr_color}; color: white; font-weight: bold;">{scenario_asr:.1%}</td>
                <td>{mean_score:.3f}</td>
                <td>{min_score:.3f}</td>
                <td>{max_score:.3f}</td>
            </tr>"""

    return f"""
    <div class="summary-section">
        <h2 class="section-header">Summary</h2>
        <div class="summary-container">
            <div class="summary-stats">
                <div class="stat-card risk-score" style="background-color: {score_color}; border: none;">
                    <div class="stat-label" style="color: rgba(255,255,255,0.9);">Overall Risk Score ↓</div>
                    <div class="stat-value" style="color: white;">{overall_score:.3f}</div>
                </div>
                <div class="stat-card risk-score" style="background-color: {asr_color}; border: none;">
                    <div class="stat-label" style="color: rgba(255,255,255,0.9);">Attack Success Rate ↓</div>
                    <div class="stat-value" style="color: white;">{attack_success_rate:.1%}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Scenarios</div>
                    <div class="stat-value">{num_scenarios}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Workflow Runs</div>
                    <div class="stat-value">{total_workflow_runs}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Evaluations</div>
                    <div class="stat-value">{total_evaluations}</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-label">Successes</div>
                    <div class="stat-value">{evaluation_successes}</div>
                </div>
                <div class="stat-card failure">
                    <div class="stat-label">Failures</div>
                    <div class="stat-value">{evaluation_failures}</div>
                </div>
            </div>
            <h3 class="plot-title">Per-Scenario Results</h3>
            <table class="scenario-table">
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>ASR</th>
                        <th>Mean Score</th>
                        <th>Min Score</th>
                        <th>Max Score</th>
                    </tr>
                </thead>
                <tbody>
                    {scenario_rows}
                </tbody>
            </table>
        </div>
    </div>
"""


def save_combined_html(
    plots: list[tuple[str, str, go.Figure | None]],
    output_path: Path,
    page_title: str = "Red Teaming Evaluation Results",
    summary: dict[str, Any] | None = None,
) -> Path:
    """Save all plots in a single interactive HTML document.

    Args:
        plots: List of (filename, title, figure) tuples.
        output_path: Path for the combined HTML file.
        page_title: Title for the HTML page.
        summary: Optional summary dictionary to display at the top of the report.

    Returns:
        Path to the saved HTML file.
    """
    html_parts: list[str] = []

    # HTML header with styling
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{page_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #76b900;
            padding-bottom: 10px;
        }}
        h2.section-header {{
            color: #333;
            margin-top: 50px;
            padding: 15px 20px;
            background-color: #76b900;
            color: white;
            border-radius: 8px;
            font-size: 1.4em;
        }}
        h3.plot-title {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .plot-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: flex;
            flex-wrap: nowrap;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 10px 12px;
            min-width: 90px;
            flex: 1;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .stat-card.risk-score {{
            color: white;
            border: none;
        }}
        .stat-card.risk-score .stat-label {{
            color: rgba(255,255,255,0.9);
        }}
        .stat-card.success {{
            border-left: 4px solid #28a745;
        }}
        .stat-card.failure {{
            border-left: 4px solid #dc3545;
        }}
        .stat-label {{
            font-size: 0.75em;
            color: #666;
            margin-bottom: 3px;
        }}
        .stat-value {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .scenario-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .scenario-table th,
        .scenario-table td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        .scenario-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        .scenario-table tr:hover {{
            background-color: #f8f9fa;
        }}
    </style>
</head>
<body>
    <h1>{page_title}</h1>
""")

    # Add summary section at the top
    html_parts.append(_render_summary_html(summary))

    # Add each plot with its title (or section header)
    for _filename, title, fig in plots:
        if fig is None:
            # This is a section header
            html_parts.append(f"""
    <h2 class="section-header">{title}</h2>
""")
        else:
            # This is a regular plot
            plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            html_parts.append(f"""
    <h3 class="plot-title">{title}</h3>
    <div class="plot-container">
        {plot_html}
    </div>
""")

    # HTML footer
    html_parts.append("""
</body>
</html>
""")

    # Write combined HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(html_parts), encoding="utf-8")
    logger.debug("Saved combined HTML: %s", output_path)

    return output_path


def generate_and_save_report(
    flat_results: list[dict[str, Any]] | pd.DataFrame,
    output_dir: Path,
    summary: dict[str, Any] | None = None,
) -> Path | None:
    """Generate and save all plots from flat results.

    This is the main entry point for plotting. It:
    1. Converts flat results to a DataFrame
    2. Generates standard plots (by scenario, group, condition, tags)
    3. Saves a combined HTML report with all plots and summary

    Args:
        flat_results: List of flat result dictionaries from _build_flat_results.
        output_dir: Base output directory. Plots are saved in a 'graphs' subfolder.
        summary: Optional summary dictionary to display at the top of the report.

    Returns:
        Path to the combined HTML report.
    """
    report_path = output_dir / "report.html"
    is_df_empty = isinstance(flat_results, pd.DataFrame) and flat_results.empty
    if is_df_empty or (isinstance(flat_results, list) and not flat_results):
        logger.warning("No results to plot")
        return None

    # Convert to DataFrame
    if isinstance(flat_results, pd.DataFrame):
        df = flat_results
    else:
        df = pd.DataFrame(flat_results)

    # Drop rows with error_message (failed evaluations)
    if "error_message" in df.columns:
        error_count = int(df["error_message"].notna().sum())
        if error_count > 0:
            logger.info("Dropping %d rows with error_message from plotting", error_count)
            df = df[df["error_message"].isna()]

    if df.empty:
        logger.warning("No valid results to plot after filtering errors")
        return None

    # Set uid as index for hover text identification
    if "uid" in df.columns:
        df = df.set_index("uid")

    # Generate plots
    plots = generate_standard_plots(df)

    if not plots:
        logger.warning("No plots generated")
        return None

    # Save combined HTML report
    report_path = save_combined_html(
        plots,
        report_path,
        page_title=f"Red Teaming Evaluation Results for run: {output_dir.name}",
        summary=summary,
    )

    return report_path

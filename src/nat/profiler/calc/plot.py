# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nat.profiler.calc.data_models import LinearFitResult
from nat.profiler.calc.data_models import SizingMetrics

logger = logging.getLogger(__name__)


# Plotting constants
class PlotConfig:
    # Simple plot settings
    SIMPLE_FIGSIZE = (12, 6)
    SIMPLE_LINEWIDTH = 2
    SIMPLE_DPI = 150

    # Enhanced plot settings
    ENHANCED_FIGSIZE = (16, 6)
    ENHANCED_DPI = 300

    # Marker and styling
    DATA_MARKER = 'o'
    OUTLIER_MARKER = 'x'
    OUTLIER_COLOR = 'crimson'
    TREND_COLOR = 'r'
    TREND_LINESTYLE = '--'
    TREND_ALPHA = 0.8
    TREND_LINEWIDTH = 2.0

    # Colors
    LLM_LATENCY_COLOR = 'steelblue'
    RUNTIME_COLOR = 'darkgreen'
    SLA_COLOR = 'red'
    NOTE_BOX_COLOR = 'mistyrose'
    NOTE_TEXT_COLOR = 'crimson'
    STATS_BOX_COLOR = 'lightblue'

    # Alpha values
    DATA_ALPHA = 0.7
    OUTLIER_ALPHA = 0.9
    GRID_ALPHA = 0.3
    SLA_ALPHA = 0.7
    NOTE_BOX_ALPHA = 0.7
    STATS_BOX_ALPHA = 0.8

    # Sizes
    DATA_POINT_SIZE = 120
    OUTLIER_POINT_SIZE = 140
    DATA_LINEWIDTH = 1

    # Font sizes
    AXIS_LABEL_FONTSIZE = 12
    TITLE_FONTSIZE = 14
    LEGEND_FONTSIZE = 10
    NOTE_FONTSIZE = 10
    STATS_FONTSIZE = 10

    # Text positioning
    NOTE_X_POS = 0.98
    NOTE_Y_POS = 0.02
    STATS_X_POS = 0.02
    STATS_Y_POS = 0.02

    # Box styling
    NOTE_BOX_PAD = 0.3
    STATS_BOX_PAD = 0.5

    # Trend line points
    TREND_LINE_POINTS = 100

    # Font weights
    AXIS_LABEL_FONTWEIGHT = 'bold'
    TITLE_FONTWEIGHT = 'bold'


def plot_concurrency_vs_time_metrics_simple(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a simple plot of concurrency vs. p95 LLM latency and workflow runtime.
    """
    plt.figure(figsize=PlotConfig.SIMPLE_FIGSIZE)
    plt.plot(df["concurrency"],
             df["llm_latency_p95"],
             label="p95 LLM Latency (s)",
             marker=PlotConfig.DATA_MARKER,
             linewidth=PlotConfig.SIMPLE_LINEWIDTH)
    plt.plot(df["concurrency"],
             df["workflow_runtime_p95"],
             label="p95 Workflow Runtime (s)",
             marker="s",
             linewidth=PlotConfig.SIMPLE_LINEWIDTH)
    plt.xlabel("Concurrency")
    plt.ylabel("Time (seconds)")
    plt.title("Concurrency vs. p95 LLM Latency and Workflow Runtime")
    plt.grid(True, alpha=PlotConfig.GRID_ALPHA)
    plt.legend()
    plt.tight_layout()

    simple_plot_path = output_dir / "concurrency_vs_p95_simple.png"
    plt.savefig(simple_plot_path, dpi=PlotConfig.SIMPLE_DPI, bbox_inches='tight')
    plt.close()
    logger.info("Simple plot saved to %s", simple_plot_path)


def plot_metric_vs_concurrency_with_optional_fit(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    metric_name: str,
    y_label: str,
    title: str,
    color: str,
    sla_value: float = 0.0,
    sla_label: str = None,
    fit: LinearFitResult | None = None,
):
    """
    Helper to plot a metric vs concurrency with pre-computed fit, outlier highlighting, and SLA line.
    Requires pre-computed fit to be provided.
    """
    marker = PlotConfig.DATA_MARKER
    outlier_marker = PlotConfig.OUTLIER_MARKER
    outlier_color = PlotConfig.OUTLIER_COLOR
    trend_color = PlotConfig.TREND_COLOR
    trend_linestyle = PlotConfig.TREND_LINESTYLE
    trend_alpha = PlotConfig.TREND_ALPHA
    trend_linewidth = PlotConfig.TREND_LINEWIDTH
    note_box_color = PlotConfig.NOTE_BOX_COLOR
    note_text_color = PlotConfig.NOTE_TEXT_COLOR
    legend_fontsize = PlotConfig.LEGEND_FONTSIZE
    outliers_x = outliers_y = np.array([])
    outliers_note = ""

    # Skip analysis plot if no fit is available
    if not fit:
        logger.warning(f"No linear fit available for {metric_name}, skipping analysis plot")
        return False

    if fit.outliers_removed:
        # Use the concurrencies that were removed to identify outlier points
        outlier_mask = np.isin(x, fit.outliers_removed)
        outliers_x = x[outlier_mask]
        outliers_y = y[outlier_mask]
        outliers_note = f"Outliers removed: concurrencies {fit.outliers_removed}"
        # Plot cleaned data (points that weren't removed as outliers)
        non_outlier_mask = ~np.isin(x, fit.outliers_removed)
        x_clean = x[non_outlier_mask]
        y_clean = y[non_outlier_mask]
        ax.scatter(x_clean,
                   y_clean,
                   alpha=PlotConfig.DATA_ALPHA,
                   s=PlotConfig.DATA_POINT_SIZE,
                   c=color,
                   edgecolors='white',
                   linewidth=PlotConfig.DATA_LINEWIDTH,
                   marker=marker,
                   label='Data Points')
        ax.scatter(outliers_x,
                   outliers_y,
                   alpha=PlotConfig.OUTLIER_ALPHA,
                   s=PlotConfig.OUTLIER_POINT_SIZE,
                   c=outlier_color,
                   marker=outlier_marker,
                   label='Removed Outliers')
    else:
        # No outliers plot all data points
        ax.scatter(x,
                   y,
                   alpha=PlotConfig.DATA_ALPHA,
                   s=PlotConfig.DATA_POINT_SIZE,
                   c=color,
                   edgecolors='white',
                   linewidth=PlotConfig.DATA_LINEWIDTH,
                   marker=marker,
                   label='Data Points')

    # Plot trend line using the fit
    x_fit = np.linspace(x.min(), x.max(), PlotConfig.TREND_LINE_POINTS)
    y_fit = fit.slope * x_fit + fit.intercept
    ax.plot(x_fit,
            y_fit,
            trend_linestyle,
            alpha=trend_alpha,
            linewidth=trend_linewidth,
            color=trend_color,
            label=f'Trend (slope={fit.slope:.4f}, R²={fit.r_squared:.3f})')

    if sla_value > 0:
        ax.axhline(y=sla_value,
                   color=PlotConfig.SLA_COLOR,
                   linestyle=':',
                   alpha=PlotConfig.SLA_ALPHA,
                   linewidth=2,
                   label=sla_label or f'SLA Threshold ({sla_value}s)')

    ax.set_xlabel('Concurrency', fontsize=PlotConfig.AXIS_LABEL_FONTSIZE, fontweight=PlotConfig.AXIS_LABEL_FONTWEIGHT)
    ax.set_ylabel(y_label, fontsize=PlotConfig.AXIS_LABEL_FONTSIZE, fontweight=PlotConfig.AXIS_LABEL_FONTWEIGHT)
    ax.set_title(title, fontsize=PlotConfig.TITLE_FONTSIZE, fontweight=PlotConfig.TITLE_FONTWEIGHT)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)
    ax.legend(fontsize=legend_fontsize)
    if outliers_note:
        ax.text(PlotConfig.NOTE_X_POS,
                PlotConfig.NOTE_Y_POS,
                outliers_note,
                transform=ax.transAxes,
                fontsize=PlotConfig.NOTE_FONTSIZE,
                color=note_text_color,
                ha='right',
                va='bottom',
                bbox=dict(boxstyle=f'round,pad={PlotConfig.NOTE_BOX_PAD}',
                          facecolor=note_box_color,
                          alpha=PlotConfig.NOTE_BOX_ALPHA))

    return True


def plot_concurrency_vs_time_metrics(metrics_per_concurrency: dict[int, SizingMetrics],
                                     output_dir: Path,
                                     target_llm_latency: float = 0.0,
                                     target_runtime: float = 0.0,
                                     llm_latency_fit: LinearFitResult | None = None,
                                     runtime_fit: LinearFitResult | None = None) -> None:
    """
    Plot concurrency vs. p95 latency and workflow runtime using metrics_per_concurrency.
    Enhanced with better styling, trend analysis, and annotations.
    Only plots valid runs and requires pre-computed fits.
    """
    rows = []

    for concurrency, metrics in metrics_per_concurrency.items():
        llm_latency = metrics.llm_latency_p95
        workflow_runtime = metrics.workflow_runtime_p95

        rows.append({
            "concurrency": concurrency, "llm_latency_p95": llm_latency, "workflow_runtime_p95": workflow_runtime
        })

    if not rows:
        logger.warning("No valid metrics data available to plot.")
        return

    plt.style.use('seaborn-v0_8')
    df = pd.DataFrame(rows).sort_values("concurrency")

    # Always generate simple plot first
    plot_concurrency_vs_time_metrics_simple(df, output_dir)

    # Check if we have fits available for analysis plots
    has_llm_latency_fit = llm_latency_fit is not None
    has_runtime_fit = runtime_fit is not None

    if not has_llm_latency_fit and not has_runtime_fit:
        logger.warning("No linear fits available for analysis plots, skipping enhanced plot")
        return

    # Create subplots based on available fits
    if has_llm_latency_fit and has_runtime_fit:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PlotConfig.ENHANCED_FIGSIZE)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None

    # Plot llm_latency if fit is available
    llm_latency_plotted = False
    if has_llm_latency_fit:
        llm_latency_plotted = plot_metric_vs_concurrency_with_optional_fit(
            ax1,
            df["concurrency"].to_numpy(),
            df["llm_latency_p95"].to_numpy(),
            metric_name="llm_latency",
            y_label='P95 LLM Latency (seconds)',
            title='Concurrency vs P95 LLM Latency',
            color=PlotConfig.LLM_LATENCY_COLOR,
            sla_value=target_llm_latency,
            sla_label=f'SLA Threshold ({target_llm_latency}s)' if target_llm_latency > 0 else None,
            fit=llm_latency_fit,
        )

    # Plot runtime if fit is available
    runtime_plotted = False
    if has_runtime_fit and ax2 is not None:
        runtime_plotted = plot_metric_vs_concurrency_with_optional_fit(
            ax2,
            df["concurrency"].to_numpy(),
            df["workflow_runtime_p95"].to_numpy(),
            metric_name="runtime",
            y_label='P95 Workflow Runtime (seconds)',
            title='Concurrency vs P95 Workflow Runtime',
            color=PlotConfig.RUNTIME_COLOR,
            sla_value=target_runtime,
            sla_label=f'SLA Threshold ({target_runtime}s)' if target_runtime > 0 else None,
            fit=runtime_fit,
        )

    # Check if any plots were successfully created
    plots_created = (llm_latency_plotted or runtime_plotted)

    if not plots_created:
        logger.warning("No analysis plots could be created, skipping enhanced plot")
        plt.close(fig)
        return

    # Add summary statistics
    stats_text = f'Data Points: {len(df)}\n'
    stats_text += f'LLM Latency Range: {df["llm_latency_p95"].min():.3f}-{df["llm_latency_p95"].max():.3f}s\n'
    stats_text += f'WF Runtime Range: {df["workflow_runtime_p95"].min():.3f}-{df["workflow_runtime_p95"].max():.3f}s'

    fig.text(PlotConfig.STATS_X_POS,
             PlotConfig.STATS_Y_POS,
             stats_text,
             fontsize=PlotConfig.STATS_FONTSIZE,
             bbox=dict(boxstyle=f'round,pad={PlotConfig.STATS_BOX_PAD}',
                       facecolor=PlotConfig.STATS_BOX_COLOR,
                       alpha=PlotConfig.STATS_BOX_ALPHA))

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    enhanced_plot_path = output_dir / "concurrency_vs_p95_analysis.png"
    plt.savefig(enhanced_plot_path,
                dpi=PlotConfig.ENHANCED_DPI,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    logger.info("Enhanced plot saved to %s", enhanced_plot_path)

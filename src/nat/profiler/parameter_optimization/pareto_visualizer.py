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
# flake8: noqa: W293

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


class ParetoVisualizer:

    def __init__(self, metric_names: list[str], directions: list[str], title_prefix: str = "Optimization Results"):
        self.metric_names = metric_names
        self.directions = directions
        self.title_prefix = title_prefix

        if len(metric_names) != len(directions):
            raise ValueError("Number of metric names must match number of directions")

    def plot_pareto_front_2d(self,
                             trials_df: pd.DataFrame,
                             pareto_trials_df: pd.DataFrame | None = None,
                             save_path: Path | None = None,
                             figsize: tuple[int, int] = (10, 8),
                             show_plot: bool = True) -> plt.Figure:
        if len(self.metric_names) != 2:
            raise ValueError("2D Pareto front visualization requires exactly 2 metrics")

        fig, ax = plt.subplots(figsize=figsize)

        # Extract metric values - support both old (values_0) and new (values_metricname) formats
        x_col = f"values_{self.metric_names[0]}" \
            if f"values_{self.metric_names[0]}" in trials_df.columns else f"values_{0}"
        y_col = f"values_{self.metric_names[1]}"\
            if f"values_{self.metric_names[1]}" in trials_df.columns else f"values_{1}"
        x_vals = trials_df[x_col].values
        y_vals = trials_df[y_col].values

        # Plot all trials
        ax.scatter(x_vals,
                   y_vals,
                   alpha=0.6,
                   s=50,
                   c='lightblue',
                   label=f'All Trials (n={len(trials_df)})',
                   edgecolors='navy',
                   linewidths=0.5)

        # Plot Pareto optimal trials if provided
        if pareto_trials_df is not None and not pareto_trials_df.empty:
            pareto_x = pareto_trials_df[x_col].values
            pareto_y = pareto_trials_df[y_col].values

            ax.scatter(pareto_x,
                       pareto_y,
                       alpha=0.9,
                       s=100,
                       c='red',
                       label=f'Pareto Optimal (n={len(pareto_trials_df)})',
                       edgecolors='darkred',
                       linewidths=1.5,
                       marker='*')

            # Add trial number labels to Pareto optimal points
            for idx in range(len(pareto_trials_df)):
                trial_number = pareto_trials_df.iloc[idx]['number'] \
                    if 'number' in pareto_trials_df.columns else pareto_trials_df.index[idx]
                ax.annotate(f'{int(trial_number)}',
                            xy=(pareto_x[idx], pareto_y[idx]),
                            xytext=(8, 8),
                            textcoords='offset points',
                            fontsize=9,
                            fontweight='bold',
                            color='darkred',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))

            # Draw Pareto front line (only for 2D)
            if len(pareto_x) > 1:
                # Sort points for line drawing based on first objective
                sorted_indices = np.argsort(pareto_x)
                ax.plot(pareto_x[sorted_indices],
                        pareto_y[sorted_indices],
                        'r--',
                        alpha=0.7,
                        linewidth=2,
                        label='Pareto Front')

        # Customize plot
        x_direction = "↓" if self.directions[0] == "minimize" else "↑"
        y_direction = "↓" if self.directions[1] == "minimize" else "↑"

        ax.set_xlabel(f"{self.metric_names[0]} {x_direction}", fontsize=12)
        ax.set_ylabel(f"{self.metric_names[1]} {y_direction}", fontsize=12)
        ax.set_title(f"{self.title_prefix}: Pareto Front Visualization", fontsize=14, fontweight='bold')

        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        # Add direction annotations
        x_annotation = (f"Better {self.metric_names[0]} ←"
                        if self.directions[0] == "minimize" else f"→ Better {self.metric_names[0]}")
        ax.annotate(x_annotation,
                    xy=(0.02, 0.98),
                    xycoords='axes fraction',
                    ha='left',
                    va='top',
                    fontsize=10,
                    style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

        y_annotation = (f"Better {self.metric_names[1]} ↓"
                        if self.directions[1] == "minimize" else f"Better {self.metric_names[1]} ↑")
        ax.annotate(y_annotation,
                    xy=(0.02, 0.02),
                    xycoords='axes fraction',
                    ha='left',
                    va='bottom',
                    fontsize=10,
                    style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("2D Pareto plot saved to: %s", save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_pareto_parallel_coordinates(self,
                                         trials_df: pd.DataFrame,
                                         pareto_trials_df: pd.DataFrame | None = None,
                                         save_path: Path | None = None,
                                         figsize: tuple[int, int] = (12, 8),
                                         show_plot: bool = True) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)

        n_metrics = len(self.metric_names)
        x_positions = np.arange(n_metrics)

        # Normalize values for better visualization
        all_values = []
        for i in range(n_metrics):
            # Support both old (values_0) and new (values_metricname) formats
            col_name = f"values_{self.metric_names[i]}"\
                if f"values_{self.metric_names[i]}" in trials_df.columns else f"values_{i}"
            all_values.append(trials_df[col_name].values)

        # Normalize each metric to [0, 1] for parallel coordinates
        normalized_values = []
        for i, values in enumerate(all_values):
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                if self.directions[i] == "minimize":
                    # For minimize: lower values get higher normalized scores
                    norm_vals = 1 - (values - min_val) / (max_val - min_val)
                else:
                    # For maximize: higher values get higher normalized scores
                    norm_vals = (values - min_val) / (max_val - min_val)
            else:
                norm_vals = np.ones_like(values) * 0.5
            normalized_values.append(norm_vals)

        # Plot all trials
        for i in range(len(trials_df)):
            trial_values = [normalized_values[j][i] for j in range(n_metrics)]
            ax.plot(x_positions, trial_values, 'b-', alpha=0.1, linewidth=1)

        # Plot Pareto optimal trials
        if pareto_trials_df is not None and not pareto_trials_df.empty:
            pareto_indices = pareto_trials_df.index
            for idx in pareto_indices:
                if idx < len(trials_df):
                    trial_values = [normalized_values[j][idx] for j in range(n_metrics)]
                    ax.plot(x_positions, trial_values, 'r-', alpha=0.8, linewidth=3)

                    # Add trial number label at the rightmost point
                    trial_number = trials_df.iloc[idx]['number'] if 'number' in trials_df.columns else idx
                    # Position label slightly to the right and above the last point
                    ax.annotate(f'{int(trial_number)}',
                                xy=(x_positions[-1], trial_values[-1]),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=9,
                                fontweight='bold',
                                color='darkred',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

        # Customize plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{name}\n({direction})" for name, direction in zip(self.metric_names, self.directions)])
        ax.set_ylabel("Normalized Performance (Higher is Better)", fontsize=12)
        ax.set_title(f"{self.title_prefix}: Parallel Coordinates Plot", fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Add legend
        legend_elements = [
            Line2D([0], [0], color='blue', alpha=0.3, linewidth=2, label='All Trials'),
            Line2D([0], [0], color='red', alpha=0.8, linewidth=3, label='Pareto Optimal'),
            Patch(facecolor='white', edgecolor='red', label='[n]: trial number')
        ]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Parallel coordinates plot saved to: %s", save_path)

        if show_plot:
            plt.show()

        return fig

    def plot_pairwise_matrix(self,
                             trials_df: pd.DataFrame,
                             pareto_trials_df: pd.DataFrame | None = None,
                             save_path: Path | None = None,
                             figsize: tuple[int, int] | None = None,
                             show_plot: bool = True) -> plt.Figure:
        n_metrics = len(self.metric_names)
        if figsize is None:
            figsize = (4 * n_metrics, 4 * n_metrics)

        fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)
        fig.suptitle(f"{self.title_prefix}: Pairwise Metric Comparison", fontsize=16, fontweight='bold')

        for i in range(n_metrics):
            for j in range(n_metrics):
                ax = axes[i, j] if n_metrics > 1 else axes

                if i == j:
                    # Diagonal: histograms
                    # Support both old (values_0) and new (values_metricname) formats
                    col_name = f"values_{self.metric_names[i]}"\
                        if f"values_{self.metric_names[i]}" in trials_df.columns else f"values_{i}"
                    values = trials_df[col_name].values
                    ax.hist(values, bins=20, alpha=0.7, color='lightblue', edgecolor='navy')
                    if pareto_trials_df is not None and not pareto_trials_df.empty:
                        pareto_values = pareto_trials_df[col_name].values
                        ax.hist(pareto_values, bins=20, alpha=0.8, color='red', edgecolor='darkred')
                    ax.set_xlabel(f"{self.metric_names[i]}")
                    ax.set_ylabel("Frequency")
                else:
                    # Off-diagonal: scatter plots
                    # Support both old (values_0) and new (values_metricname) formats
                    x_col = f"values_{self.metric_names[j]}"\
                        if f"values_{self.metric_names[j]}" in trials_df.columns else f"values_{j}"
                    y_col = f"values_{self.metric_names[i]}"\
                        if f"values_{self.metric_names[i]}" in trials_df.columns else f"values_{i}"
                    x_vals = trials_df[x_col].values
                    y_vals = trials_df[y_col].values

                    ax.scatter(x_vals, y_vals, alpha=0.6, s=30, c='lightblue', edgecolors='navy', linewidths=0.5)

                    if pareto_trials_df is not None and not pareto_trials_df.empty:
                        pareto_x = pareto_trials_df[x_col].values
                        pareto_y = pareto_trials_df[y_col].values
                        ax.scatter(pareto_x,
                                   pareto_y,
                                   alpha=0.9,
                                   s=60,
                                   c='red',
                                   edgecolors='darkred',
                                   linewidths=1,
                                   marker='*')

                        # Add trial number labels to Pareto optimal points
                        for idx in range(len(pareto_trials_df)):
                            trial_number = pareto_trials_df.iloc[idx]['number'] \
                                if 'number' in pareto_trials_df.columns else pareto_trials_df.index[idx]
                            ax.annotate(f'{int(trial_number)}',
                                        xy=(pareto_x[idx], pareto_y[idx]),
                                        xytext=(6, 6),
                                        textcoords='offset points',
                                        fontsize=8,
                                        fontweight='bold',
                                        color='darkred',
                                        bbox=dict(boxstyle='round,pad=0.2',
                                                  facecolor='white',
                                                  edgecolor='red',
                                                  alpha=0.8))

                    ax.set_xlabel(f"{self.metric_names[j]} ({self.directions[j]})")
                    ax.set_ylabel(f"{self.metric_names[i]} ({self.directions[i]})")

                ax.grid(True, alpha=0.3)

        # Add legend to the figure
        legend_elements = [
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   markerfacecolor='lightblue',
                   markeredgecolor='navy',
                   markersize=8,
                   alpha=0.6,
                   label='All Trials'),
            Line2D([0], [0],
                   marker='*',
                   color='w',
                   markerfacecolor='red',
                   markeredgecolor='darkred',
                   markersize=10,
                   alpha=0.9,
                   label='Pareto Optimal'),
            Patch(facecolor='white', edgecolor='red', label='[n]: trial number')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.9, fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Pairwise matrix plot saved to: %s", save_path)

        if show_plot:
            plt.show()

        return fig


def load_trials_from_study(study: optuna.Study) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Get all trials
    trials_df = study.trials_dataframe()

    # Get Pareto optimal trials
    pareto_trials = study.best_trials
    pareto_trial_numbers = [trial.number for trial in pareto_trials]
    pareto_trials_df = trials_df[trials_df['number'].isin(pareto_trial_numbers)]

    return trials_df, pareto_trials_df


def load_trials_from_csv(csv_path: Path, metric_names: list[str],
                         directions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials_df = pd.read_csv(csv_path)

    # Extract values columns
    value_cols = [col for col in trials_df.columns if col.startswith('values_')]
    if not value_cols:
        raise ValueError("CSV file must contain 'values_' columns with metric scores")

    # Compute Pareto optimal solutions manually
    pareto_mask = compute_pareto_optimal_mask(trials_df, value_cols, directions)
    pareto_trials_df = trials_df[pareto_mask]

    return trials_df, pareto_trials_df


def compute_pareto_optimal_mask(df: pd.DataFrame, value_cols: list[str], directions: list[str]) -> np.ndarray:
    values = df[value_cols].values
    n_trials = len(values)

    # Normalize directions: convert all to maximization
    normalized_values = values.copy()
    for i, direction in enumerate(directions):
        if direction == "minimize":
            normalized_values[:, i] = -normalized_values[:, i]

    is_pareto = np.ones(n_trials, dtype=bool)

    for i in range(n_trials):
        if is_pareto[i]:
            # Compare with all other solutions
            dominates = np.all(normalized_values[i] >= normalized_values, axis=1) & \
                       np.any(normalized_values[i] > normalized_values, axis=1)
            is_pareto[dominates] = False

    return is_pareto


def create_pareto_visualization(data_source: optuna.Study | Path | pd.DataFrame,
                                metric_names: list[str],
                                directions: list[str],
                                output_dir: Path | None = None,
                                title_prefix: str = "Optimization Results",
                                show_plots: bool = True) -> dict[str, plt.Figure]:
    # Load data based on source type
    if hasattr(data_source, 'trials_dataframe'):
        # Optuna study object
        trials_df, pareto_trials_df = load_trials_from_study(data_source)
    elif isinstance(data_source, str | Path):
        # CSV file path
        trials_df, pareto_trials_df = load_trials_from_csv(Path(data_source), metric_names, directions)
    elif isinstance(data_source, pd.DataFrame):
        # DataFrame
        trials_df = data_source
        value_cols = [col for col in trials_df.columns if col.startswith('values_')]
        pareto_mask = compute_pareto_optimal_mask(trials_df, value_cols, directions)
        pareto_trials_df = trials_df[pareto_mask]
    else:
        raise ValueError("data_source must be an Optuna study, CSV file path, or pandas DataFrame")

    visualizer = ParetoVisualizer(metric_names, directions, title_prefix)
    figures = {}

    logger.info("Creating Pareto front visualizations...")
    logger.info("Total trials: %d", len(trials_df))
    logger.info("Pareto optimal trials: %d", len(pareto_trials_df))

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if len(metric_names) == 2:
            # 2D scatter plot
            save_path = output_dir / "pareto_front_2d.png" if output_dir else None
            fig = visualizer.plot_pareto_front_2d(trials_df, pareto_trials_df, save_path, show_plot=show_plots)
            figures["2d_scatter"] = fig

        if len(metric_names) >= 2:
            # Parallel coordinates plot
            save_path = output_dir / "pareto_parallel_coordinates.png" if output_dir else None
            fig = visualizer.plot_pareto_parallel_coordinates(trials_df,
                                                              pareto_trials_df,
                                                              save_path,
                                                              show_plot=show_plots)
            figures["parallel_coordinates"] = fig

            # Pairwise matrix plot
            save_path = output_dir / "pareto_pairwise_matrix.png" if output_dir else None
            fig = visualizer.plot_pairwise_matrix(trials_df, pareto_trials_df, save_path, show_plot=show_plots)
            figures["pairwise_matrix"] = fig

        logger.info("Visualization complete!")
        if output_dir:
            logger.info("Plots saved to: %s", output_dir)

    except Exception as e:
        logger.error("Error creating visualizations: %s", e)
        raise

    return figures

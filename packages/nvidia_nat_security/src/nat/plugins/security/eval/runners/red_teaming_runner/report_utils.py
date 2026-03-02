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

logger = logging.getLogger(__name__)


def generate_and_save_report(
    flat_results: list[dict[str, Any]] | pd.DataFrame,
    output_dir: Path,
    summary: dict[str, Any] | None = None,
) -> Path | None:
    """Generate and save a simple HTML report from flat results.

    This helper intentionally keeps plotting dependencies optional.
    If plotly is unavailable, the function logs and returns ``None`` without failing eval.
    """
    _ = summary  # reserved for richer report rendering
    if isinstance(flat_results, pd.DataFrame):
        df = flat_results.copy()
    else:
        if not flat_results:
            logger.warning("No results to plot")
            return None
        df = pd.DataFrame(flat_results)

    if "error_message" in df.columns:
        error_count = int(df["error_message"].notna().sum())
        if error_count > 0:
            logger.info("Dropping %d rows with error_message from plotting", error_count)
            df = df[df["error_message"].isna()]

    if df.empty:
        logger.warning("No valid results to plot after filtering errors")
        return None

    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        logger.warning("Skipping red-team HTML report generation: optional dependency 'plotly' is not installed")
        return None

    report_path = output_dir / "report.html"
    output_dir.mkdir(parents=True, exist_ok=True)

    if "scenario_id" in df.columns and "score" in df.columns:
        fig = px.box(df, x="scenario_id", y="score", points="all", title="Risk Score Distribution by Scenario")
    elif "score" in df.columns:
        fig = px.histogram(df, x="score", nbins=20, title="Risk Score Distribution")
    else:
        logger.warning("Skipping red-team HTML report generation: no 'score' column in results")
        return None

    html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    report_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Red Teaming Evaluation Results</title></head><body>"
        f"<h1>Red Teaming Evaluation Results for run: {output_dir.name}</h1>{html}</body></html>"
    )
    report_path.write_text(report_html, encoding="utf-8")
    logger.debug("Saved red-team report: %s", report_path)
    return report_path


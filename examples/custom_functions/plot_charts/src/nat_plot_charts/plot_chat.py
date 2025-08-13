# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_data_from_file(file_path: str) -> dict[str, Any]:
    """Load data from a JSON file."""
    try:
        if not os.path.isabs(file_path):
            # If relative path, try to find it in common locations
            search_paths = [
                file_path,
                os.path.join(os.getcwd(), file_path),
                os.path.join(os.path.dirname(__file__), "..", "..", file_path),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", file_path),
            ]

            for search_path in search_paths:
                if os.path.exists(search_path):
                    file_path = search_path
                    break
            else:
                raise FileNotFoundError(f"Could not find data file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Successfully loaded data from %s", file_path)
        return data
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, str(e))
        raise


def create_line_plot(data: dict[str, Any], output_path: str, figure_size: tuple[int, int]) -> str:
    """Create a line plot from the data."""
    fig, ax = plt.subplots(figsize=figure_size)

    x_values = data.get("xValues", [])
    y_values = data.get("yValues", [])

    for series in y_values:
        label = series.get("label", "Series")
        series_data = series.get("data", [])
        ax.plot(x_values, series_data, marker='o', label=label, linewidth=2)

    ax.set_xlabel("X Values")
    ax.set_ylabel("Y Values")
    ax.set_title("Line Chart")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_bar_plot(data: dict[str, Any], output_path: str, figure_size: tuple[int, int]) -> str:
    """Create a bar plot from the data."""
    import numpy as np

    fig, ax = plt.subplots(figsize=figure_size)

    x_values = data.get("xValues", [])
    y_values = data.get("yValues", [])

    if not y_values:
        raise ValueError("No data series found for plotting")

    x_pos = np.arange(len(x_values))
    width = 0.8 / len(y_values)

    for i, series in enumerate(y_values):
        label = series.get("label", f"Series {i+1}")
        series_data = series.get("data", [])
        offset = (i - len(y_values) / 2 + 0.5) * width
        ax.bar(x_pos + offset, series_data, width, label=label)

    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")
    ax.set_title("Bar Chart")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_scatter_plot(data: dict[str, Any], output_path: str, figure_size: tuple[int, int]) -> str:
    """Create a scatter plot from the data."""
    fig, ax = plt.subplots(figsize=figure_size)

    x_values = data.get("xValues", [])
    y_values = data.get("yValues", [])

    # Convert x_values to numeric if they're strings representing numbers
    try:
        x_numeric = [float(x) for x in x_values]
    except (ValueError, TypeError):
        # If conversion fails, use index positions
        x_numeric = list(range(len(x_values)))

    for series in y_values:
        label = series.get("label", "Series")
        series_data = series.get("data", [])
        ax.scatter(x_numeric, series_data, label=label, s=100, alpha=0.7)

    ax.set_xlabel("X Values")
    ax.set_ylabel("Y Values")
    ax.set_title("Scatter Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def determine_chart_type(user_request: str, available_types: list[str]) -> str:
    """Determine the best chart type based on user request."""
    request_lower = user_request.lower()

    # Simple keyword matching for chart type detection
    if any(word in request_lower for word in ["line", "trend", "over time", "timeline"]):
        return "line" if "line" in available_types else available_types[0]
    elif any(word in request_lower for word in ["bar", "column", "compare", "comparison"]):
        return "bar" if "bar" in available_types else available_types[0]
    elif any(word in request_lower for word in ["scatter", "correlation", "relationship"]):
        return "scatter" if "scatter" in available_types else available_types[0]

    # Default to first available type
    return available_types[0] if available_types else "line"


async def generate_chart_description(llm: BaseChatModel, data: dict[str, Any], chart_type: str) -> str:
    """Generate a description of the chart using the LLM."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        "Based on the following data, provide a brief description of what the {chart_type} chart shows:\n\n"
        "Data: {data}\n\n"
        "Please provide a 1-2 sentence description focusing on the key insights or patterns visible in the data.")

    try:
        chain = prompt | llm
        response = await chain.ainvoke({"data": json.dumps(data, indent=2), "chart_type": chart_type})

        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            else:
                return str(content).strip()
        else:
            return str(response).strip()
    except Exception as e:
        logger.warning("Failed to generate chart description: %s", str(e))
        return f"Generated {chart_type} chart from the provided data."

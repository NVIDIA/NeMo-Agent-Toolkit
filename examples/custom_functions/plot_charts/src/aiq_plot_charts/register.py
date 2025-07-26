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

import logging
from pathlib import Path

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PlotChartsWorkflowConfig(FunctionBaseConfig, name="plot_charts"):
    """Configuration for the plot charts workflow."""
    llm_name: str
    data_file_path: str = "example_data.json"
    output_directory: str = "outputs"
    chart_types: list[str] = ["line", "bar", "scatter"]
    max_data_points: int = 100
    figure_size: tuple[int, int] = (10, 6)


@register_function(config_type=PlotChartsWorkflowConfig)
async def plot_charts_function(config: PlotChartsWorkflowConfig, builder: Builder):
    """
    Create charts from data based on user requests.

    This function can generate line charts, bar charts, and scatter plots
    from JSON data files based on user instructions.
    """

    from aiq_plot_charts.plot_chat import create_bar_plot
    from aiq_plot_charts.plot_chat import create_line_plot
    from aiq_plot_charts.plot_chat import create_scatter_plot
    from aiq_plot_charts.plot_chat import determine_chart_type
    from aiq_plot_charts.plot_chat import generate_chart_description
    from aiq_plot_charts.plot_chat import load_data_from_file

    # Get the LLM from builder configuration
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Ensure output directory exists
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _create_chart(user_request: str) -> str:
        """Internal function to create charts based on user requests."""
        logger.info("Processing chart request: %s", user_request)

        try:
            # Load data from configured file
            data = load_data_from_file(config.data_file_path)

            # Validate data structure
            if not data.get("xValues") or not data.get("yValues"):
                return "Error: Data file must contain 'xValues' and 'yValues' fields."

            # Check data size limits
            total_points = len(data["xValues"]) * len(data["yValues"])
            if total_points > config.max_data_points:
                return (f"Error: Data contains {total_points} points, which exceeds the limit of "
                        f"{config.max_data_points}.")

            # Determine chart type from user request
            chart_type = determine_chart_type(user_request, config.chart_types)
            logger.info("Selected chart type: %s", chart_type)

            # Generate unique filename
            import time
            timestamp = int(time.time())
            filename = f"{chart_type}_chart_{timestamp}.png"
            output_path = output_dir / filename

            # Create the appropriate chart
            if chart_type == "line":
                saved_path = create_line_plot(data, str(output_path), config.figure_size)
            elif chart_type == "bar":
                saved_path = create_bar_plot(data, str(output_path), config.figure_size)
            elif chart_type == "scatter":
                saved_path = create_scatter_plot(data, str(output_path), config.figure_size)
            else:
                return (f"Error: Unsupported chart type '{chart_type}'. "
                        f"Available types: {config.chart_types}")

            # Generate description using LLM
            description = await generate_chart_description(llm, data, chart_type)

            logger.info("Successfully created chart: %s", saved_path)

            return (f"Successfully created {chart_type} chart saved to: {saved_path}\n\n"
                    f"Chart description: {description}")

        except FileNotFoundError as e:
            logger.error("Data file not found: %s", str(e))
            return (f"Error: Could not find data file at '{config.data_file_path}'. "
                    f"Please check the file path in your configuration.")
        except Exception as e:
            logger.error("Error creating chart: %s", str(e))
            return f"Error creating chart: {str(e)}"

    # Return the function as a FunctionInfo
    yield FunctionInfo.from_fn(
        _create_chart,
        description=("Creates charts (line, bar, or scatter plots) from data based on user requests. "
                     f"Supports chart types: {', '.join(config.chart_types)}. "
                     f"Data is loaded from: {config.data_file_path}"))

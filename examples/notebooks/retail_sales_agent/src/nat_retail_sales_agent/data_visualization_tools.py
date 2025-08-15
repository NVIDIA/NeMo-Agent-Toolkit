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
"""
Data visualization tools for retail sales analysis.
"""
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class PlotSalesTrendForStoresConfig(FunctionBaseConfig, name="plot_sales_trend_for_stores"):
    """Plot sales trend for a specific store."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=PlotSalesTrendForStoresConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plot_sales_trend_for_stores_function(config: PlotSalesTrendForStoresConfig, _builder: Builder):
    """Create a visualization of sales trends over time."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _plot_sales_trend_for_stores(store_id: str) -> str:
        """
        Create a line chart showing sales trends over time.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            product_name: Optional product name to filter by

        Returns:
            Dictionary containing chart data and image
        """
        if store_id not in df["StoreID"].unique():
            data = df
            title = "Sales Trend for All Stores"
        else:
            data = df[df["StoreID"] == store_id]
            title = f"Sales Trend for Store {store_id}"

        plt.figure(figsize=(10, 5))
        trend = data.groupby("Date")["Revenue"].sum()
        trend.plot(title=title)
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.tight_layout()
        plt.savefig("sales_trend.png")

        return "Sales trend plot saved to sales_trend.png"

    yield FunctionInfo.from_fn(
        _plot_sales_trend_for_stores,
        description=(
            "This tool can be used to plot the sales trend for a specific store or all stores. "
            "It takes in a store ID creates and saves an image of a plot of the revenue trend for that store."))


class PlotAndCompareRevenueAcrossStoresConfig(FunctionBaseConfig, name="plot_and_compare_revenue_across_stores"):
    """Plot and compare revenue across stores."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=PlotAndCompareRevenueAcrossStoresConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plot_revenue_across_stores_function(config: PlotAndCompareRevenueAcrossStoresConfig, _builder: Builder):
    """Create a visualization comparing sales trends between stores."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _plot_revenue_across_stores(_input_message: str) -> str:
        """
        Create a multi-line chart comparing sales trends between stores.

        Args:
            input_message: Input message to plot the revenue across stores

        Returns:
            Dictionary containing comparison chart data and image
        """
        pivot = df.pivot_table(index="Date", columns="StoreID", values="Revenue", aggfunc="sum")
        pivot.plot(figsize=(12, 6), title="Revenue Trends Across Stores")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.legend(title="StoreID")
        plt.tight_layout()
        plt.savefig("revenue_across_stores.png")

        return "Revenue trends across stores plot saved to revenue_across_stores.png"

    yield FunctionInfo.from_fn(
        _plot_revenue_across_stores,
        description=(
            "This tool can be used to plot and compare the revenue trends across stores. Use this tool only if the "
            "user asks for a comparison of revenue trends across stores."
            "It takes in an input message and creates and saves an image of a plot of the revenue trends across stores."
        ))


class PlotAverageDailyRevenueConfig(FunctionBaseConfig, name="plot_average_daily_revenue"):
    """Plot average daily revenue for stores and products."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=PlotAverageDailyRevenueConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def plot_average_daily_revenue_function(config: PlotAverageDailyRevenueConfig, _builder: Builder):
    """Create a bar chart showing average daily revenue by day of week."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _plot_average_daily_revenue(_input_message: str) -> str:
        """
        Create a bar chart showing average revenue by day of the week.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary containing revenue chart data and image
        """
        daily_revenue = df.groupby(["StoreID", "Product", "Date"])["Revenue"].sum().reset_index()

        avg_daily_revenue = daily_revenue.groupby(["StoreID", "Product"])["Revenue"].mean().unstack()

        avg_daily_revenue.plot(kind="bar", figsize=(12, 6), title="Average Daily Revenue per Store by Product")
        plt.ylabel("Average Revenue")
        plt.xlabel("Store ID")
        plt.xticks(rotation=0)
        plt.legend(title="Product", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("average_daily_revenue.png")

        return "Average daily revenue plot saved to average_daily_revenue.png"

    yield FunctionInfo.from_fn(
        _plot_average_daily_revenue,
        description=("This tool can be used to plot the average daily revenue for stores and products "
                     "It takes in an input message and creates and saves an image of a grouped bar chart "
                     "of the average daily revenue"))


class GraphSummarizerConfig(FunctionBaseConfig, name="graph_summarizer"):
    """Analyze and summarize chart data."""
    llm_name: LLMRef = Field(description="The name of the LLM to use for the graph summarizer.")


@register_function(config_type=GraphSummarizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def graph_summarizer_function(config: GraphSummarizerConfig, builder: Builder):
    """Analyze chart data and provide natural language summaries."""
    import base64

    from openai import OpenAI

    client = OpenAI()

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _graph_summarizer(image_path: str) -> str:
        """
        Analyze chart data and provide insights and summaries.

        Args:
            image_path: The path to the image to analyze

        Returns:
            Dictionary containing analysis and insights
        """

        def encode_image(image_path: str):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)

        response = client.responses.create(
            model=llm.model_name,
            input=[{
                "role":
                    "user",
                "content": [{
                    "type": "input_text",
                    "text": "Please summarize the key insights from this graph in natural language."
                }, {
                    "type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"
                }]
            }],
            temperature=0.3,
        )

        return response.output_text

    yield FunctionInfo.from_fn(
        _graph_summarizer,
        description=("This tool can be used to summarize the key insights from a graph in natural language. "
                     "It takes in the path to an image and returns a summary of the key insights from the graph."))

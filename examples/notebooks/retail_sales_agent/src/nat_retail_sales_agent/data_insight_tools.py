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
Data insight tools for retail sales analysis.
"""
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GetTotalProductSalesDataConfig(FunctionBaseConfig, name="get_total_product_sales_data"):
    """Get total sales data by product."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=GetTotalProductSalesDataConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_total_product_sales_data_function(config: GetTotalProductSalesDataConfig, _builder: Builder):
    """Get total sales data for a specific product."""
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _get_total_product_sales_data(product_name: str) -> str:
        """
        Retrieve total sales data for a specific product.

        Args:
            product_name: Name of the product

        Returns:
            String message containing total sales data
        """
        df['Product'] = df["Product"].apply(lambda x: x.lower())
        revenue = df[df['Product'] == product_name]['Revenue'].sum()
        units_sold = df[df['Product'] == product_name]['UnitsSold'].sum()

        return f"Revenue for {product_name} are {revenue} and total units sold are {units_sold}"

    yield FunctionInfo.from_fn(
        _get_total_product_sales_data,
        description=("This tool can be used to get the total sales data for a specific product. "
                     "It takes in a product name and returns the total sales data for that product."))


class GetSalesPerDayConfig(FunctionBaseConfig, name="get_sales_per_day"):
    """Get total sales across all products per day."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=GetSalesPerDayConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def get_sales_per_day_function(config: GetSalesPerDayConfig, builder: Builder):
    """Get total sales across all products per day."""
    import pandas as pd

    df = pd.read_csv(config.data_path)
    df['Product'] = df["Product"].apply(lambda x: x.lower())

    async def _get_sales_per_day(date: str, product: str) -> str:
        """
        Calculate total sales data across all products for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            product: Product name

        Returns:
            String message with the total sales for the day
        """
        if date == "None":
            return "Please provide a date in YYYY-MM-DD format."
        total_revenue = df[(df['Date'] == date) & (df['Product'] == product)]['Revenue'].sum()
        total_units_sold = df[(df['Date'] == date) & (df['Product'] == product)]['UnitsSold'].sum()

        return f"Total revenue for {date} is {total_revenue} and total units sold is {total_units_sold}"

    yield FunctionInfo.from_fn(
        _get_sales_per_day,
        description=(
            "This tool can be used to calculate the total sales across all products per day. "
            "It takes in a date in YYYY-MM-DD format and a product name and returns the total sales for that product "
            "on that day."))


class DetectOutliersIQRConfig(FunctionBaseConfig, name="detect_outliers_iqr"):
    """Detect outliers in sales data using IQR method."""
    data_path: str = Field(description="Path to the data file")


@register_function(config_type=DetectOutliersIQRConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def detect_outliers_iqr_function(config: DetectOutliersIQRConfig, _builder: Builder):
    """Detect outliers in sales data using the Interquartile Range (IQR) method."""
    import pandas as pd

    df = pd.read_csv(config.data_path)

    async def _detect_outliers_iqr(metric: str) -> str:
        """
        Detect outliers in retail data using the IQR method.

        Args:
            metric: Specific metric to check for outliers

        Returns:
            Dictionary containing outlier analysis results
        """
        if metric == "None":
            column = "Revenue"
        else:
            column = metric

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)]

        return f"Outliers in {column} are {outliers.to_dict('records')}"

    yield FunctionInfo.from_fn(
        _detect_outliers_iqr,
        description=("Detect outliers in retail data using the IQR method and a given metric which can be Revenue "
                     "or UnitsSold."))

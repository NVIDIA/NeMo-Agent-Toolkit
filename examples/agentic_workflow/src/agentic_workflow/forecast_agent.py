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

import os
import pandas as pd
import json

from pydantic import BaseModel
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

class ForecastAgentConfig(FunctionBaseConfig, name="forecast_agent"):
    """
    Configuration for forecasting time based on product type and quantity.
    """
    forecast_reference_file: str = "/home/varshashiveshwar/projects/AgentIQ/examples/agentic_workflow/src/agentic_workflow/data/forecast_reference.csv"

@register_function(config_type=ForecastAgentConfig)
async def forecast_agent(tool_config: ForecastAgentConfig, builder: Builder):
    """
    Function to forecast number of days based on product type and quantity.
    """

    def load_forecast_data(file_path: str) -> pd.DataFrame:
        """
        Load forecast data from the given CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Forecast file '{file_path}' not found.")
        return pd.read_csv(file_path)

    async def _forecast(input_data: str) -> str:
        """
        Forecast based on the order details (product type and quantity).
        The input data is expected to be a JSON string.
        """
       # try:
       #     data = json.loads(input_data)
       # except json.JSONDecodeError:
       #     raise ValueError("Invalid JSON format in input data.")

       # product_type = data.get("product_type")
       # quantity = data.get("quantity", 1)
        product_type = 'cylinder'
        quantity = 6

        df = load_forecast_data(tool_config.forecast_reference_file)

        product_row = df[df["product_type"] == product_type]
        if product_row.empty:
            raise ValueError(f"Product type '{product_type}' not found in forecast reference.")

        days_per_unit = int(product_row["days_per_unit"].values[0])
        forecast_days = quantity * days_per_unit

        return f"Forecast days: {forecast_days}"

    try:
        yield FunctionInfo.from_fn(
            _forecast,
            description="Forecasts the number of days required based on product type and quantity."
        )
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up forecast_agent workflow.")

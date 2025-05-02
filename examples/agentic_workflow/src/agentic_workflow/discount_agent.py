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

class DiscountResponse(BaseModel):
    total_price: float
    discount_applied: bool

class DiscountAgentConfig(FunctionBaseConfig, name="discount_agent"):
    """
    Configuration for calculating discounts based on product pricing.
    """
    price_reference_file: str = "/home/varshashiveshwar/projects/AgentIQ/examples/agentic_workflow/src/agentic_workflow/data/price_reference.csv"

@register_function(config_type=DiscountAgentConfig)
async def discount_agent(tool_config: DiscountAgentConfig, builder: Builder):
    """
    Function to apply discount based on quantity and product type.
    """

    def load_price_data(file_path: str) -> pd.DataFrame:
        """
        Load price data from the given CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Price file '{file_path}' not found.")
        return pd.read_csv(file_path)

    async def _apply_discount(order_data: str) -> str:
        """
        Apply discount based on the order details (product type and quantity).
        The order data is expected to be a JSON string.
        """

        product_type = 'cylinder'
        quantity = 100

        # Load price data from CSV file
        df = load_price_data(tool_config.price_reference_file)

        # Get the price per unit for the product type
        product_row = df[df["product_type"] == product_type]
        if product_row.empty:
            raise ValueError(f"Product type '{product_type}' not found in price reference.")

        price_per_unit = float(product_row["price_per_unit"].values[0])
        total_price = price_per_unit * quantity

        # Apply a 10% discount if quantity is greater than 10
        discount_applied = quantity > 10
        if discount_applied:
            total_price *= 0.9
        result = f"Total price: {total_price}, Discount applied: {discount_applied}"

        return result

    try:
        yield FunctionInfo.from_fn(
            _apply_discount,
            description="Calculates the total price and applies discount if applicable."
        )
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up discount_agent workflow.")

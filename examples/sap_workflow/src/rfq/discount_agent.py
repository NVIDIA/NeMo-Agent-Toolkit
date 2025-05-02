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

import pandas as pd
import os

from pydantic import BaseModel
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

class DiscountResponse(BaseModel):
    total_price: float
    discount_applied: bool

class DiscountAgentConfig(FunctionBaseConfig, name="discount_agent"):
    price_reference_file: str = "price_reference.csv"

def load_price_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Price file '{file_path}' not found.")
    return pd.read_csv(file_path)

@register_function(config_type=DiscountAgentConfig)
async def discount_agent(tool_config: DiscountAgentConfig, builder: Builder):

    async def _apply_discount(order_data: dict) -> DiscountResponse:
        product_type = order_data.get("product_type")
        quantity = order_data.get("quantity", 1)

        df = load_price_data(tool_config.price_reference_file)

        product_row = df[df["product_type"] == product_type]
        if product_row.empty:
            raise ValueError(f"Product type '{product_type}' not found in price reference.")

        price_per_unit = float(product_row["price_per_unit"].values[0])
        total_price = price_per_unit * quantity

        # Simple discount rule: >10 units = 10% discount
        discount_applied = quantity > 10
        if discount_applied:
            total_price *= 0.9

        return DiscountResponse(total_price=total_price, discount_applied=discount_applied)

    yield FunctionInfo.from_fn(
        _apply_discount,
        description="Calculates the total price and applies discount if applicable."
    )

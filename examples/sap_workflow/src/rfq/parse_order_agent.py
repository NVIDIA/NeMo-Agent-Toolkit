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

import json
import os

from pydantic import BaseModel
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

class OrderDetails(BaseModel):
    customer_name: str
    order_items: list[str]
    special_instructions: str

class ParseOrderResponse(BaseModel):
    order_details: OrderDetails

class ParseOrderAgentConfig(FunctionBaseConfig, name="parse_order_agent"):
    order_file_path: str = "customer_order.json"  # Default to customer_order.json

def load_order_data(file_path: str) -> dict:
    """Load order data from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Order file '{file_path}' not found.")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

@register_function(config_type=ParseOrderAgentConfig)
async def parse_order_agent(tool_config: ParseOrderAgentConfig, builder: Builder):

    async def _parse_order(_: dict) -> ParseOrderResponse:
        # Ignoring the input dict since we are reading from file
        order_data = load_order_data(tool_config.order_file_path)

        customer_name = order_data.get("customer_name", "")
        order_items = order_data.get("order_items", [])  # correct key!
        special_instructions = order_data.get("special_instructions", "")

        order_details = OrderDetails(
            customer_name=customer_name,
            order_items=order_items,
            special_instructions=special_instructions
        )

        return ParseOrderResponse(order_details=order_details)

    yield FunctionInfo.from_fn(
        _parse_order,
        description="Parse the customer order from a JSON file and extract customer name, order items, and special instructions."
    )

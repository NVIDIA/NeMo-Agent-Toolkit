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

class DecisionResponse(BaseModel):
    is_valid_product: bool
    reason: str

class DecisionAgentConfig(FunctionBaseConfig, name="decision_agent"):
    product_reference_file: str = "product_reference.csv"

def load_product_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Product reference file '{file_path}' not found.")
    return pd.read_csv(file_path)

@register_function(config_type=DecisionAgentConfig)
async def decision_agent(tool_config: DecisionAgentConfig, builder: Builder):

    async def _decide(order_data: dict) -> DecisionResponse:
        product_type = order_data.get("product_type")

        df = load_product_data(tool_config.product_reference_file)
        valid_products = df["product_type"].tolist()

        if product_type in valid_products:
            return DecisionResponse(is_valid_product=True, reason="Valid product type.")
        else:
            return DecisionResponse(is_valid_product=False, reason=f"Invalid product type '{product_type}'.")

    yield FunctionInfo.from_fn(
        _decide,
        description="Validates if the product type in the order is acceptable."
    )

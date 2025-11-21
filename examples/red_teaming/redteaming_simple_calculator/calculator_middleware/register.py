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

from nat.cli.register_workflow import register_function_middleware

from .calculator_middleware import CalculatorMiddleware
from .calculator_middleware import CalculatorMiddlewareConfig


@register_function_middleware(config_type=CalculatorMiddlewareConfig)
async def calculator_middleware(config: CalculatorMiddlewareConfig, builder):
    """Build a calculator middleware from configuration.

    Args:
        config: The calculator middleware configuration
        builder: The workflow builder (unused but required by component pattern)

    Yields:
        A configured calculator middleware instance
    """
    yield CalculatorMiddleware(payload=config.payload)

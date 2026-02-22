# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PingConfig(FunctionBaseConfig, name="simple_agentcore_ping"):
    pass


@register_function(config_type=PingConfig)
async def simple_agentcore_ping(_: PingConfig, __: Builder):
    """
    Create a simple health check function for AgentCore compatibility.

    This function provides a ping endpoint that returns a healthy status,
    used by Amazon Bedrock AgentCore for health monitoring.

    Args:
        _: Configuration (unused)
        __: Builder (unused)

    Yields:
        FunctionInfo wrapping a health check function
    """

    async def _ping(unused: str | None) -> dict[str, str]:  # noqa: ARG001
        """Return health status."""
        return {"status": "healthy"}

    yield FunctionInfo.from_fn(_ping, description="Health check")

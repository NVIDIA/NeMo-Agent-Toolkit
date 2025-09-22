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

import requests
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import TelemetryMetricsHostHeartbeatCheckPrompts


class TelemetryMetricsHostHeartbeatCheckToolConfig(FunctionBaseConfig, name="telemetry_metrics_host_heartbeat_check"):
    description: str = Field(default=TelemetryMetricsHostHeartbeatCheckPrompts.TOOL_DESCRIPTION,
                             description="Description of the tool.")
    llm_name: LLMRef
    prompt: str = Field(default=TelemetryMetricsHostHeartbeatCheckPrompts.PROMPT,
                        description="Main prompt for the telemetry metrics host heartbeat check task.")
    offline_mode: bool = Field(default=True, description="Whether to run in offline model")
    metrics_url: str = Field(default="", description="URL of the monitoring system")


@register_function(config_type=TelemetryMetricsHostHeartbeatCheckToolConfig)
async def telemetry_metrics_host_heartbeat_check_tool(config: TelemetryMetricsHostHeartbeatCheckToolConfig,
                                                      builder: Builder):

    async def _arun(host_id: str) -> str:
        utils.log_header("Telemetry Metrics Host Heartbeat Check", dash_length=50)

        try:
            if not config.offline_mode:
                # Example implementation using a monitoring system's API to check host status
                monitoring_url = config.metrics_url

                # Customize query based on your monitoring setup and metrics
                # This example checks if a host's monitoring agent is reporting as up
                query = f'up{{instance=~"{host_id}:9100"}}'  # Adjust port and query pattern for your environment

                url = f"{monitoring_url}/api/query"
                params = {"query": query}

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if data is not None:
                    data = data["data"]
            else:
                # In offline model, load test data from CSV file
                df = utils.get_offline_data()
                data = utils.load_column_or_static(
                    df=df, host_id=host_id, column="telemetry_metrics_host_heartbeat_check_tool:heartbeat_check_output")

            # Additional LLM reasoning layer on playbook output to provide a summary of the results
            utils.log_header("LLM Reasoning", dash_length=30)

            conclusion = await utils.llm_ainvoke(config, builder, user_prompt=config.prompt.format(data=data))

            utils.logger.debug(conclusion)
            utils.log_footer(dash_length=50)

            return conclusion

        except Exception as e:
            utils.logger.error("Error during telemetry metrics host heartbeat check: %s", str(e))
            raise

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )

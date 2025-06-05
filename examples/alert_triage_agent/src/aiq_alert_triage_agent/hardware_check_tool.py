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

import subprocess

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import ToolReasoningLayerPrompts


class HardwareCheckToolConfig(FunctionBaseConfig, name="hardware_check"):
    description: str = Field(
        default=("This tool checks hardware health status using IPMI monitoring to detect power state, "
                 "hardware degradation, and anomalies that could explain alerts. Args: host_id: str"),
        description="Description of the tool for the agent.")
    llm_name: LLMRef
    offline_mode: bool = Field(default=True, description="Whether to run in offline model")


def _get_ipmi_monitor_data(ip_address, username, password):
    """
    Capture IPMI monitoring data using the ipmimonitoring command.

    NOTE: This is just an example implementation of hardware status checking using IPMI.
    Users should implement their own hardware check commands specific to their environment
    and infrastructure setup. The key is to return hardware health/status information in
    a format that can be analyzed.

    Args:
        ip_address (str): The IP address of the device.
        username (str): The user credential for ipmi monitoring.
        password (str): The password credential for ipmi monitoring.

    Returns:
        str: The command's output if successful, otherwise None.
    """
    # Construct the ipmimonitoring command with required parameters
    command = ["ipmimonitoring", "-h", ip_address, "-u", username, "-p", password, "--privilege-level=USER"]

    try:
        # Execute the ipmimonitoring command and capture output
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout

    except subprocess.CalledProcessError as e:
        # Log error and return None if command fails
        utils.logger.error("Error executing IPMI monitoring command. Details: %s", e.stderr)
        return None


@register_function(config_type=HardwareCheckToolConfig)
async def hardware_check_tool(config: HardwareCheckToolConfig, builder: Builder):

    async def _arun(host_id: str) -> str:
        utils.log_header("Hardware Status Checker")

        try:
            if not config.offline_mode:
                ip = "ipmi_ip"  # Replace with your actual IPMI IP address
                user = "ipmi_user"  # Replace with your actual username
                pwd = "ipmi_password"  # Replace with your actual password
                monitoring_data = _get_ipmi_monitor_data(ip, user, pwd)
            else:
                # In offline model, load test data from CSV file
                df = utils.get_offline_data()

                # Get IPMI data from test data, falling back to static data if needed
                monitoring_data = utils.load_column_or_static(
                    df=df,
                    host_id=host_id,
                    column="hardware_check_tool:ipmi_output",
                )

            if monitoring_data is not None:
                # Additional LLM reasoning layer on playbook output to provide a summary of the results
                utils.log_header("LLM Reasoning", dash_length=50)

                prompt = ToolReasoningLayerPrompts.HARDWARE_CHECK.format(input_data=monitoring_data)

                # Get analysis from LLM
                conclusion = await utils.llm_ainvoke(config, builder, prompt)

                # The conclusion from the LLM should not include any sensitive information around the provided
                # credentials. We commented this out to be extra cautious. If you are testing and debugging in a safe
                # environment, uncomment this line to see the conclusion.
                # utils.logger.debug(conclusion)
                utils.log_footer()

                return conclusion

            # Handle case where no IPMI data could be retrieved
            utils.logger.debug("No hardware data available")
            return ("Hardware check failed: Unable to retrieve hardware monitoring data. "
                    "This could indicate connectivity issues with the IPMI interface, "
                    "invalid credentials, or that the IPMI service is not responding.")

        except Exception as e:
            # Log and re-raise any errors that occur
            utils.logger.error("Error during hardware check: %s", str(e))
            raise e

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )

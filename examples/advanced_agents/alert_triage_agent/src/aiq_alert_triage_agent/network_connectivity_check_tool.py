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

import socket
import subprocess

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from . import utils
from .prompts import NetworkConnectivityCheckPrompts


class NetworkConnectivityCheckToolConfig(FunctionBaseConfig, name="network_connectivity_check"):
    description: str = Field(default=NetworkConnectivityCheckPrompts.TOOL_DESCRIPTION,
                             description="Description of the tool.")
    llm_name: LLMRef
    prompt: str = Field(default=NetworkConnectivityCheckPrompts.PROMPT,
                        description="Main prompt for the network connectivity check task.")
    offline_mode: bool = Field(default=True, description="Whether to run in offline model")


def _check_service_banner(host: str, port: int = 80, connect_timeout: float = 10, read_timeout: float = 10) -> str:
    """
    Connects to host:port, reads until the Telnet banner (‘Escape character is '^]'.’) or times out.
    Returns whatever was read (decoded to utf‑8), or an empty string on failure/timeout.
    """
    pattern = b"Escape character is '^]'."
    buffer = b''
    try:
        # 1) Open the TCP connection (replaces telnetlib.Telnet)
        with socket.create_connection((host, port), timeout=connect_timeout) as sock:
            # 2) Set a timeout on subsequent reads
            sock.settimeout(read_timeout)

            # 3) Keep reading until we see the banner or EOF
            while pattern not in buffer:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                buffer += chunk

        # 4) Decode what we got (ignore any non‑UTF8 bytes)
        return buffer.decode('utf-8', errors='ignore')

    except (socket.timeout, ConnectionRefusedError, OSError):
        # timed out or could not connect
        return ''


@register_function(config_type=NetworkConnectivityCheckToolConfig)
async def network_connectivity_check_tool(config: NetworkConnectivityCheckToolConfig, builder: Builder):

    async def _arun(host_id: str) -> str:
        utils.log_header("Network Connectivity Tester")

        try:
            if not config.offline_mode:
                # NOTE: The ping and telnet commands below are example implementations of network connectivity checking.
                # Users should implement their own network connectivity check logic specific to their environment
                # and infrastructure setup.

                # Example ping command to test basic connectivity
                result = subprocess.run(["ping", "-c", "3", host_id], capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    ping_data = result.stdout
                else:
                    ping_data = result.stderr

                # Example telnet command to test service availability
                telnet_port = 80  # example port
                telnet_data = _check_service_banner(host_id, port=telnet_port, connect_timeout=10, read_timeout=10)

            else:
                # Load test data
                df = utils.get_offline_data()

                # Get ping data from test data, falling back to static data if needed
                ping_data = utils.load_column_or_static(df=df,
                                                        host_id=host_id,
                                                        column="network_connectivity_check_tool:ping_output")

                # Get telnet data from test data, falling back to static data if needed
                telnet_data = utils.load_column_or_static(df=df,
                                                          host_id=host_id,
                                                          column="network_connectivity_check_tool:telnet_output")

            # Additional LLM reasoning layer on playbook output to provide a summary of the results
            utils.log_header("LLM Reasoning", dash_length=50)

            prompt = config.prompt.format(ping_data=ping_data, telnet_data=telnet_data)
            conclusion = await utils.llm_ainvoke(config, builder, prompt)

            utils.logger.debug(conclusion)
            utils.log_footer()
            return conclusion
        except Exception as e:
            utils.logger.error("Error during connectivity check: %s", str(e))
            raise e

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )

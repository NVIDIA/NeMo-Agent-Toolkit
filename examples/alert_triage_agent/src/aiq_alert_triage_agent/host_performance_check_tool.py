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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from . import utils
from .playbooks import HOST_PERFORMANCE_CHECK_PLAYBOOK
from .prompts import ToolReasoningLayerPrompts


class HostPerformanceCheckToolConfig(FunctionBaseConfig, name="host_performance_check"):
    description: str = Field(
        default=("This is the Host Performance Check Tool. This tool retrieves CPU usage, memory usage, "
                 "and hardware I/O usage details for a given host. Args: host_id: str"),
        description="Description of the tool for the agent.")
    llm_name: LLMRef
    offline_mode: bool = Field(default=True, description="Whether to run in offline model")


async def _run_ansible_playbook_for_host_performance_check(config: HostPerformanceCheckToolConfig,
                                                           builder: Builder,
                                                           ansible_host: str,
                                                           ansible_user: str,
                                                           ansible_port: int,
                                                           ansible_private_key_path: str) -> list[dict]:
    """
    This function runs a playbook that gathers CPU, memory, and disk I/O metrics and performs
    threshold checks for high resource usage. The playbook executes various system commands
    and collects their output for analysis.

    NOTE: The playbook provided is just an example implementation of host performance monitoring.
    Users should implement their own playbook specific to their environment and monitoring needs.
    The key is to collect performance metrics and threshold data that are relevant for your
    infrastructure and use cases.
    """
    # NOTE: This is an example playbook - users should customize the playbook
    # to collect metrics relevant to their specific monitoring requirements
    playbook = HOST_PERFORMANCE_CHECK_PLAYBOOK

    output = await utils.run_ansible_playbook(playbook=playbook,
                                              ansible_host=ansible_host,
                                              ansible_user=ansible_user,
                                              ansible_port=ansible_port,
                                              ansible_private_key_path=ansible_private_key_path)

    # Extract and structure task results
    extracted_tasks = []
    for task in output.get("task_results", []):
        result = task.get("result", {})
        task_details = {
            "task": task.get("task"),
            "host": task.get("host"),
            "cmd": result.get("cmd"),
            "start": result.get("start"),
            "end": result.get("end"),
            "delta": result.get("delta"),
            "stdout_lines": result.get("stdout_lines"),
            # Run additional LLM reasoning layer on playbook output to break down the task and improve
            # the LLM's understanding of non-natural language system output
            "structured_data": await _parse_stdout_lines(config, builder, result.get("stdout_lines")),
        }
        extracted_tasks.append(task_details)

    return extracted_tasks


async def _parse_stdout_lines(config, builder, stdout_lines):
    """
    Parses the stdout_lines output using nvda_nim to extract structured JSON data.

    Args:
        stdout_lines (list of str): List of output lines from the ansible task.

    Returns:
        str: Structured data parsed from the output in string format.
    """
    response = None
    try:
        # Join the list of lines into a single text block
        input_data = "\n".join(stdout_lines) if stdout_lines else ""

        prompt = ToolReasoningLayerPrompts.HOST_PERFORMANCE_CHECK_PARSING.format(input_data=input_data)

        response = await utils.llm_ainvoke(config=config, builder=builder, user_prompt=prompt)
    except Exception as e:
        response = ('{{"error": "Failed to parse stdout from the playbook run.", '
                    '"exception": "{}", "raw_response": "{}"}}').format(str(e), response)
    return response


@register_function(config_type=HostPerformanceCheckToolConfig)
async def host_performance_check_tool(config: HostPerformanceCheckToolConfig, builder: Builder):

    async def _arun(host_id: str) -> str:
        utils.log_header("Host Performance Analyzer")

        try:
            if not config.offline_mode:
                # In production mode, use actual Ansible connection details
                # Replace placeholder values with connection info from configuration
                ansible_host = "your.host.example.name"  # Input your target host
                ansible_user = "ansible_user"  # Input your SSH user
                ansible_port = 22  # Input your SSH port
                ansible_private_key_path = "/path/to/private/key"  # Input path to your SSH key

                # Run Ansible playbook to collect performance metrics
                output = await _run_ansible_playbook_for_host_performance_check(
                    config=config,
                    builder=builder,
                    ansible_host=ansible_host,
                    ansible_user=ansible_user,
                    ansible_port=ansible_port,
                    ansible_private_key_path=ansible_private_key_path)
            else:
                # In offline model, load performance data from test dataset
                df = utils.get_offline_data()

                # Get CPU metrics from test data, falling back to static data if needed
                data_top_cpu = utils.load_column_or_static(df=df,
                                                           host_id=host_id,
                                                           column="host_performance_check_tool:top_output")
                data_ps_cpu = utils.load_column_or_static(df=df,
                                                          host_id=host_id,
                                                          column="host_performance_check_tool:ps_output")

                output = f"`top` :{data_top_cpu} and `ps` :{data_ps_cpu}"

            # Additional LLM reasoning layer on playbook output to provide a summary of the results
            utils.log_header("LLM Reasoning", dash_length=50)

            prompt_template = ToolReasoningLayerPrompts.HOST_PERFORMANCE_CHECK_ANALYSIS.format(input_data=output)

            conclusion = await utils.llm_ainvoke(config, builder, user_prompt=prompt_template)

            utils.logger.debug(conclusion)
            utils.log_footer()

            return conclusion

        except Exception as e:
            utils.logger.error("Error during host performance check: %s", str(e))
            raise e

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )

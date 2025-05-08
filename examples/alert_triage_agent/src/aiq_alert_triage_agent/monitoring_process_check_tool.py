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
from .playbooks import MONITOR_PROCESS_CHECK_PLAYBOOK
from .prompts import ToolReasoningLayerPrompts


class MonitoringProcessCheckToolConfig(FunctionBaseConfig, name="monitoring_process_check"):
    description: str = Field(default=("This tool checks the status of critical monitoring processes and services "
                                      "on a target host by executing system commands. Args: host_id: str"),
                             description="Description of the tool for the agent.")
    llm_name: LLMRef
    test_mode: bool = Field(default=True, description="Whether to run in test mode")


async def _run_ansible_playbook_for_monitor_process_check(ansible_host: str,
                                                          ansible_user: str,
                                                          ansible_port: int,
                                                          ansible_private_key_path: str) -> list[dict]:
    """
    This function runs a playbook that checks the status of critical monitoring processes
    on the target host. The playbook executes system commands to gather process information
    and service status.

    NOTE: The playbook provided is an example implementation. Users should customize the
    playbook to check processes specific to their monitoring infrastructure.
    """
    output = await utils.run_ansible_playbook(playbook=MONITOR_PROCESS_CHECK_PLAYBOOK,
                                              ansible_host=ansible_host,
                                              ansible_user=ansible_user,
                                              ansible_port=ansible_port,
                                              ansible_private_key_path=ansible_private_key_path)

    extracted_tasks = []
    # Iterate over task_results if available; otherwise use an empty list.
    for task in output.get("task_results", []):
        result = task.get("result", {})
        task_details = {
            "task": task.get("task"),
            "host": task.get("host"),
            "cmd": result.get("cmd"),
            "stdout_lines": result.get("stdout_lines"),
        }
        extracted_tasks.append(task_details)

    return extracted_tasks


@register_function(config_type=MonitoringProcessCheckToolConfig)
async def monitoring_process_check_tool(config: MonitoringProcessCheckToolConfig, builder: Builder):

    async def _arun(host_id: str) -> str:
        try:
            if not config.test_mode:
                # In production mode, use actual Ansible connection details
                # Replace placeholder values with connection info from configuration
                ansible_host = "your.host.example.name"  # Input your target host
                ansible_user = "ansible_user"  # Input your SSH user
                ansible_port = 22  # Input your SSH port
                ansible_private_key_path = "/path/to/private/key"  # Input path to your SSH key

                output = await _run_ansible_playbook_for_monitor_process_check(
                    ansible_host=ansible_host,
                    ansible_user=ansible_user,
                    ansible_port=ansible_port,
                    ansible_private_key_path=ansible_private_key_path)
                output_for_prompt = f"`ps` and `top` result:{output}"
            else:
                # In test mode, load performance data from test dataset
                df = utils.get_test_data()

                # Load process status data from ps command output
                ps_data = utils.load_column_or_static(df=df,
                                                      host_id=host_id,
                                                      column="monitor_process_check_tool:ps_output")
                # Load systemd service status data from systemctl command output
                systemctl_data = utils.load_column_or_static(df=df,
                                                             host_id=host_id,
                                                             column="monitor_process_check_tool:systemctl_output")

                output_for_prompt = f"`ps` result:{ps_data} and `systemctl` result:{systemctl_data}"

            # Additional LLM reasoning layer on playbook output to provide a summary of the results
            utils.log_header("LLM Reasoning", dash_length=50)

            prompt = ToolReasoningLayerPrompts.MONITORING_PROCESS_CHECK.format(input_data=output_for_prompt)

            conclusion = await utils.llm_ainvoke(config, builder, prompt)

            utils.log_header("LLM Reasoning", dash_length=50)
            utils.logger.debug(conclusion)
            utils.log_footer()
            return conclusion

        except Exception as e:
            utils.logger.error("Error during monitoring process check: %s", str(e))
            raise e

    yield FunctionInfo.from_fn(
        _arun,
        description=config.description,
    )

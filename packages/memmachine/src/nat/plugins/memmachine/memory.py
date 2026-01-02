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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


class MemMachineMemoryClientConfig(MemoryBaseConfig, RetryMixin, name="memmachine_memory"):
    """
    Configuration for MemMachine memory client.
    
    Based on the MemMachine Python SDK as documented at:
    https://github.com/MemMachine/MemMachine/blob/main/docs/examples/python.mdx
    
    Note: This integration is for local/self-hosted MemMachine instances.
    LLM API keys (e.g., OpenAI) are configured in the MemMachine cfg.yml file,
    not in this client configuration.
    """
    base_url: str  # Base URL of the MemMachine server (e.g., "http://localhost:8080")
    org_id: str | None = None  # Optional default organization ID
    project_id: str | None = None  # Optional default project ID
    timeout: int = 30  # Request timeout in seconds
    max_retries: int = 3  # Maximum number of retries for failed requests


@register_memory(config_type=MemMachineMemoryClientConfig)
async def memmachine_memory_client(config: MemMachineMemoryClientConfig, builder: Builder):
    from .memmachine_editor import MemMachineEditor
    # Import and initialize the MemMachine Python SDK
    try:
        from memmachine import MemMachineClient
    except ImportError as e:
        raise ImportError(
            f"Could not import MemMachineClient from memmachine package. "
            f"Error: {e}. "
            "Please ensure memmachine package is installed: pip install memmachine. "
            "See https://github.com/MemMachine/MemMachine/blob/main/docs/examples/python.mdx "
            "for installation instructions."
        ) from e

    # Initialize MemMachineClient with base_url
    # This follows the documented SDK pattern for local instances:
    # client = MemMachineClient(base_url="http://localhost:8080")
    # Note: api_key is not needed for local/self-hosted MemMachine instances
    try:
        client = MemMachineClient(
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize MemMachineClient with base_url '{config.base_url}'. "
            f"Error: {e}. "
            "Please ensure the MemMachine server is running and the base_url is correct."
        ) from e

    # If default org_id and project_id are provided, create/get the project
    # Otherwise, the editor will create projects as needed
    memmachine_instance = client
    if config.org_id and config.project_id:
        try:
            project = client.create_project(
                org_id=config.org_id,
                project_id=config.project_id,
                description=f"NeMo Agent Toolkit project: {config.project_id}"
            )
            memmachine_instance = project
        except Exception as e:
            # If project creation fails, fall back to using the client directly
            # The editor will handle project creation on-demand
            pass

    memory_editor = MemMachineEditor(memmachine_instance=memmachine_instance)

    if isinstance(config, RetryMixin):
        memory_editor = patch_with_retry(
            memory_editor,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors
        )

    yield memory_editor
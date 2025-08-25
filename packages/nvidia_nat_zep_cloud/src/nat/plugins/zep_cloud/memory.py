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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_memory
from nat.data_models.memory import MemoryBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


class ZepMemoryClientConfig(MemoryBaseConfig, RetryMixin, name="zep_memory"):
    base_url: str | None = None
    timeout: float | None = None
    follow_redirects: bool | None = None


@register_memory(config_type=ZepMemoryClientConfig)
async def zep_memory_client(config: ZepMemoryClientConfig, builder: Builder):
    import os

    from zep_cloud.client import AsyncZep

    from nat.plugins.zep_cloud.zep_editor import ZepEditor

    zep_api_key = os.environ.get("ZEP_API_KEY")

    if zep_api_key is None:
        raise RuntimeError("Zep API key is not set. Please specify it in the environment variable 'ZEP_API_KEY'.")

    zep_client = AsyncZep(api_key=zep_api_key,
                          base_url=config.base_url,
                          timeout=config.timeout,
                          follow_redirects=config.follow_redirects)
    memory_editor = ZepEditor(zep_client)

    if isinstance(config, RetryMixin):
        memory_editor = patch_with_retry(memory_editor,
                                         retries=config.num_retries,
                                         retry_codes=config.retry_on_status_codes,
                                         retry_on_messages=config.retry_on_errors)

    yield memory_editor

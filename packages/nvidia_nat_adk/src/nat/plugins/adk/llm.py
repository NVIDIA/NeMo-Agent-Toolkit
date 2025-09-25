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
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.llm.litellm_llm import LiteLlmModelConfig


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def litellm_adk(config: LiteLlmModelConfig, _builder: Builder):
    """Create and yield a Google ADK `LiteLlm` client from a NAT `LiteLlmModelConfig`.

    Args:
        config (LiteLlmModelConfig): The configuration for the LiteLlm model.
        _builder (Builder): The NAT builder instance.
    """
    from google.adk.models.lite_llm import LiteLlm

    yield LiteLlm(**config.model_dump(
        exclude={"type", "max_retries", "thinking"},
        by_alias=True,
        exclude_none=True,
    ))

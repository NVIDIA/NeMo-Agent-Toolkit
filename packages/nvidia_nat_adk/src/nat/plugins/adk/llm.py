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

from collections.abc import AsyncIterator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.llm.litellm_llm import LiteLlmModelConfig


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.ADK)
async def litellm_adk(
    litellm_config: LiteLlmModelConfig,
    _builder: Builder,  # pylint: disable=W0613 (_builder not used)
) -> AsyncIterator["LiteLlm"]:  # type: ignore # noqa: F821 (forward reference of LiteLlm)
    """Create and yield a Google ADK `LiteLlm` client from a NAT `LiteLlmModelConfig`.

    Args:
        litellm_config (LiteLlmModelConfig): The configuration for the LiteLlm model.
        _builder (Builder): The NAT builder instance.

    Yields:
        AsyncIterator[LiteLlm]: An async iterator that yields a LiteLlm client.
    """

    try:
        from google.adk.models.lite_llm import LiteLlm
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Google ADK not installed; pip install google-adk") from e

    llm = LiteLlm(**litellm_config.model_dump(
        exclude={"type", "max_retries"},
        by_alias=True,
        exclude_none=True,
    ))

    yield llm

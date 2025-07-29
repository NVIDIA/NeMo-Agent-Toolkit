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

import os

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client
from aiq.data_models.retry_mixin import RetryMixin
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig
from aiq.utils.exception_handlers.automatic_retries import patch_with_retry


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def nim_crewai(llm_config: NIMModelConfig, builder: Builder):

    from crewai import LLM

    config_obj = {
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
        "model": f"nvidia_nim/{llm_config.model_name}",
    }

    # Because CrewAI uses a different environment variable for the API key, we need to set it here manually
    if ("api_key" not in config_obj or config_obj["api_key"] is None):

        if ("NVIDIA_NIM_API_KEY" in os.environ):
            # Dont need to do anything. User has already set the correct key
            pass
        else:
            nvidai_api_key = os.getenv("NVIDIA_API_KEY")

            if (nvidai_api_key is not None):
                # Transfer the key to the correct environment variable for LiteLLM
                os.environ["NVIDIA_NIM_API_KEY"] = nvidai_api_key

    client = LLM(**config_obj)

    if isinstance(llm_config, RetryMixin):

        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    yield client


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.CREWAI)
async def openai_crewai(llm_config: OpenAIModelConfig, builder: Builder):

    from crewai import LLM

    config_obj = {
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
    }

    client = LLM(**config_obj)

    if isinstance(llm_config, RetryMixin):

        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    yield client

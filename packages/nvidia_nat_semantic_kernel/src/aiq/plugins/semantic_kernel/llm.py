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
from nat.data_models.retry_mixin import RetryMixin
from nat.llm.openai_llm import OpenAIModelConfig
from nat.utils.exception_handlers.automatic_retries import patch_with_retry


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.SEMANTIC_KERNEL)
async def openai_semantic_kernel(llm_config: OpenAIModelConfig, builder: Builder):

    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    config_obj = {
        **llm_config.model_dump(exclude={"type"}, by_alias=True),
    }

    llm = OpenAIChatCompletion(ai_model_id=config_obj.get("model"))

    if isinstance(llm_config, RetryMixin):
        llm = patch_with_retry(llm,
                               retries=llm_config.num_retries,
                               retry_codes=llm_config.retry_on_status_codes,
                               retry_on_messages=llm_config.retry_on_errors)

    yield llm

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
# pylint: disable=raising-format-tuple

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.llm import APITypeEnum


def validate_no_responses_api(llm_config, framework: LLMFrameworkEnum):
    """Validate that the LLM config does not use the Responses API."""

    if llm_config.api_type == APITypeEnum.RESPONSES:
        raise ValueError(f"Responses API is not supported for config {str(type(llm_config))} in framework {framework}. "
                         f"Please use a different API type.")

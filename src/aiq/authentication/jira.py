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

from aiq.builder.builder import Builder
from aiq.data_models.authentication import AuthenticationBaseConfig


class JiraConfig(LLMBaseConfig, name="jira_api"):
    pass


@register_api_provider(config_type=JiraConfig)
async def jira_provider(api_config: JiraConfig, builder: Builder):
    pass


@register_api_client(config_type=JiraConfig, wrapper_type=None)
async def jira_client(api_config: JiraConfig, builder: Builder):
    pass

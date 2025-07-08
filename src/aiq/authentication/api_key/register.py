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

from aiq.authentication.api_key.api_key_client import APIKeyClient
from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_authentication_client


@register_authentication_client(config_type=APIKeyConfig)
async def api_key_client(authentication_provider: APIKeyConfig, builder: Builder):

    yield APIKeyClient(config=authentication_provider)

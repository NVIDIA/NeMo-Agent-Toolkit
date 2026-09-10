# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.authentication.jwt.jwt_auth_provider_config import JwtAuthProviderConfig
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider


@register_auth_provider(config_type=JwtAuthProviderConfig)
async def jwt_auth_provider(config: JwtAuthProviderConfig, builder: Builder):
    from nat.authentication.jwt.jwt_auth_provider import JwtAuthProvider

    yield JwtAuthProvider(config)

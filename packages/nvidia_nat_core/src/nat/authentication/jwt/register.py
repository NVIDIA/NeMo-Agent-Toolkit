# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.authentication.jwt.jwt_auth_provider_config import JwtAuthProviderConfig
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_auth_provider


@register_auth_provider(config_type=JwtAuthProviderConfig)
async def jwt_auth_provider(config: JwtAuthProviderConfig, builder: Builder):
    from nat.authentication.jwt.jwt_auth_provider import JwtAuthProvider

    yield JwtAuthProvider(config)

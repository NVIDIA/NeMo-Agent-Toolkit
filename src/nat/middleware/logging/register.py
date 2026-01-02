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
from nat.cli.register_workflow import register_middleware
from nat.middleware.logging.logging_middleware import LoggingMiddleware
from nat.middleware.logging.logging_middleware_config import LoggingMiddlewareConfig


@register_middleware(config_type=LoggingMiddlewareConfig)
async def logging_middleware(config: LoggingMiddlewareConfig, builder: Builder):
    """Build a logging middleware from configuration.

    Args:
        config: The logging middleware configuration
        builder: The workflow builder

    Yields:
        A configured logging middleware instance
    """
    yield LoggingMiddleware(config=config, builder=builder)

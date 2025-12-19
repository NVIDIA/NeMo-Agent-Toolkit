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

import logging

from nat.builder.builder import Builder
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.logging.logging_middleware_config import LoggingMiddlewareConfig
from nat.middleware.middleware import InvocationContext

logger = logging.getLogger(__name__)


class LoggingMiddleware(DynamicFunctionMiddleware):
    """Middleware that logs intercepted function inputs/outputs.

    This middleware extends DynamicFunctionMiddleware to get automatic chain
    orchestration and dynamic discovery features. Custom logic is implemented
    through the pre_invoke and post_invoke hooks.
    """

    def __init__(self, config: LoggingMiddlewareConfig, builder: Builder):
        """Initialize logging middleware.

        Args:
            config: Logging middleware configuration
            builder: Workflow builder
        """
        super().__init__(config=config, builder=builder)
        self._config: LoggingMiddlewareConfig = config

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Log inputs before function execution.

        Returns:
            None to pass through unchanged (logging only, no modification)
        """
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        logger.log(log_level, f"Calling {context.function_context.name} with args: {context.modified_args}")

        # Log if args were modified by prior middleware
        if context.modified_args != context.original_args:
            logger.log(log_level, f"  (original args were: {context.original_args})")

        return None  # Pass through unchanged

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Log outputs after function execution.

        Returns:
            None to pass through unchanged (logging only, no modification)
        """
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        logger.log(log_level, f"Function {context.function_context.name} returned: {context.output}")
        return None  # Pass through unchanged

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
from collections.abc import Callable
from typing import Any

from starlette.datastructures import Headers

from nat.builder.context import Context
from nat.observability.processor.redaction.contextual_span_redaction_processor import ContextualSpanRedactionProcessor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class SpanHeaderRedactionProcessor(ContextualSpanRedactionProcessor[dict[str, Any]]):
    """Processor that redacts the Span based on multiple headers and callback logic.

    Uses context-scoped atomic updates to avoid redundant callback executions within a single context.
    Since headers are static per request, the callback result is cached for the entire context using
    an asynccontextmanager to ensure atomic operations.

    Args:
        headers: List of header keys to extract and pass to the callback
        attributes: List of Span attribute keys to redact
        callback: Callable that determines if redaction should occur
        enabled: Whether the processor is enabled (default: True)
        force_redact: If True, always redact regardless of header checks (default: False)
        redaction_value: The value to replace redacted attributes with (default: "[REDACTED]")
    """

    def __init__(self,
                 headers: list[str],
                 attributes: list[str],
                 callback: Callable[..., Any],
                 enabled: bool = True,
                 force_redact: bool = False,
                 redaction_value: str = "[REDACTED]",
                 redaction_tag: str | None = None):
        # Initialize the base class with common parameters
        super().__init__(attributes=attributes,
                         callback=callback,
                         enabled=enabled,
                         force_redact=force_redact,
                         redaction_value=redaction_value,
                         redaction_tag=redaction_tag)
        # Store header-specific configuration
        self.headers = headers

    @override
    def extract_data_from_context(self) -> dict[str, Any] | None:
        """Extract header data from the context.

        Returns:
            dict[str, Any] | None: Dictionary of header names to values, or None if no headers.
        """

        context = Context.get()
        headers: Headers | None = context.metadata.headers

        if headers is None or not self.headers:
            return None

        header_map: dict[str, Any] = {header: headers.get(header, None) for header in self.headers}

        return header_map

    @override
    def validate_data(self, data: dict[str, Any]) -> bool:
        """Validate that the extracted headers are suitable for callback execution.

        Args:
            data (dict[str, Any]): The extracted header dictionary.

        Returns:
            bool: True if headers exist and are not all None, False otherwise.
        """
        # Skip callback if no headers were found (all None values)
        return bool(data) and not all(value is None for value in data.values())

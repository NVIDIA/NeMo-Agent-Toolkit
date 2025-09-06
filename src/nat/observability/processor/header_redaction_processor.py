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
from functools import lru_cache
from typing import Any

from starlette.datastructures import Headers

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.observability.processor.redaction_processor import SpanRedactionProcessor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


def default_callback(_header_map: dict[str, Any]) -> bool:
    """Default callback that always returns False, indicating no redaction should occur.

    Args:
        _header_map: Dictionary of header names to values (unused).

    Returns:
        bool: Always False, indicating the span should not be redacted.
    """
    return False


class HeaderRedactionProcessor(SpanRedactionProcessor):
    """Processor that redacts the span based on multiple headers and callback logic.

    Uses an LRU cache to avoid redundant callback executions for the same header combinations,
    providing bounded memory usage and automatic eviction of least recently used entries.

    Args:
        attributes: List of span attribute keys to redact.
        headers: List of header keys to extract and pass to the callback function.
        callback: Function that receives a dict of headers and determines if redaction should occur.
                 The callback receives headers in the order specified in the headers list.
        enabled: Whether the processor is enabled (default: True).
        force_redact: If True, always redact regardless of header checks (default: False).
        redaction_value: The value to replace redacted attributes with (default: "[REDACTED]").
    """

    def __init__(self,
                 attributes: list[str] | None = None,
                 headers: list[str] | None = None,
                 callback: Callable[[dict[str, Any]], bool] | None = None,
                 enabled: bool = True,
                 force_redact: bool = False,
                 redaction_value: str = "[REDACTED]"):
        self.attributes = attributes or []
        self.headers = headers or []
        self.callback = callback or default_callback
        self.enabled = enabled
        self.force_redact = force_redact
        self.redaction_value = redaction_value

    @override
    def should_redact(self, item: Span, context: Context) -> bool:
        """Determine if this span should be redacted based on header values.

        Extracts the specified headers from the context and passes them to the
        callback function to determine if redaction should occur.

        Args:
            item (Span): The span to check.
            context (Context): The current context containing headers.

        Returns:
            bool: True if the span should be redacted, False otherwise.
        """
        # If force_redact is enabled, always redact regardless of other conditions
        if self.force_redact:
            return True

        if not self.enabled:
            return False

        headers: Headers | None = context.metadata.headers

        if headers is None or not self.headers:
            return False

        header_map: dict[str, Any] = {header: headers.get(header, None) for header in self.headers}

        # Skip callback if no headers were found (all None values)
        if not header_map or all(value is None for value in header_map.values()):
            return False

        # Use LRU cached method to determine if redaction is needed
        header_tuple = tuple((header, header_map.get(header)) for header in self.headers)
        return self._should_redact_cached(self.callback, header_tuple)

    @staticmethod
    @lru_cache(maxsize=128)
    def _should_redact_cached(callback: Callable[[dict[str, Any]], bool], header_tuple: tuple) -> bool:
        """Static cached method for checking if redaction should occur.

        This method uses lru_cache to avoid redundant callback executions.
        By being static, it avoids the 'self' hashing issue.

        Args:
            callback: The callback function to execute.
            header_tuple: Tuple of (key, value) pairs from headers in self.headers order.

        Returns:
            bool: True if the span should be redacted, False otherwise.
        """
        # Convert tuple back to dict and execute callback
        header_dict = dict(header_tuple)
        return callback(header_dict)

    @override
    def redact_item(self, item: Span) -> Span:
        """Redact the span.

        Args:
            item (Span): The span to redact.

        Returns:
            Span: The redacted span.
        """
        for key in self.attributes:
            if key in item.attributes:
                item.attributes[key] = self.redaction_value

        return item

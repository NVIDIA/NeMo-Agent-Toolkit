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

from starlette.datastructures import Headers

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.observability.processor.redaction_processor import SpanRedactionProcessor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


def default_callback(_auth_key: str) -> bool:
    """Default callback that always returns False."""
    return False


class HeaderRedactionProcessor(SpanRedactionProcessor):
    """Processor that redacts the span based on auth key, span attributes, and callback.

    Uses an LRU cache to avoid redundant callback executions for the same auth keys,
    providing bounded memory usage and automatic eviction of least recently used entries.

    Args:
        attributes: List of span attribute keys to redact.
        header: The header key to check for authentication.
        callback: Function to determine if the auth key should trigger redaction.
        enabled: Whether the processor is enabled (default: True).
        force_redact: If True, always redact regardless of header checks (default: False).
        redaction_value: The value to replace redacted attributes with (default: "[REDACTED]").
    """

    def __init__(self,
                 attributes: list[str] | None = None,
                 header: str | None = None,
                 callback: Callable[[str], bool] | None = None,
                 enabled: bool = True,
                 force_redact: bool = False,
                 redaction_value: str = "[REDACTED]"):
        self.attributes = attributes or []
        self.header = header
        self.callback = callback or default_callback
        self.enabled = enabled
        self.force_redact = force_redact
        self.redaction_value = redaction_value

    @override
    def should_redact(self, item: Span, context: Context) -> bool:
        """Determine if this span should be redacted based on header auth.

        Args:
            item (Span): The span to check.
            context (Context): The current context.

        Returns:
            bool: True if the span should be redacted, False otherwise.
        """
        # If force_redact is enabled, always redact regardless of other conditions
        if self.force_redact:
            return True

        if not self.enabled:
            return False

        headers: Headers | None = context.metadata.headers

        if headers is None or self.header is None:
            return False

        auth_key = headers.get(self.header, None)

        if not auth_key:
            return False

        # Use LRU cached method to determine if redaction is needed
        return self._should_redact_impl(auth_key)

    @lru_cache(maxsize=128)
    def _should_redact_impl(self, auth_key: str) -> bool:
        """Implementation method for checking if redaction should occur.

        This method uses lru_cache to avoid redundant callback executions.

        Args:
            auth_key (str): The authentication key to check.

        Returns:
            bool: True if the span should be redacted, False otherwise.
        """
        return self.callback(auth_key)

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

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
from abc import abstractmethod
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from nat.observability.processor.redaction.redaction_processor import RedactionContext
from nat.observability.processor.redaction.redaction_processor import RedactionContextState
from nat.observability.processor.redaction.redaction_processor import RedactionInputT
from nat.observability.processor.redaction.redaction_processor import RedactionProcessor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

# Type variable for the data type extracted from context
RedactionDataT = TypeVar('RedactionDataT')


class ContextualRedactionProcessor(RedactionProcessor[RedactionInputT, RedactionDataT]):
    """Generic processor with context-aware caching for any data type.

    Provides a framework for redaction processors that need to:
    - Extract data from the request context (headers, cookies, query params, etc.)
    - Execute callbacks to determine redaction decisions
    - Cache results within the request context to avoid redundant callback executions
    - Handle race conditions with atomic operations

    This class handles all the generic caching, context management, and callback
    execution logic. Subclasses only need to implement data extraction and validation.

    Args:
        callback: Callable that determines if redaction should occur based on extracted data
        enabled: Whether the processor is enabled
        force_redact: If True, always redact regardless of data checks
        redaction_value: The value to replace redacted attributes with
    """

    def __init__(
        self,
        callback: Callable[..., Any],
        enabled: bool,
        force_redact: bool,
        redaction_value: str,
    ):
        self.callback = callback
        self.enabled = enabled
        self.force_redact = force_redact
        self.redaction_value = redaction_value
        self._redaction_context = RedactionContext(RedactionContextState())

    @abstractmethod
    def extract_data_from_context(self) -> RedactionDataT | None:
        """Extract the relevant data from the context for redaction decision.

        This method must be implemented by subclasses to extract their specific
        data type (headers, cookies, query params, etc.) from the request context

        Returns:
            RedactionDataT | None: The extracted data, or None if no relevant data found
        """
        pass

    @abstractmethod
    def validate_data(self, data: RedactionDataT) -> bool:
        """Validate that the extracted data is suitable for callback execution.

        This method allows subclasses to implement their own validation logic
        (e.g., checking if headers exist, if cookies are not empty, etc.).

        Args:
            data (RedactionDataT): The extracted data to validate

        Returns:
            bool: True if the data is valid for callback execution, False otherwise
        """
        pass

    @override
    async def should_redact(self, item: RedactionInputT) -> bool:
        """Determine if this span should be redacted based on extracted data.

        Extracts the relevant data from the context, validates it, and passes it to the
        callback function to determine if redaction should occur. Results are cached
        within the request context to avoid redundant callback executions.

        Args:
            item (RedactionInputT): The item to check

        Returns:
            bool: True if the span should be redacted, False otherwise
        """
        # If force_redact is enabled, always redact regardless of other conditions
        if self.force_redact:
            return True

        if not self.enabled:
            return False

        # Extract data using subclass implementation
        data = self.extract_data_from_context()
        if data is None:
            return False

        # Validate data using subclass implementation
        if not self.validate_data(data):
            return False

        # Use the generic caching framework for callback execution
        async with self._redaction_context.redaction_manager() as manager:
            return await manager.redaction_check(self.callback, data)

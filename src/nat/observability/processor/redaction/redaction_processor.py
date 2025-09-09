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
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Generic
from typing import TypeVar

from nat.observability.processor.processor import Processor
from nat.utils.callable_utils import ainvoke_any
from nat.utils.type_utils import override

RedactionInputT = TypeVar('RedactionInputT')
RedactionDataT = TypeVar('RedactionDataT')

logger = logging.getLogger(__name__)


class RedactionProcessor(Processor[RedactionInputT, RedactionInputT], Generic[RedactionInputT, RedactionDataT]):
    """Abstract base class for redaction processors."""

    @abstractmethod
    async def should_redact(self, item: RedactionInputT) -> bool:
        """Determine if this item should be redacted.

        Args:
            item (RedactionInputT): The item to check.

        Returns:
            bool: True if the item should be redacted, False otherwise.
        """
        pass

    @abstractmethod
    async def redact_item(self, item: RedactionInputT) -> RedactionInputT:
        """Redact the item.

        Args:
            item (RedactionInputT): The item to redact.

        Returns:
            RedactionInputT: The redacted item.
        """
        pass

    @override
    async def process(self, item: RedactionInputT) -> RedactionInputT:
        """Perform redaction on the item if it should be redacted.

        Args:
            item (RedactionInputT): The item to process.

        Returns:
            RedactionInputT: The processed item.
        """
        if await self.should_redact(item):
            return await self.redact_item(item)
        return item


@dataclass
class RedactionContextState:
    """Generic context state for redaction results.

    Stores the redaction result in a context variable to avoid redundant
    callback executions within the same request context.
    """

    redaction_result: ContextVar[bool
                                 | None] = field(default_factory=lambda: ContextVar("redaction_result", default=None))


class RedactionManager(Generic[RedactionDataT]):
    """Generic manager for atomic redaction operations.

    Handles state mutations and ensures atomic callback execution
    with proper result caching within a request context.

    Args:
        RedactionDataT: The type of data being processed for redaction decisions.
    """

    def __init__(self, context_state: RedactionContextState):
        self._context_state = context_state

    def set_redaction_result(self, result: bool) -> None:
        """Set the redaction result in the context.

        Args:
            result (bool): The redaction result to cache.
        """
        self._context_state.redaction_result.set(result)

    def clear_redaction_result(self) -> None:
        """Clear the cached redaction result from the context."""
        self._context_state.redaction_result.set(None)

    async def redaction_check(self, callback: Callable[..., Any], data: RedactionDataT) -> bool:
        """Execute redaction callback with atomic result caching.

        Checks for existing cached results first, then executes the callback
        and caches the result atomically. Since data is static per request,
        subsequent calls within the same context return the cached result.

        Supports sync/async functions, generators, and async generators.

        Args:
            callback (Callable[..., Any]): The callback to execute (sync/async function, generator, etc.).
            data (RedactionDataT): The data to pass to the callback for redaction decision.

        Returns:
            bool: True if the item should be redacted, False otherwise.
        """
        # Check if we already have a result for this context
        existing_result = self._context_state.redaction_result.get()
        if existing_result is not None:
            return existing_result

        # Execute callback and cache result
        result_value = await ainvoke_any(callback, data)
        result = bool(result_value)
        self.set_redaction_result(result)
        return result


class RedactionContext(Generic[RedactionDataT]):
    """Generic context provider for redaction operations.

    Provides read-only access to redaction state and manages the
    RedactionManager lifecycle through async context managers.

    Args:
        RedactionDataT: The type of data being processed for redaction decisions.
    """

    def __init__(self, context: RedactionContextState):
        self._context_state: RedactionContextState = context

    @property
    def redaction_result(self) -> bool | None:
        """Get the current redaction result from context.

        Returns:
            bool | None: The cached redaction result, or None if not set.
        """
        return self._context_state.redaction_result.get()

    @asynccontextmanager
    async def redaction_manager(self) -> AsyncGenerator[RedactionManager[RedactionDataT], None]:
        """Provide a redaction manager within an async context.

        Creates and yields a RedactionManager instance for atomic
        redaction operations within the current context.

        Yields:
            RedactionManager[RedactionDataT]: Manager instance for redaction operations.
        """
        yield RedactionManager(self._context_state)

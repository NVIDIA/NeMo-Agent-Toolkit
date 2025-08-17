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

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class EvalTraceContext:
    """
    Evaluation trace context manager for coordinating traces.

    This class provides a framework-agnostic way to:
    1. Track evaluation calls/contexts
    2. Ensure proper parent-child relationships in traces
    3. Synchronize evaluation completion with trace exports
    """

    def __init__(self):
        self.eval_call = None  # Store the evaluation call/context for propagation
        self.pending_exports = 0  # Track pending trace exports across all exporters
        self.export_complete_event = asyncio.Event()
        self.export_complete_event.set()  # Initially set since no operations pending

    def set_eval_call(self, eval_call: Any) -> None:
        """Set the evaluation call/context for propagation to traces."""
        self.eval_call = eval_call
        if eval_call:
            logger.debug("Set evaluation call context: %s", getattr(eval_call, 'id', str(eval_call)))

    def get_eval_call(self) -> Any:
        """Get the current evaluation call/context."""
        return self.eval_call

    def track_export_start(self) -> None:
        """Called when any trace export operation begins."""
        self.pending_exports += 1
        if self.export_complete_event.is_set():
            self.export_complete_event.clear()
        logger.debug("Export operation started. Pending exports: %d", self.pending_exports)

    def track_export_complete(self) -> None:
        """Called when any trace export operation completes."""
        self.pending_exports = max(0, self.pending_exports - 1)
        if self.pending_exports == 0:
            self.export_complete_event.set()
        logger.debug("Export operation completed. Pending exports: %d", self.pending_exports)

    async def wait_for_exports(self) -> None:
        """Wait for all trace exports to complete."""
        if self.pending_exports > 0:
            logger.debug("Waiting for %d pending exports to complete", self.pending_exports)
            await self.export_complete_event.wait()
            logger.debug("All exports completed")

    @contextmanager
    def evaluation_context(self):
        """
        Context manager that can be overridden by framework-specific implementations.
        Default implementation is a no-op.
        """
        yield


class WeaveEvalTraceContext(EvalTraceContext):
    """
    Weave-specific implementation of evaluation trace context.
    """

    def __init__(self):
        super().__init__()
        self.available = False
        self.set_call_stack: Callable[[Any], Any] | None = None

        try:
            from weave.trace.context.call_context import set_call_stack
            self.set_call_stack = set_call_stack
            self.available = True
        except ImportError:
            self.available = False
            logger.debug("Weave not available for trace context")

    @contextmanager
    def evaluation_context(self):
        """Set the evaluation call as active context for Weave traces."""
        if self.available and self.eval_call and self.set_call_stack:
            try:
                with self.set_call_stack([self.eval_call]):
                    logger.debug("Set Weave evaluation call context: %s",
                                 getattr(self.eval_call, 'id', str(self.eval_call)))
                    yield
            except Exception as e:
                logger.warning("Failed to set Weave evaluation call context: %s", e)
                yield
        else:
            yield


class EvalExporterMixin(ABC):
    """
    Mixin to add evaluation context integration to any exporter.

    This mixin provides functionality to coordinate with evaluation runs,
    enabling proper trace hierarchy and export coordination.
    """

    def __init__(self, *args, eval_context: 'EvalTraceContext | None' = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_context = eval_context

    def set_eval_context(self, eval_context: 'EvalTraceContext') -> None:
        """Set the evaluation trace context."""
        self.eval_context = eval_context

    @abstractmethod
    def get_current_call_context(self) -> Any:
        """Get the current call context from the tracing framework.

        Returns:
            The current call/span/trace context from your specific tracing framework,
            or None if no current context exists or your framework doesn't support it.
        """
        pass

    def get_evaluation_parent_call(self) -> Any:
        """
        Get the parent call for traces, prioritizing evaluation context.

        Returns:
            The parent call to use, following this priority:
            1. Evaluation call from context (highest priority)
            2. Current call from tracing framework
            3. None (no parent)
        """
        # Priority 1: Evaluation call from context (if available)
        eval_call = self.eval_context.get_eval_call() if self.eval_context else None
        current_call = self.get_current_call_context()

        if eval_call and current_call and getattr(current_call, 'id', None) == getattr(eval_call, 'id', None):
            logger.debug("Using evaluation call as parent: %s", getattr(eval_call, 'id', str(eval_call)))
            return eval_call

        # Priority 2: Current call from framework
        if current_call:
            logger.debug("Using current call as parent: %s", getattr(current_call, 'id', str(current_call)))
            return current_call

        # Priority 3: Fallback to evaluation call if available
        if eval_call:
            logger.debug("Using evaluation call as fallback parent: %s", getattr(eval_call, 'id', str(eval_call)))
            return eval_call

        return None

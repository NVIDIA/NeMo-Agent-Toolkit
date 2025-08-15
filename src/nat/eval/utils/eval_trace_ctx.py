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
from contextlib import contextmanager
from typing import Any
from typing import Protocol

logger = logging.getLogger(__name__)


class TraceExporter(Protocol):
    """Protocol for trace exporters that can coordinate with evaluation context."""

    def track_operation_start(self) -> None:
        """Called when a trace export operation begins."""
        ...

    def track_operation_complete(self) -> None:
        """Called when a trace export operation completes."""
        ...


class EvalTraceContext:
    """
    Generic evaluation trace context manager for coordinating traces across different exporters.

    This class provides a framework-agnostic way to:
    1. Track evaluation calls/contexts
    2. Coordinate with multiple trace exporters
    3. Ensure proper parent-child relationships in traces
    4. Synchronize evaluation completion with trace exports
    """

    def __init__(self):
        self.eval_call = None  # Store the evaluation call/context for propagation
        self.pending_exports = 0  # Track pending trace exports across all exporters
        self.export_complete_event = asyncio.Event()
        self.export_complete_event.set()  # Initially set since no operations pending
        self.registered_exporters: list[TraceExporter] = []

    def register_exporter(self, exporter: TraceExporter) -> None:
        """Register a trace exporter for coordination."""
        if exporter not in self.registered_exporters:
            self.registered_exporters.append(exporter)
            logger.debug("Registered trace exporter: %s", type(exporter).__name__)

    def unregister_exporter(self, exporter: TraceExporter) -> None:
        """Unregister a trace exporter."""
        if exporter in self.registered_exporters:
            self.registered_exporters.remove(exporter)
            logger.debug("Unregistered trace exporter: %s", type(exporter).__name__)

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

        # Notify all registered exporters
        for exporter in self.registered_exporters:
            try:
                exporter.track_operation_start()
            except Exception as e:
                logger.warning("Error notifying exporter %s of operation start: %s", type(exporter).__name__, e)

    def track_export_complete(self) -> None:
        """Called when any trace export operation completes."""
        self.pending_exports = max(0, self.pending_exports - 1)
        if self.pending_exports == 0:
            self.export_complete_event.set()
        logger.debug("Export operation completed. Pending exports: %d", self.pending_exports)

        # Notify all registered exporters
        for exporter in self.registered_exporters:
            try:
                exporter.track_operation_complete()
            except Exception as e:
                logger.warning("Error notifying exporter %s of operation complete: %s", type(exporter).__name__, e)

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

    def reset(self) -> None:
        """Reset the context state."""
        self.eval_call = None
        self.pending_exports = 0
        self.export_complete_event.set()
        logger.debug("Reset evaluation trace context")


class WeaveEvalTraceContext(EvalTraceContext):
    """
    Weave-specific implementation of evaluation trace context.
    """

    def __init__(self):
        super().__init__()
        self.available = False
        self._weave_imports = {}

        try:
            from weave.trace.context.call_context import get_current_call
            from weave.trace.context.call_context import set_call_stack
            self._weave_imports['get_current_call'] = get_current_call
            self._weave_imports['set_call_stack'] = set_call_stack
            self.available = True
        except ImportError:
            self.available = False
            logger.debug("Weave not available for trace context")

    def capture_current_call(self) -> bool:
        """Capture the current Weave call as the evaluation context."""
        if not self.available:
            return False

        try:
            current_call = self._weave_imports['get_current_call']()
            if current_call:
                self.set_eval_call(current_call)
                return True
        except Exception as e:
            logger.warning("Failed to capture current Weave call: %s", e)

        return False

    @contextmanager
    def evaluation_context(self):
        """Set the evaluation call as active context for Weave traces."""
        if not self.available or not self.eval_call:
            yield
            return

        try:
            set_call_stack = self._weave_imports['set_call_stack']
            with set_call_stack([self.eval_call]):
                logger.debug("Set Weave evaluation call context: %s",
                             getattr(self.eval_call, 'id', str(self.eval_call)))
                yield
        except Exception as e:
            logger.warning("Failed to set Weave evaluation call context: %s", e)
            yield


# Global instance for use across the evaluation system
_eval_trace_context: EvalTraceContext | None = None


def get_eval_trace_context() -> EvalTraceContext:
    """Get the global evaluation trace context instance."""
    global _eval_trace_context
    if _eval_trace_context is None:
        # Try to create Weave-specific context first, fall back to generic
        try:
            _eval_trace_context = WeaveEvalTraceContext()
            logger.debug("Created Weave evaluation trace context")
        except Exception:
            _eval_trace_context = EvalTraceContext()
            logger.debug("Created generic evaluation trace context")

    return _eval_trace_context


def reset_eval_trace_context() -> None:
    """Reset the global evaluation trace context."""
    global _eval_trace_context
    if _eval_trace_context:
        _eval_trace_context.reset()
        _eval_trace_context = None
    logger.debug("Reset global evaluation trace context")


class EvalExporterBase(ABC):
    """
    Base class for trace exporters that want to coordinate with evaluation context.
    """

    def __init__(self):
        self.eval_context = get_eval_trace_context()
        self.eval_context.register_exporter(self)

    def track_operation_start(self) -> None:
        """Called when a trace export operation begins. Override if needed."""
        pass

    def track_operation_complete(self) -> None:
        """Called when a trace export operation completes. Override if needed."""
        pass

    @abstractmethod
    def get_current_call_context(self) -> Any:
        """Get the current call context from the tracing framework."""
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
        # Priority 1: Evaluation call from context
        eval_call = self.eval_context.get_eval_call()
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

    def __del__(self):
        """Cleanup: unregister from evaluation context."""
        try:
            if hasattr(self, 'eval_context') and self.eval_context:
                self.eval_context.unregister_exporter(self)
        except Exception:
            pass  # Ignore errors during cleanup

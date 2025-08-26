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
import copy
import logging
import weakref
from abc import abstractmethod
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import overload

from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.observability.exporter.exporter import Exporter
from nat.utils.reactive.subject import Subject
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

IsolatedAttributeT = TypeVar('IsolatedAttributeT')


class IsolatedAttribute(Generic[IsolatedAttributeT]):
    """Descriptor for copy-on-write isolation.

    This descriptor uses Python's descriptor protocol to automatically manage
    attribute isolation during object copying. It enables efficient concurrent
    execution by sharing expensive resources while isolating mutable state.

    Performance Note: This pattern shares expensive resources (HTTP clients,
    auth headers) while isolating cheap mutable state (task sets, events).
    Tasks are tracked for monitoring but don't block shutdown - they complete
    asynchronously in the event loop. Critical for high-throughput concurrent execution.

    Implementation Note: Uses Python descriptor protocol (__get__, __set__, __set_name__)
    for automatic attribute isolation on object copying.

    Example:
        class MyExporter(BaseExporter):
            # Expensive HTTP client shared across instances
            _client = expensive_http_client

            # Cheap mutable state isolated per instance
            _tasks: IsolatedAttribute[set] = IsolatedAttribute(set)

        exporter1 = MyExporter(endpoint="https://api.service.com")
        exporter2 = exporter1.create_isolated_instance(context)
        # exporter2 shares _client but has isolated _tasks tracking
    """

    def __init__(self, factory: Callable[[], IsolatedAttributeT]):
        self.factory = factory
        self.name: str | None = None
        self._private_name: str

    def __set_name__(self, owner, name):
        self.name = name
        self._private_name = f"__{name}_isolated"

    @overload
    def __get__(self, obj: None, objtype: type[Any] | None = None) -> "IsolatedAttribute[IsolatedAttributeT]":
        ...

    @overload
    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> IsolatedAttributeT:
        ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if not hasattr(obj, self._private_name):
            setattr(obj, self._private_name, self.factory())

        return getattr(obj, self._private_name)

    def __set__(self, obj, value: IsolatedAttributeT):
        setattr(obj, self._private_name, value)

    def reset_for_copy(self, obj):
        """Reset the attribute for a copied object."""
        if hasattr(obj, self._private_name):
            delattr(obj, self._private_name)


class BaseExporter(Exporter):
    """Abstract base class for event exporters with isolated copy support.

    This class provides the foundation for creating event exporters that can handle
    concurrent execution through copy-on-write isolation. It manages the lifecycle
    of event subscriptions and provides hooks for processing events.

    The class supports isolation for concurrent execution by automatically resetting
    mutable state when creating isolated copies using descriptors.

    Performance Design:
        - Export tasks run asynchronously in the event loop background
        - stop() method does not wait for background tasks to complete
        - Tasks are tracked for monitoring but cleaned up automatically
        - This keeps observability "off the hot path" for optimal performance

    Args:
        context_state (ContextState, optional): The context state to use for the exporter. Defaults to None.
    """

    # Class-level tracking for debugging and monitoring
    _instance_count: int = 0
    _active_instances: set[weakref.ref] = set()
    _isolated_instances: set[weakref.ref] = set()

    # Use descriptors for automatic isolation with proper generic typing
    _tasks: IsolatedAttribute[set[asyncio.Task]] = IsolatedAttribute(set)
    _ready_event: IsolatedAttribute[asyncio.Event] = IsolatedAttribute(asyncio.Event)
    _shutdown_event: IsolatedAttribute[asyncio.Event] = IsolatedAttribute(asyncio.Event)

    def __init__(self, context_state: ContextState | None = None):
        """Initialize the BaseExporter."""
        if context_state is None:
            context_state = ContextState.get()

        self._context_state = context_state
        self._subscription = None
        self._running = False
        # Get the event loop (set to None if not available, will be set later)
        self._loop = None
        self._is_isolated_instance = False

        # Track instance creation
        BaseExporter._instance_count += 1
        BaseExporter._active_instances.add(weakref.ref(self, self._cleanup_instance_tracking))

        # Note: _tasks, _ready_event, _shutdown_event are descriptors

    @classmethod
    def _cleanup_instance_tracking(cls, ref):
        """Cleanup callback for weakref when instance is garbage collected."""
        cls._active_instances.discard(ref)
        cls._isolated_instances.discard(ref)

    @classmethod
    def get_active_instance_count(cls) -> int:
        """Get the number of active BaseExporter instances.

        Returns:
            int: Number of active instances (cleaned up automatically via weakref)
        """
        # Clean up dead references automatically via weakref callback
        return len(cls._active_instances)

    @classmethod
    def get_isolated_instance_count(cls) -> int:
        """Get the number of active isolated BaseExporter instances.

        Returns:
            int: Number of active isolated instances
        """
        return len(cls._isolated_instances)

    @classmethod
    def log_instance_stats(cls) -> None:
        """Log current instance statistics for debugging."""
        total = cls.get_active_instance_count()
        isolated = cls.get_isolated_instance_count()
        original = total - isolated

        logger.info("BaseExporter instances - Total: %d, Original: %d, Isolated: %d", total, original, isolated)

        if isolated > 50:  # Warn if we have many isolated instances
            warning_msg = (f"High number of isolated BaseExporter instances ({isolated}). "
                           "Check for potential memory leaks.")
            logger.warning(warning_msg)

    def __del__(self):
        """Destructor with memory leak warnings.

        Warns if the exporter is being garbage collected while still running,
        which indicates stop() was never called. Task tracking is used for
        diagnostics but stop() doesn't wait for tasks to complete.

        This method is defensive against partial initialization - if the object
        failed to initialize completely, some attributes may not exist.
        """
        try:
            # Check if object was fully initialized before checking for active resources
            is_running = getattr(self, '_running', False)
            has_tasks = hasattr(self, '__tasks_isolated') and bool(getattr(self, '_tasks', None))

            if is_running or has_tasks:
                # Safely get name and task count
                try:
                    name = self.name
                except (AttributeError, TypeError):
                    # Fallback if name property fails due to missing attributes
                    name = f"{self.__class__.__name__} (partially initialized)"

                task_count = len(self._tasks) if has_tasks else 0

                logger.warning(
                    "%s: Exporter being garbage collected with active resources. "
                    "Running: %s, Tasks: %s. "
                    "Call stop() explicitly to avoid memory leaks.",
                    name,
                    is_running,
                    task_count)

        except Exception as e:
            # Last resort: log that cleanup had issues but don't raise
            # This prevents exceptions during garbage collection
            try:
                class_name = self.__class__.__name__
                logger.debug("Exception during %s cleanup: %s", class_name, e)
            except Exception:
                # If even logging fails, silently ignore to prevent GC issues
                pass

    @property
    def name(self) -> str:
        """Get the name of the exporter.

        Returns:
            str: The unique name of the exporter.
        """
        try:
            suffix = " (isolated)" if getattr(self, '_is_isolated_instance', False) else ""
            return f"{self.__class__.__name__}{suffix}"
        except AttributeError:
            # Fallback for partially initialized objects
            return f"{self.__class__.__name__} (partial)"

    @property
    def is_isolated_instance(self) -> bool:
        """Check if this is an isolated instance.

        Returns:
            bool: True if this is an isolated instance, False otherwise
        """
        return self._is_isolated_instance

    @abstractmethod
    def export(self, event: IntermediateStep) -> None:
        """This method is called on each event from the event stream to initiate the trace export.

        This is the base implementation that can be overridden by subclasses.
        By default, it does nothing - subclasses should implement their specific export logic.

        Args:
            event (IntermediateStep): The event to be exported.
        """
        pass

    @override
    def on_error(self, exc: Exception) -> None:
        """Handle an error in the event subscription.

        Args:
            exc (Exception): The error to handle.
        """
        logger.error("Error in event subscription: %s", exc, exc_info=True)

    @override
    def on_complete(self) -> None:
        """Handle the completion of the event stream.

        This method is called when the event stream is complete.
        """
        logger.info("Event stream completed. No more events will arrive.")

    def _start(self) -> Subject | None:
        """Start the exporter.

        Returns:
            Subject | None: The subject to subscribe to.
        """
        subject = self._context_state.event_stream.get()
        if subject is None:
            return None

        if not hasattr(subject, 'subscribe'):
            logger.error("Event stream subject does not support subscription")
            return None

        def on_next_wrapper(event: IntermediateStep) -> None:
            self.export(event)

        self._subscription = subject.subscribe(
            on_next=on_next_wrapper,
            on_error=self.on_error,
            on_complete=self.on_complete,
        )

        self._running = True
        self._ready_event.set()
        return subject

    async def _pre_start(self):
        """Called before the exporter starts."""
        pass

    @override
    @asynccontextmanager
    async def start(self) -> AsyncGenerator[None]:
        """Start the exporter and yield control to the caller."""
        try:
            await self._pre_start()

            if self._running:
                logger.debug("Listener already running.")
                yield
                return

            subject = self._start()
            if subject is None:
                logger.warning("No event stream available.")
                yield
                return

            yield  # let the caller do their workflow

        finally:
            await self.stop()

    async def _cleanup(self):
        """Clean up any resources."""
        pass

    async def _cancel_tasks(self):
        """Cancel all scheduled tasks.

        Note: This method is NOT called during normal stop() operation for performance.
        It's available for special cases where explicit task completion is needed.
        """
        tasks_to_cancel = set(self._tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning("Error while canceling task %s: %s", task.get_name(), e)

    async def wait_for_tasks(self, timeout: float = 5.0):
        """Wait for all tracked tasks to complete with a timeout.

        Note: This method is NOT called during normal stop() operation for performance.
        It's available for special cases where explicit task completion is needed.

        Args:
            timeout (float, optional): The timeout in seconds. Defaults to 5.0.
        """
        if not self._tasks:
            return

        try:
            # Wait for all tasks to complete with a timeout
            await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("%s: Some tasks did not complete within %s seconds", self.name, timeout)
        except Exception as e:
            logger.exception("%s: Error while waiting for tasks: %s", self.name, e)

    @override
    async def stop(self):
        """Stop the exporter immediately without waiting for background tasks.

        This method performs fast shutdown by:
        1. Setting running=False to prevent new export tasks
        2. Signaling shutdown to waiting code
        3. Cleaning up subscriptions and resources
        4. Clearing task tracking (tasks continue in event loop)

        Performance: Does not block waiting for background export tasks to complete.
        Background tasks will finish asynchronously and clean themselves up.

        Note: This method is called when the exporter is no longer needed.
        """
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        await self._cleanup()

        if self._subscription:
            self._subscription.unsubscribe()
        self._subscription = None

        self._tasks.clear()

    async def wait_ready(self):
        """Wait for the exporter to be ready.

        This method is called when the exporter is ready to export events.
        """
        await self._ready_event.wait()

    def create_isolated_instance(self, context_state: ContextState) -> "BaseExporter":
        """Create an isolated copy with automatic descriptor-based state reset.

        This method creates a shallow copy that shares expensive resources
        (HTTP clients, auth headers) while isolating mutable state through
        the IsolatedAttribute descriptor pattern.

        Args:
            context_state: The isolated context state for the new instance

        Returns:
            BaseExporter: Isolated instance sharing expensive resources
        """
        # Create shallow copy
        isolated_instance = copy.copy(self)

        # Reset context state
        isolated_instance._context_state = context_state

        # Mark as isolated instance and track it
        isolated_instance._is_isolated_instance = True
        BaseExporter._isolated_instances.add(weakref.ref(isolated_instance, self._cleanup_instance_tracking))

        # Reset IsolatedAttribute descriptors automatically
        for attr_name in dir(type(self)):
            attr_value = getattr(type(self), attr_name, None)
            if isinstance(attr_value, IsolatedAttribute):
                attr_value.reset_for_copy(isolated_instance)

        # Reset basic attributes that aren't descriptors but need isolation
        isolated_instance._subscription = None
        isolated_instance._running = False

        return isolated_instance

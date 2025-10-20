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
from datetime import datetime
from datetime import timedelta

import aiorwlock
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.workflow import Workflow
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.config import Config
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class WorkflowInfo(BaseModel):
    """
    Container for all user-specific workflow resources.

    This model tracks the complete lifecycle of a user's workflow,
    including the workflow instance itself, metadata for cleanup,
    and task management for proper resource disposal.
    """

    user_id: str = Field(description="Unique identifier for the user")
    workflow: Workflow = Field(description="The user's workflow instance")
    session_manager: SessionManager = Field(description="Session manager for handling requests")
    builder: WorkflowBuilder = Field(description="Builder instance kept alive for cleanup")
    last_activity: datetime = Field(description="Timestamp of last request from this user")
    ref_count: int = Field(default=0, description="Number of active requests using this workflow")
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, description="Lock for thread-safe ref_count updates")
    stop_event: asyncio.Event = Field(default_factory=asyncio.Event, description="Event to signal workflow shutdown")
    lifetime_task: asyncio.Task | None = Field(default=None, description="Task managing workflow lifecycle")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowManager:
    """
    Manages per-user workflow instances with lazy initialization and cleanup.

    This component is the central coordinator for the per-user workflow
    architecture. It handles:
    - Lazy creation of workflows on first user request
    - Thread-safe access to user workflows via RWLock
    - Automatic cleanup of inactive user workflows
    - Resource limits (max concurrent users)
    - Graceful shutdown of all workflows

    Thread-Safety Model:
    - RWLock for _user_workflows dict access
    - Multiple readers can check/get workflows concurrently
    - Writer has exclusive access for creation/deletion
    - Per-user lock in UserWorkflowData for ref_count updates

    Cleanup Strategy:
    - Periodic background checks (default: every 5 minutes)
    - Remove workflows inactive for > user_idle_timeout (default: 1 hour)
    - Respects ref_count to avoid cleaning active workflows
    - Cleanup on-demand when user limit is reached
    """

    def __init__(
            self,
            config: Config,
            max_users: int = 100,
            max_concurrent_requests: int = 1000,
            max_requests_per_user: int = 8,
            workflow_idle_timeout: timedelta = timedelta(hours=1),
            cleanup_check_interval: timedelta = timedelta(minutes=5),
    ):
        """
        Initialize WorkflowManager.

        Args:
            config: NAT configuration (base template for all users)
            max_users: Maximum concurrent user workflows allowed
            max_concurrent_requests: Maximum concurrent requests across all users
            max_requests_per_user: Maximum concurrent requests per user (for SessionManager)
            workflow_idle_timeout: Time after which inactive workflows are cleaned
            cleanup_check_interval: Interval for periodic cleanup checks
        """
        self._config = config
        self._max_users = max_users
        self._max_concurrent_requests = max_concurrent_requests
        self._max_requests_per_user = max_requests_per_user
        self._workflow_idle_timeout = workflow_idle_timeout
        self._cleanup_check_interval = cleanup_check_interval

        # Per-user workflow storage
        self._workflows: dict[str, WorkflowInfo] = {}
        self._rwlock = aiorwlock.RWLock()

        # Cleanup control
        self._last_cleanup: datetime = datetime.now()

        # Global request limiting
        if max_concurrent_requests > 0:
            self._request_semaphore: asyncio.Semaphore | None = asyncio.Semaphore(max_concurrent_requests)
        else:
            self._request_semaphore = None

        logger.info(
            "WorkflowManager initialized: max_users=%d, max_concurrent_requests=%d, "
            "workflow_idle_timeout=%s, cleanup_interval=%s",
            max_users,
            max_concurrent_requests,
            workflow_idle_timeout,
            cleanup_check_interval,
        )

    async def get_or_create_workflow(self, user_id: str) -> tuple[Workflow, SessionManager]:
        """
        Get existing workflow for user or create lazily.

        This is the main entry point for request handlers. It implements
        a fast path (existing workflow) and slow path (create new workflow)
        with appropriate locking to prevent race conditions.

        Fast Path (Read Lock):
        - Check if workflow exists
        - Update last_activity
        - Return immediately

        Slow Path (Write Lock):
        - Double-check existence (race prevention)
        - Verify user limit not exceeded
        - Create new workflow resources
        - Store in registry

        Args:
            user_id: Unique identifier for the user

        Returns:
            Tuple of (Workflow, SessionManager) for the user

        Raises:
            RuntimeError: If maximum concurrent users exceeded after cleanup
            RuntimeError: If workflow initialization fails or times out
            ValueError: If user_id is invalid or empty
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")

        # Throttled cleanup on access
        now = datetime.now()
        if now - self._last_cleanup > self._cleanup_check_interval:
            await self._cleanup_inactive_workflows()
            self._last_cleanup = now

        # Fast path: workflow exists (reader lock for concurrent access)
        async with self._rwlock.reader:
            if user_id in self._workflows:
                workflow_info = self._workflows[user_id]
                workflow_info.last_activity = datetime.now()
                logger.debug("Found existing workflow for user: %s", self._truncate_user_id(user_id))
                return workflow_info.workflow, workflow_info.session_manager

        # Check limit before creating
        if len(self._workflows) >= self._max_users:
            logger.info("User limit reached (%d), attempting cleanup", self._max_users)
            await self._cleanup_inactive_workflows()

        # Slow path: create workflow (writer lock for exclusive access)
        async with self._rwlock.writer:
            # Double-check after acquiring writer lock (race condition prevention)
            if user_id in self._workflows:
                workflow_info = self._workflows[user_id]
                workflow_info.last_activity = datetime.now()
                logger.debug("Workflow created by concurrent request for user: %s", self._truncate_user_id(user_id))
                return workflow_info.workflow, workflow_info.session_manager

            # Re-check limit inside writer lock
            if len(self._workflows) >= self._max_users:
                logger.warning("User limit reached (%d), rejecting new user: %s",
                               self._max_users,
                               self._truncate_user_id(user_id))
                raise RuntimeError(f"Maximum concurrent users ({self._max_users}) exceeded")

            # Create new workflow for user
            logger.info("Creating workflow for user: %s", self._truncate_user_id(user_id))
            start_time = datetime.now()

            workflow, session_manager, builder, lifetime_task, stop_event = await self._create_user_workflow(user_id)

            init_duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Workflow initialization completed for user %s in %.2fs",
                self._truncate_user_id(user_id),
                init_duration,
            )

            workflow_info = WorkflowInfo(
                user_id=user_id,
                workflow=workflow,
                session_manager=session_manager,
                builder=builder,
                last_activity=datetime.now(),
                lifetime_task=lifetime_task,
                stop_event=stop_event,
            )

            self._workflows[user_id] = workflow_info
            logger.info("Total active users: %d", len(self._workflows))

            return workflow, session_manager

    async def _create_user_workflow(
            self, user_id: str) -> tuple[Workflow, SessionManager, WorkflowBuilder, asyncio.Task, asyncio.Event]:
        """
        Create a new workflow instance for a specific user.

        This method orchestrates the creation of all per-user resources:
        1. Prepare user-specific config from base config
        2. Create WorkflowBuilder with user config
        3. Start lifetime task that manages workflow lifecycle
        4. Wait for initialization to complete (with timeout)
        5. Return all resources for storage in UserWorkflowData

        The lifetime task pattern ensures that the builder's async context
        manager is entered and exited in the same task, preventing
        asyncio cancellation issues.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Tuple of (workflow, session_manager, builder, lifetime_task, stop_event)

        Raises:
            RuntimeError: If initialization times out (default: 300s)
            RuntimeError: If initialization fails with exception
        """
        # Prepare user-specific config
        user_config = self._prepare_user_config(user_id)

        # Storage for workflow components
        ready = asyncio.Event()
        stop_event = asyncio.Event()

        async def _lifetime():
            """Lifecycle management for user workflow."""
            try:
                # Create builder
                builder = WorkflowBuilder(general_config=user_config.general)

                async with builder:
                    # Populate builder - this is where MCP auth happens per user
                    await builder.populate_builder(user_config)

                    # Build workflow
                    workflow = await builder.build()
                    session_manager = SessionManager(workflow, max_concurrency=self._max_requests_per_user)

                    # Store in closure for retrieval
                    _lifetime.workflow = workflow
                    _lifetime.session_manager = session_manager
                    _lifetime.builder = builder

                    # Signal ready
                    ready.set()

                    # Wait for stop signal
                    await stop_event.wait()

            except Exception as e:
                logger.error(
                    "Failed to initialize workflow for user %s: %s",
                    self._truncate_user_id(user_id),
                    e,
                    exc_info=True,
                )
                ready.set()  # Ensure we don't hang the waiter
                raise

        # Start lifetime task
        task = asyncio.create_task(_lifetime(), name=f"workflow-{self._truncate_user_id(user_id)}")

        # Wait for initialization with timeout
        timeout = 300  # 5 minutes
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.error("Workflow initialization timed out after %ds for user %s",
                         timeout,
                         self._truncate_user_id(user_id))
            raise RuntimeError(f"Workflow initialization timed out for user {self._truncate_user_id(user_id)}")

        # Check if initialization succeeded
        if task.done():
            try:
                await task  # Re-raise exception if failed
            except Exception as e:
                logger.error("Workflow initialization failed for user %s: %s", self._truncate_user_id(user_id), e)
                raise RuntimeError(
                    f"Workflow initialization failed for user {self._truncate_user_id(user_id)}: {e}") from e

        return (_lifetime.workflow, _lifetime.session_manager, _lifetime.builder, task, stop_event)

    def _prepare_user_config(self, user_id: str) -> Config:
        """
        Prepare user-specific config from base config.

        Creates a deep copy of the base config and applies user-specific
        modifications:
        - Sets auth_provider.default_user_id to user_id
        - Disables allow_default_user_id_for_tool_calls
        - Preserves all other configuration

        Deep copy is critical to prevent cross-user contamination.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Deep copy of config with user-specific settings
        """
        # Deep copy to avoid cross-contamination
        user_config = copy.deepcopy(self._config)

        # Inject user_id into auth provider configs
        if hasattr(user_config, "authentication") and user_config.authentication:
            for auth_name, auth_config in user_config.authentication.items():
                # Override default_user_id with actual user_id
                if hasattr(auth_config, "default_user_id"):
                    auth_config.default_user_id = user_id
                    logger.debug("Set auth provider %s default_user_id to %s",
                                 auth_name,
                                 self._truncate_user_id(user_id))

                # Disable default user fallback
                if hasattr(auth_config, "allow_default_user_id_for_tool_calls"):
                    auth_config.allow_default_user_id_for_tool_calls = False
                    logger.debug("Disabled allow_default_user_id_for_tool_calls for auth provider %s", auth_name)

        return user_config

    async def _cleanup_inactive_workflows(self) -> int:
        """
        Remove workflows for inactive users.

        Cleanup Process:
        1. Acquire write lock on _user_workflows
        2. Identify inactive users (ref_count=0, idle > timeout)
        3. Remove from dictionary
        4. Release write lock
        5. Close resources outside lock (avoid deadlock)

        This method is called:
        - Periodically via throttled check in get_or_create_workflow
        - On-demand when user limit is reached

        Returns:
            Number of workflows cleaned up
        """
        to_close: list[tuple[str, WorkflowInfo]] = []

        async with self._rwlock.writer:
            current_time = datetime.now()
            inactive_users = []

            for user_id, user_data in self._workflows.items():
                # Skip cleanup if workflow is actively being used
                if user_data.ref_count > 0:
                    continue

                idle_time = current_time - user_data.last_activity
                if idle_time > self._workflow_idle_timeout:
                    inactive_users.append(user_id)

            for user_id in inactive_users:
                logger.info(
                    "Cleaning up workflow for inactive user: %s (idle: %ds)",
                    self._truncate_user_id(user_id),
                    (current_time - self._workflows[user_id].last_activity).total_seconds(),
                )
                user_data = self._workflows.pop(user_id, None)
                if user_data:
                    to_close.append((user_id, user_data))

            if inactive_users:
                logger.info("Total active users after cleanup: %d", len(self._workflows))

        # Close workflows outside the lock
        for user_id, user_data in to_close:
            try:
                if user_data.stop_event and user_data.lifetime_task:
                    if not user_data.lifetime_task.done():
                        user_data.stop_event.set()
                        await user_data.lifetime_task
                    else:
                        logger.debug("Workflow for user %s already done", self._truncate_user_id(user_id))
                else:
                    # Fallback: manually close builder
                    logger.warning("No lifetime task for user %s, cleaning up builder directly",
                                   self._truncate_user_id(user_id))
                    await user_data.builder.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error cleaning up workflow for user %s: %s", self._truncate_user_id(user_id), e)

        return len(to_close)

    async def shutdown(self) -> None:
        """
        Shutdown all user workflows gracefully.

        Called during FastAPI lifespan shutdown. Ensures all resources
        are properly released:
        1. Signal all lifetime tasks to stop
        2. Wait for all tasks to complete
        3. Clear workflow registry

        This method does not use locks as it runs during shutdown
        when no new requests are being processed.
        """
        logger.info("Shutting down WorkflowManager...")

        # Get all workflows
        user_workflows = list(self._workflows.values())

        # Signal all to stop
        for user_data in user_workflows:
            if user_data.stop_event:
                user_data.stop_event.set()

        # Wait for all to complete
        for user_data in user_workflows:
            try:
                if user_data.lifetime_task:
                    await user_data.lifetime_task
            except Exception as e:
                logger.warning("Error shutting down workflow for user %s: %s",
                               self._truncate_user_id(user_data.user_id),
                               e)

        self._workflows.clear()
        logger.info("WorkflowManager shutdown complete")

    async def acquire_request_slot(self) -> None:
        """
        Acquire a slot for request processing from global pool.

        This method should be called before processing a request to enforce
        the global max_concurrent_requests limit. It acquires the global
        semaphore, blocking if necessary.

        Must be paired with release_request_slot() in a try/finally block.

        Raises:
            RuntimeError: If semaphore is not configured
        """
        if self._request_semaphore is None:
            return

        await self._request_semaphore.acquire()

    def release_request_slot(self) -> None:
        """
        Release a request processing slot back to global pool.

        This method should be called after request processing completes,
        regardless of success or failure. Use in try/finally block.
        """
        if self._request_semaphore is None:
            return

        self._request_semaphore.release()

    @property
    def active_user_count(self) -> int:
        """
        Current number of active user workflows.

        Returns:
            Count of users with workflows in memory
        """
        return len(self._workflows)

    @property
    def user_limit(self) -> int:
        """
        Maximum allowed concurrent users.

        Returns:
            Configured max_users value
        """
        return self._max_users

    @property
    def active_request_count(self) -> int:
        """
        Current number of actively executing requests across all users.

        Returns:
            Number of requests currently being processed
        """
        if self._request_semaphore is None:
            return 0
        return self._max_concurrent_requests - self._request_semaphore._value

    @property
    def request_limit(self) -> int:
        """
        Maximum allowed concurrent requests across all users.

        Returns:
            Configured max_concurrent_requests value
        """
        return self._max_concurrent_requests

    @staticmethod
    def _truncate_user_id(user_id: str, length: int = 8) -> str:
        """
        Truncate user ID for logging (privacy protection).

        Args:
            user_id: Full user identifier
            length: Number of characters to show before truncation

        Returns:
            Truncated user ID (e.g., "user123***" for privacy)
        """
        if len(user_id) <= length:
            return user_id
        return f"{user_id[:length]}***"

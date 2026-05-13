# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
In-memory execution store for HTTP HITL and OAuth interactive workflows.

Each *execution* tracks a single background workflow run that may be paused
while waiting for a human interaction response or an OAuth consent.
"""

import asyncio
import logging
import time
import typing
import uuid
from dataclasses import dataclass
from dataclasses import field

from nat.data_models.interactive import HumanPrompt
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive_http import ExecutionStatus

logger = logging.getLogger(__name__)

# Default TTL for completed / failed executions (seconds).
DEFAULT_EXECUTION_TTL: int = 600  # 10 minutes


@dataclass
class PendingInteraction:
    """State for a single outstanding human interaction within an execution."""
    interaction_id: str
    prompt: HumanPrompt
    future: asyncio.Future[HumanResponse] = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class PendingOAuth:
    """State for an outstanding OAuth flow within an execution."""
    auth_url: str
    oauth_state: str
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class ExecutionRecord:
    """Full state for a single execution."""
    execution_id: str
    status: ExecutionStatus = ExecutionStatus.RUNNING
    task: asyncio.Task | None = None

    # Result / error – populated on completion
    result: typing.Any = None
    error: str | None = None

    # Pending interaction (at most one at a time per execution)
    pending_interaction: PendingInteraction | None = None

    # Pending OAuth (at most one at a time per execution)
    pending_oauth: PendingOAuth | None = None

    # Signalling channel: the first time the execution needs interaction or
    # OAuth, the handler awaiting *first_outcome* is notified so it can
    # return 202 to the client.
    first_outcome: asyncio.Event = field(default_factory=asyncio.Event)

    # Lifecycle timestamps
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None


class ExecutionStore:
    """Thread-safe (asyncio-safe) in-memory store for HTTP interactive executions."""

    def __init__(self, ttl_seconds: int = DEFAULT_EXECUTION_TTL) -> None:
        self._executions: dict[str, ExecutionRecord] = {}
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    async def create_execution(self) -> ExecutionRecord:
        """Create a new execution and return its record."""
        execution_id = str(uuid.uuid4())
        record = ExecutionRecord(execution_id=execution_id)
        async with self._lock:
            self._executions[execution_id] = record
        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def get(self, execution_id: str) -> ExecutionRecord | None:
        async with self._lock:
            return self._executions.get(execution_id)

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    async def set_interaction_required(
        self,
        execution_id: str,
        prompt: HumanPrompt,
        interaction_id: str | None = None,
    ) -> PendingInteraction:
        """
        Mark the execution as waiting for human interaction.

        Returns the ``PendingInteraction`` whose ``.future`` should be
        awaited by the background task.
        """
        if interaction_id is None:
            interaction_id = str(uuid.uuid4())

        pending = PendingInteraction(interaction_id=interaction_id, prompt=prompt)

        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise KeyError(f"Execution {execution_id} not found")
            record.status = ExecutionStatus.INTERACTION_REQUIRED
            record.pending_interaction = pending
            record.first_outcome.set()

        return pending

    async def set_oauth_required(
        self,
        execution_id: str,
        auth_url: str,
        oauth_state: str,
    ) -> None:
        """Mark the execution as waiting for OAuth consent."""
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise KeyError(f"Execution {execution_id} not found")
            record.status = ExecutionStatus.OAUTH_REQUIRED
            record.pending_oauth = PendingOAuth(auth_url=auth_url, oauth_state=oauth_state)
            record.first_outcome.set()

    async def set_running(self, execution_id: str) -> None:
        """Transition back to running (after interaction / OAuth completes)."""
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise KeyError(f"Execution {execution_id} not found")
            record.status = ExecutionStatus.RUNNING
            record.pending_interaction = None
            record.pending_oauth = None

    async def set_completed(self, execution_id: str, result: typing.Any) -> None:
        """Mark the execution as successfully completed."""
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise KeyError(f"Execution {execution_id} not found")
            record.status = ExecutionStatus.COMPLETED
            record.result = result
            record.completed_at = time.monotonic()
            record.first_outcome.set()

    async def set_failed(self, execution_id: str, error: str) -> None:
        """Mark the execution as failed."""
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise KeyError(f"Execution {execution_id} not found")
            record.status = ExecutionStatus.FAILED
            record.error = error
            record.completed_at = time.monotonic()
            record.first_outcome.set()

    # ------------------------------------------------------------------
    # Interaction resolution
    # ------------------------------------------------------------------

    async def resolve_interaction(
        self,
        execution_id: str,
        interaction_id: str,
        response: HumanResponse,
    ) -> None:
        """
        Resolve a pending interaction by setting the future result.

        Raises ``KeyError`` if the execution or interaction does not exist.
        Raises ``ValueError`` if the interaction has already been resolved.
        """
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise KeyError(f"Execution {execution_id} not found")
            pending = record.pending_interaction
            if pending is None or pending.interaction_id != interaction_id:
                raise KeyError(f"Interaction {interaction_id} not found for execution {execution_id}")
            if pending.future.done():
                raise ValueError(f"Interaction {interaction_id} has already been resolved")

        # Set the result outside the lock to avoid holding it while
        # the background task resumes.
        pending.future.set_result(response)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Remove completed/failed executions older than TTL. Returns count removed."""
        now = time.monotonic()
        to_remove: list[str] = []
        async with self._lock:
            for eid, record in self._executions.items():
                if record.completed_at is not None and (now - record.completed_at) > self._ttl_seconds:
                    to_remove.append(eid)
            for eid in to_remove:
                del self._executions[eid]
        if to_remove:
            logger.debug("Cleaned up %d expired executions", len(to_remove))
        return len(to_remove)

    async def remove(self, execution_id: str) -> None:
        """Explicitly remove an execution."""
        async with self._lock:
            self._executions.pop(execution_id, None)

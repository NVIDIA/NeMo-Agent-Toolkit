# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Telemetry handler with batching, dead-letter queue, and bounded retries.

Designed for short-lived CLI processes: the typical usage is to construct the
handler as a context manager, enqueue one event, and let ``__exit__`` trigger
the final flush. A 2-second default per-request timeout caps the worst-case
additional CLI latency from telemetry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC
from datetime import datetime
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING
from typing import Any

from nat.utils.telemetry.config import NAT_TELEMETRY_DRY_RUN
from nat.utils.telemetry.config import NAT_TELEMETRY_ENDPOINT
from nat.utils.telemetry.config import SESSION_PREFIX
from nat.utils.telemetry.config import STDOUT_ENDPOINT_SENTINEL
from nat.utils.telemetry.config import TELEMETRY_ENABLED
from nat.utils.telemetry.events import TelemetryEvent
from nat.utils.telemetry.payload import QueuedEvent
from nat.utils.telemetry.payload import build_payload

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

DEFAULT_FLUSH_INTERVAL_SECONDS: float = 120.0
DEFAULT_MAX_QUEUE_SIZE: int = 50
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 2.0
"""Hard cap on per-request HTTP latency. Keeps short-lived CLI invocations
from being delayed by an unresponsive telemetry endpoint."""


def _resolve_client_version() -> str:
    """Best-effort lookup of the installed nvidia-nat-core version."""
    for package in ("nvidia-nat-core", "nvidia-nat"):
        try:
            return importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


CLIENT_VERSION: str = _resolve_client_version()


class NATTelemetryHandler:
    """Batches, flushes, and retries NAT telemetry events.

    The handler is a no-op when the global ``TELEMETRY_ENABLED`` flag is
    false: :meth:`enqueue` drops every event immediately and the timer loop
    has nothing to send. Lifecycle methods remain safe to call regardless.

    Parameters
    ----------
    flush_interval_seconds:
        Periodic flush cadence used by the background timer loop.
    max_queue_size:
        When the in-memory queue reaches this size, an early flush is
        triggered.
    max_retries:
        Maximum re-send attempts per event before it is dropped.
    request_timeout_seconds:
        Per-request HTTP timeout. Bounds telemetry-induced latency.
    source_client_version:
        Reported as ``clientVer`` in the wire envelope. Defaults to the
        installed ``nvidia-nat-core`` version.
    session_id:
        Identifier used to group related events. ``NAT_SESSION_PREFIX`` is
        prepended if set.
    """

    def __init__(
        self,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        source_client_version: str = CLIENT_VERSION,
        session_id: str = "undefined",
    ) -> None:
        self._flush_interval = flush_interval_seconds
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries
        self._request_timeout = request_timeout_seconds
        self._source_client_version = source_client_version
        self._session_id = f"{SESSION_PREFIX}{session_id}" if SESSION_PREFIX else session_id

        self._events: list[QueuedEvent] = []
        self._dlq: list[QueuedEvent] = []  # dead-letter queue for retryable failures
        self._flush_signal = asyncio.Event()
        self._timer_task: asyncio.Task | None = None
        self._running = False

    # ------------------------------------------------------------------ public

    def enqueue(self, event: TelemetryEvent) -> None:
        """Queue an event for the next flush. Silently no-ops when disabled."""
        if not TELEMETRY_ENABLED:
            return
        if not isinstance(event, TelemetryEvent):
            # Best-effort: never disrupt the caller because of a bad event.
            return
        queued = QueuedEvent(event=event, timestamp=datetime.now(UTC))
        self._events.append(queued)
        if len(self._events) >= self._max_queue_size:
            self._flush_signal.set()

    async def astart(self) -> None:
        if self._running:
            return
        self._running = True
        self._timer_task = asyncio.create_task(self._timer_loop())

    async def astop(self) -> None:
        self._running = False
        self._flush_signal.set()
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None
        await self._flush_events()

    async def aflush(self) -> None:
        self._flush_signal.set()

    def start(self) -> None:
        self._run_sync(self.astart())

    def stop(self) -> None:
        self._run_sync(self.astop())

    def flush(self) -> None:
        self._flush_signal.set()

    # ----------------------------------------------------------- context mgmt

    def __enter__(self) -> NATTelemetryHandler:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    async def __aenter__(self) -> NATTelemetryHandler:
        await self.astart()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.astop()

    # ------------------------------------------------------------- internals

    def _run_sync(self, coro: Any) -> Any:
        """Run an async coroutine from sync code, even if a loop is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return asyncio.run(coro)

    async def _timer_loop(self) -> None:
        while self._running:
            try:
                await asyncio.wait_for(self._flush_signal.wait(), timeout=self._flush_interval)
            except TimeoutError:
                pass
            self._flush_signal.clear()
            await self._flush_events()

    async def _flush_events(self) -> None:
        # Drain DLQ first so retries don't get starved by a steady stream of
        # new events.
        dlq_events, self._dlq = self._dlq, []
        new_events, self._events = self._events, []
        events_to_send = dlq_events + new_events
        if events_to_send:
            await self._send_events(events_to_send)

    async def _send_events(self, events: list[QueuedEvent]) -> None:
        if not events:
            return

        try:
            payload = build_payload(
                events,
                source_client_version=self._source_client_version,
                session_id=self._session_id,
            )
        except Exception:  # noqa: BLE001 - defensive; never raise to caller
            logger.debug("Failed to build telemetry payload", exc_info=True)
            return

        # Debug sink: write JSON lines to stderr, no network call.
        if NAT_TELEMETRY_ENDPOINT == STDOUT_ENDPOINT_SENTINEL:
            sys.stderr.write(json.dumps(payload) + "\n")
            sys.stderr.flush()
            return

        # No endpoint configured: build path is still exercised, but nothing
        # is sent. This is the default state until an ingest URL is provisioned.
        if not NAT_TELEMETRY_ENDPOINT:
            logger.debug(
                "NAT_TELEMETRY_ENDPOINT is unset; %d event(s) built but not sent",
                len(events),
            )
            return

        if NAT_TELEMETRY_DRY_RUN:
            logger.debug("NAT_TELEMETRY_DRY_RUN: would POST %d event(s) to %s", len(events), NAT_TELEMETRY_ENDPOINT)
            return

        # Lazy httpx import keeps CLI startup fast when telemetry is disabled.
        import httpx
        async with httpx.AsyncClient(timeout=self._request_timeout) as client:
            await self._send_with_client(client, events, payload)

    async def _send_with_client(
        self,
        client: httpx.AsyncClient,
        events: list[QueuedEvent],
        payload: dict[str, Any],
    ) -> None:
        try:
            response = await client.post(NAT_TELEMETRY_ENDPOINT, json=payload)
            logger.debug(
                "Telemetry POST %s -> %s (%d event(s))",
                NAT_TELEMETRY_ENDPOINT,
                response.status_code,
                len(events),
            )
            # 2xx success and 4xx (bad payload) are terminal. Retrying a 400/422
            # won't help, so drop the events.
            if response.is_success or response.status_code in (400, 422):
                return
            # 413: split the batch in half and retry recursively.
            if response.status_code == 413:
                if len(events) == 1:
                    return
                mid = len(events) // 2
                first, second = events[:mid], events[mid:]
                first_payload = build_payload(
                    first,
                    source_client_version=self._source_client_version,
                    session_id=self._session_id,
                )
                second_payload = build_payload(
                    second,
                    source_client_version=self._source_client_version,
                    session_id=self._session_id,
                )
                await self._send_with_client(client, first, first_payload)
                await self._send_with_client(client, second, second_payload)
                return
            # 408 (timeout) and 5xx: transient, queue for the next flush.
            if response.status_code == 408 or response.status_code >= 500:
                self._add_to_dlq(events)
        except Exception:  # noqa: BLE001 - any network/client failure is transient
            logger.debug("Telemetry POST failed", exc_info=True)
            self._add_to_dlq(events)

    def _add_to_dlq(self, events: list[QueuedEvent]) -> None:
        for queued in events:
            queued.retry_count += 1
            if queued.retry_count > self._max_retries:
                continue
            self._dlq.append(queued)

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
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.utils.telemetry import handler as handler_module
from nat.utils.telemetry.events import CliCommandEvent
from nat.utils.telemetry.events import TaskStatusEnum
from nat.utils.telemetry.handler import NATTelemetryHandler
from nat.utils.telemetry.payload import QueuedEvent


def _make_event(command: str = "run") -> CliCommandEvent:
    return CliCommandEvent(
        command=command,
        task_status=TaskStatusEnum.SUCCESS,
        duration_ms=1,
        exit_code=0,
        python_version="3.11.7",
    )


# ---------------------------------------------------------------- opt-out gate


def test_enqueue_is_no_op_when_telemetry_disabled():
    h = NATTelemetryHandler()
    with patch.object(handler_module, "TELEMETRY_ENABLED", False):
        h.enqueue(_make_event())
    assert h._events == []


def test_enqueue_is_no_op_for_non_event():
    h = NATTelemetryHandler()
    with patch.object(handler_module, "TELEMETRY_ENABLED", True):
        h.enqueue("not an event")  # type: ignore[arg-type]
    assert h._events == []


def test_enqueue_appends_when_enabled():
    h = NATTelemetryHandler()
    with patch.object(handler_module, "TELEMETRY_ENABLED", True):
        h.enqueue(_make_event())
    assert len(h._events) == 1


# ------------------------------------------------------------- flush behaviour


def test_max_queue_size_triggers_flush_signal():
    h = NATTelemetryHandler(max_queue_size=2)
    with patch.object(handler_module, "TELEMETRY_ENABLED", True):
        h.enqueue(_make_event())
        assert not h._flush_signal.is_set()
        h.enqueue(_make_event())
        assert h._flush_signal.is_set()


def test_flush_events_drains_dlq_first():
    h = NATTelemetryHandler()

    queued_old = QueuedEvent(event=_make_event("dlq"), timestamp=_now())
    queued_new = QueuedEvent(event=_make_event("new"), timestamp=_now())
    h._dlq.append(queued_old)
    h._events.append(queued_new)

    sent: list[list[QueuedEvent]] = []

    async def fake_send(events: list[QueuedEvent]) -> None:
        sent.append(events)

    with patch.object(h, "_send_events", side_effect=fake_send):
        asyncio.run(h._flush_events())

    assert len(sent) == 1
    sent_commands = [e.event.command for e in sent[0]]
    assert sent_commands == ["dlq", "new"]
    assert h._dlq == []
    assert h._events == []


# ------------------------------------------------------------ retry behaviour


def test_add_to_dlq_drops_after_max_retries():
    h = NATTelemetryHandler(max_retries=2)
    queued = QueuedEvent(event=_make_event(), timestamp=_now(), retry_count=0)

    h._add_to_dlq([queued])
    assert queued.retry_count == 1
    assert h._dlq == [queued]

    h._dlq = []
    h._add_to_dlq([queued])
    assert queued.retry_count == 2
    assert h._dlq == [queued]

    h._dlq = []
    h._add_to_dlq([queued])
    assert queued.retry_count == 3
    assert h._dlq == []  # exceeded max_retries; dropped


# --------------------------------------------------- HTTP send (status codes)


@pytest.mark.parametrize("status_code", [200, 204, 400, 422])
def test_send_with_client_terminal_status_does_not_dlq(status_code: int):
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())

    response = MagicMock()
    response.status_code = status_code
    response.is_success = 200 <= status_code < 300

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    asyncio.run(h._send_with_client(client, [queued], {"events": [{}]}))
    assert h._dlq == []


@pytest.mark.parametrize("status_code", [408, 500, 502, 503])
def test_send_with_client_transient_status_routes_to_dlq(status_code: int):
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())

    response = MagicMock()
    response.status_code = status_code
    response.is_success = False

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    asyncio.run(h._send_with_client(client, [queued], {"events": [{}]}))
    assert len(h._dlq) == 1
    assert h._dlq[0].retry_count == 1


def test_send_with_client_413_splits_batch():
    h = NATTelemetryHandler()
    events = [QueuedEvent(event=_make_event(f"e{i}"), timestamp=_now()) for i in range(4)]

    # First call: 413. Subsequent recursive calls return 200 to terminate.
    response_413 = MagicMock()
    response_413.status_code = 413
    response_413.is_success = False

    response_ok = MagicMock()
    response_ok.status_code = 200
    response_ok.is_success = True

    responses = iter([response_413, response_ok, response_ok])

    client = MagicMock()
    client.post = AsyncMock(side_effect=lambda *a, **kw: next(responses))

    asyncio.run(h._send_with_client(client, events, {"events": [{}]}))
    # 1 initial call (413) + 2 split calls = 3 posts
    assert client.post.call_count == 3
    assert h._dlq == []


def test_send_with_client_413_drops_when_single_event():
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())

    response = MagicMock()
    response.status_code = 413
    response.is_success = False

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    asyncio.run(h._send_with_client(client, [queued], {"events": [{}]}))
    assert h._dlq == []
    assert client.post.call_count == 1


def test_send_with_client_network_error_routes_to_dlq():
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())

    client = MagicMock()
    client.post = AsyncMock(side_effect=RuntimeError("boom"))

    asyncio.run(h._send_with_client(client, [queued], {"events": [{}]}))
    assert len(h._dlq) == 1


# -------------------------------------------------------------- debug sinks


def test_send_events_writes_to_stderr_when_endpoint_is_stdout(capsys):
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())
    with patch.object(handler_module, "NAT_TELEMETRY_ENDPOINT", "stdout"):
        asyncio.run(h._send_events([queued]))
    captured = capsys.readouterr()
    assert captured.err.strip().startswith("{")
    assert '"clientId": "nvidia-nat-cli"' in captured.err


def test_send_events_dry_run_skips_post():
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())
    with patch.object(handler_module, "NAT_TELEMETRY_DRY_RUN", True), \
         patch.object(handler_module, "NAT_TELEMETRY_ENDPOINT", "https://example.invalid/ingest"), \
         patch("httpx.AsyncClient") as mock_client:
        asyncio.run(h._send_events([queued]))
    mock_client.assert_not_called()


def test_send_events_blank_endpoint_skips_post(capsys):
    h = NATTelemetryHandler()
    queued = QueuedEvent(event=_make_event(), timestamp=_now())
    with patch.object(handler_module, "NAT_TELEMETRY_ENDPOINT", ""), \
         patch("httpx.AsyncClient") as mock_client:
        asyncio.run(h._send_events([queued]))
    mock_client.assert_not_called()
    # Nothing written to stderr either - blank endpoint is a silent no-op,
    # not a debug sink.
    captured = capsys.readouterr()
    assert captured.err == ""


# ------------------------------------------------------------ context manager


def test_context_manager_lifecycle():
    h = NATTelemetryHandler()
    with patch.object(handler_module, "TELEMETRY_ENABLED", True), \
         patch.object(h, "_send_events", new=AsyncMock()) as mock_send:
        with h:
            h.enqueue(_make_event())
        mock_send.assert_awaited()


# ----------------------------------------------------------------------- util


def _now():
    from datetime import datetime
    from datetime import timezone
    return datetime.now(timezone.utc)

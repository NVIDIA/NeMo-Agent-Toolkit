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

import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import TracebackType

import pytest

from nat.registry_handlers.process_utils import run_command

# A command that occupies a child process for a while, the way a real
# `uv pip install` / `twine upload` / `pip search` would. Built from
# sys.executable so the test is portable.
SLEEP = 0.25
SLOW_CMD = (sys.executable, "-c", f"import time; time.sleep({SLEEP})")


class _Heartbeat:
    """A trivial concurrent task used to detect event-loop starvation.

    If the loop is blocked for the duration of the call under test, this never
    gets to run and ``ticks`` stays at zero.
    """

    def __init__(self, interval: float = 0.005):
        self._interval = interval
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.ticks = 0

    async def _run(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self._interval)
            self.ticks += 1

    async def __aenter__(self) -> "_Heartbeat":
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0.02)  # let it establish a baseline
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task


# ---------------------------------------------------------------------------
# Behaviour parity with subprocess.run
# ---------------------------------------------------------------------------


async def test_run_command_returns_completed_process():
    result = await run_command(sys.executable, "-c", "print('hello')", text=True)

    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""


async def test_run_command_captures_stderr():
    result = await run_command(sys.executable, "-c", "import sys; sys.stderr.write('boom')", text=True)

    assert result.returncode == 0
    assert result.stderr == "boom"


async def test_run_command_raises_on_non_zero_exit():
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        await run_command(sys.executable, "-c", "raise SystemExit(3)")

    assert exc_info.value.returncode == 3


async def test_run_command_can_skip_the_check():
    result = await run_command(sys.executable, "-c", "raise SystemExit(3)", check=False)

    assert result.returncode == 3


async def test_run_command_text_false_returns_bytes():
    result = await run_command(sys.executable, "-c", "print('hello')")

    assert isinstance(result.stdout, bytes)


# ---------------------------------------------------------------------------
# The point of the change: the event loop must stay responsive
# ---------------------------------------------------------------------------


async def test_run_command_does_not_block_the_event_loop():
    """A concurrent task must keep running while the child process is alive.

    This is the regression guard for the registry handlers having used
    ``subprocess.run`` from async code, which froze every other in-flight
    workflow for the full duration of the command.
    """
    async with _Heartbeat() as beat:
        before = beat.ticks
        await run_command(*SLOW_CMD)
        during = beat.ticks - before

    assert during > 0, (
        "event loop was starved for the whole subprocess call; "
        "the handler blocked instead of awaiting the child process"
    )


async def test_run_command_yields_to_the_loop_repeatedly():
    """It should not merely yield once; the loop stays usable throughout."""
    async with _Heartbeat(interval=0.005) as beat:
        before = beat.ticks
        started = time.perf_counter()
        await run_command(*SLOW_CMD)
        elapsed = time.perf_counter() - started
        during = beat.ticks - before

    # Generous upper bound: a *starved* loop yields ~0 ticks, a responsive one
    # yields roughly elapsed/interval. Assert a low floor so this is not flaky
    # on a loaded CI worker.
    assert during >= 5, f"only {during} loop ticks in {elapsed:.3f}s"
    assert elapsed >= SLEEP, "command should still have taken its full time"


async def test_concurrent_run_commands_overlap():
    """Two slow commands started together should not serialise the loop."""
    async with _Heartbeat() as beat:
        before = beat.ticks
        started = time.perf_counter()
        await asyncio.gather(run_command(*SLOW_CMD), run_command(*SLOW_CMD))
        elapsed = time.perf_counter() - started
        during = beat.ticks - before

    assert during > 0
    # Serialised execution would take ~2x SLEEP; concurrent awaits should be
    # well under that. Kept loose to stay robust on slow machines.
    assert elapsed < SLEEP * 2, f"commands appear to have serialised ({elapsed:.3f}s)"


# ---------------------------------------------------------------------------
# Regressions found in review
# ---------------------------------------------------------------------------


async def test_run_command_normalises_crlf_like_text_mode():
    """text=True must apply universal newlines, like subprocess.run(text=True).

    Without this, CRLF output leaves a stray ``\\r`` and callers that strip a
    fixed-width affix (the PyPI parser strips ``(`` and ``)``) keep the closing
    bracket: ``name (1.0)\\r\\n`` yields ``1.0)`` instead of ``1.0``.
    """
    result = await run_command(sys.executable, "-c", r"import sys; sys.stdout.write('name (1.0)\r\n')", text=True)

    assert "\r" not in result.stdout
    assert result.stdout == "name (1.0)\n"

    fields = result.stdout.split("\n")[0].split(" ")
    assert fields[1][1:-1] == "1.0", "version should parse without the trailing bracket"


async def test_run_command_kills_the_child_when_cancelled():
    """Cancelling the caller must not leave the child process running.

    Cancelling ``communicate()`` does not signal the child, so an install or
    upload would keep running after its request task was gone.
    """
    import os

    marker = Path(tempfile.gettempdir()) / f"nat_run_command_cancel_{os.getpid()}.marker"
    marker.unlink(missing_ok=True)

    # child writes the marker if it is allowed to finish
    prog = f"import time; time.sleep(0.6); open({str(marker)!r}, 'w').write('finished')"
    task = asyncio.create_task(run_command(sys.executable, "-c", prog))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # give the child more than enough time to finish if it were still alive
    await asyncio.sleep(1.0)
    try:
        assert not marker.exists(), "child process was orphaned and kept running after cancellation"
    finally:
        marker.unlink(missing_ok=True)

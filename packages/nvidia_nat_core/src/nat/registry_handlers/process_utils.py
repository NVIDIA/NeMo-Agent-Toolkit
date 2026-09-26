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

import asyncio
import contextlib
import subprocess


def _decode(raw: bytes | None) -> str:
    """Decode captured output the way ``subprocess.run(text=True)`` would.

    Universal-newline mode translates ``\\r\\n`` and lone ``\\r`` to ``\\n``, which
    plain ``bytes.decode()`` does not do. Callers that split on ``"\\n"`` and then
    strip a fixed-width affix (e.g. parsing ``name (1.0)``) get a trailing ``)``
    or ``\\r`` without the translation.
    """
    if raw is None:
        return ""
    return raw.decode(errors="replace").replace("\r\n", "\n").replace("\r", "\n")


async def run_command(
    *args: str, check: bool = True, text: bool = False, capture: bool = True
) -> subprocess.CompletedProcess:
    """Run an external command without blocking the event loop.

    The registry handlers shell out to package managers (``uv pip install``,
    ``twine upload``, ``pip search``). Those calls are network bound and can
    take seconds to minutes, so invoking them with :func:`subprocess.run` from
    an async handler stalls the entire event loop for the duration and every
    other in-flight agent workflow stops making progress.

    This is the :func:`asyncio` equivalent of ``subprocess.run(cmd, check=True,
    capture_output=True)``, and deliberately mirrors its behaviour so callers
    keep the same error types and messages:

    * a non-zero exit raises :class:`subprocess.CalledProcessError`
    * ``text=True`` decodes the output to ``str`` instead of ``bytes``

    Args:
        *args: Program and arguments, as passed to :func:`subprocess.run`.
        check: Raise :class:`subprocess.CalledProcessError` on a non-zero exit.
        text: Decode ``stdout``/``stderr`` to ``str``.
        capture: Capture ``stdout``/``stderr`` into the returned object. Pass
            ``False`` for commands that may prompt on the terminal, such as
            ``twine upload`` to a private index, which calls ``input()`` when
            no username is configured. Capturing hides that prompt while the
            child still blocks waiting for an answer, and stdin is inherited
            either way.

    Returns:
        A :class:`subprocess.CompletedProcess` for the finished command. With
        ``capture=False`` the output streams straight to the terminal and
        ``stdout``/``stderr`` are ``None``.

    Raises:
        subprocess.CalledProcessError: If ``check`` is set and the exit code is non-zero.
    """
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE if capture else None,
        stderr=asyncio.subprocess.PIPE if capture else None,
    )
    if not capture:
        # Output is inherited, so there is nothing to gather. wait() still yields
        # to the loop, and cancelling it must not leave the child running.
        try:
            await process.wait()
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            raise
        if check and process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, args)
        return subprocess.CompletedProcess(args=args, returncode=process.returncode)

    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        # Cancelling communicate() does not signal the child, so an install or
        # upload would keep running after its request task is gone. Kill it and
        # reap it before propagating, otherwise it is orphaned.
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
        raise

    if text:
        # Match subprocess.run(text=True): str output in universal-newline mode,
        # so CRLF is normalised to LF just as the text wrapper would do it.
        stdout = _decode(stdout)
        stderr = _decode(stderr)

    completed = subprocess.CompletedProcess(args=args, returncode=process.returncode, stdout=stdout, stderr=stderr)

    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args, output=stdout, stderr=stderr)

    return completed

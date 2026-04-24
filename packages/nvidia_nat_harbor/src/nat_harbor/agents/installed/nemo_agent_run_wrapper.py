#!/usr/bin/env python3
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

"""Wrapper to run a NAT workflow and emit plain text output."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import shlex
import sys
import traceback
from uuid import uuid4

_log_level_str = os.environ.get("NVIDIA_NAT_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _log_level_str, logging.WARNING))
os.environ.setdefault("NAT_LOG_LEVEL", _log_level_str)


def to_bool(value: str | None) -> bool:
    """Parse env-var style boolean strings."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def maybe_enable_debugpy() -> None:
    """Optionally enable debugpy attach for breakpoint debugging."""
    port_str = os.environ.get("NVIDIA_NAT_DEBUGPY_PORT")
    if not port_str:
        return

    try:
        port = int(port_str)
    except ValueError:
        print(
            f"[nemo-agent-wrapper] Ignoring invalid NVIDIA_NAT_DEBUGPY_PORT={port_str!r}",
            file=sys.stderr,
        )
        return

    host = os.environ.get("NVIDIA_NAT_DEBUGPY_HOST", "127.0.0.1")
    wait_for_client = to_bool(os.environ.get("NVIDIA_NAT_DEBUGPY_WAIT_FOR_CLIENT"))

    try:
        debugpy = importlib.import_module("debugpy")
    except ImportError:
        print(
            "[nemo-agent-wrapper] debugpy not installed; cannot enable debugger attach.",
            file=sys.stderr,
        )
        return

    try:
        debugpy.listen((host, port))
        print(
            f"[nemo-agent-wrapper] debugpy listening on {host}:{port} "
            f"(wait_for_client={wait_for_client})",
            file=sys.stderr,
        )
        if wait_for_client:
            debugpy.wait_for_client()
            print("[nemo-agent-wrapper] debugpy client attached.", file=sys.stderr)
    except Exception:
        print("[nemo-agent-wrapper] Failed to initialize debugpy:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def _write_trajectory(intermediate_steps_dicts: list[dict], output_path: str) -> None:
    """Convert intermediate steps to ATIF trajectory and write to JSON file."""
    from nat.plugins.eval.utils.intermediate_step_adapter import IntermediateStepAdapter
    from nat.utils.atif_converter import IntermediateStepToATIFConverter

    adapter = IntermediateStepAdapter()
    steps = adapter.validate_intermediate_steps(intermediate_steps_dicts)

    converter = IntermediateStepToATIFConverter()
    atif_trajectory = converter.convert(
        steps,
        session_id=str(uuid4()),
        agent_name="nemo-agent",
    )

    trajectory_dict = atif_trajectory.to_json_dict(exclude_none=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(trajectory_dict, file, indent=2, ensure_ascii=False)


def normalize_result_text(raw_text: str) -> str:
    """Normalize NAT output for text-only benchmark verifiers."""
    text = raw_text.strip()
    if not text:
        return text

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    try:
        parts = shlex.split(text)
    except ValueError:
        return text

    if not parts:
        return text

    if parts[0] == "echo" and ">" in parts and "/app/result.json" in parts and len(parts) >= 2:
        payload = parts[1]
        try:
            json.loads(payload)
            return payload
        except json.JSONDecodeError:
            return text

    return text


async def main(
    config_path: str, instruction: str, trajectory_path: str | None = None
) -> None:
    """Run the NAT workflow and print normalized result text."""
    maybe_enable_debugpy()

    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins
    from nat.runtime.session import SessionManager
    from nat.utils.io.yaml_tools import yaml_load

    discover_and_register_plugins(PluginTypes.COMPONENT)

    config_dict = yaml_load(config_path)
    config = Config(**config_dict)

    async with WorkflowBuilder.from_config(config) as builder:
        session_manager = await SessionManager.create(config=config, shared_builder=builder)
        async with session_manager.session(user_id="harbor") as session:
            async with session.run(instruction) as runner:
                intermediate_task = None
                if trajectory_path:
                    try:
                        from nat.builder.runtime_event_subscriber import pull_intermediate

                        intermediate_task = asyncio.ensure_future(pull_intermediate())
                    except Exception:
                        print(
                            "[nemo-agent-wrapper] Failed to start trajectory collection:",
                            file=sys.stderr,
                        )
                        traceback.print_exc(file=sys.stderr)

                result = await runner.result()
                print(normalize_result_text(str(result)))

                if intermediate_task is not None and trajectory_path is not None:
                    try:
                        intermediate_steps_dicts = await intermediate_task
                        _write_trajectory(intermediate_steps_dicts, trajectory_path)
                    except Exception:
                        print(
                            "[nemo-agent-wrapper] Failed to write trajectory:",
                            file=sys.stderr,
                        )
                        traceback.print_exc(file=sys.stderr)

        await session_manager.shutdown()


if __name__ == "__main__":
    trajectory_path = None
    args = sys.argv[1:]
    if "--trajectory-output" in args:
        idx = args.index("--trajectory-output")
        if idx + 1 < len(args):
            trajectory_path = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            print("--trajectory-output requires a path argument", file=sys.stderr)
            sys.exit(1)

    if len(args) < 2:
        print(
            f"Usage: {sys.argv[0]} <config_file> <instruction> [--trajectory-output <path>]",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = args[0]
    instruction = " ".join(args[1:])
    asyncio.run(main(config_path, instruction, trajectory_path))


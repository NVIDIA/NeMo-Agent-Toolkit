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
"""Default inline runner implementation for phase-1 library mode."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from nat_harbor.agents.installed.library_mode import NemoInlineRunner
from nat_harbor.agents.installed.library_mode import NemoInlineRunnerInput
from nat_harbor.agents.installed.library_mode import NemoInlineRunnerResult
from nat_harbor.agents.installed.nemo_agent_run_wrapper import _write_trajectory
from nat_harbor.agents.installed.nemo_agent_run_wrapper import normalize_result_text


@contextmanager
def _temporary_environment(env: dict[str, str]) -> Generator[None, None, None]:
    """Temporarily overlay environment variables for inline workflow execution."""
    previous: dict[str, str | None] = {}
    for key, value in env.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


class DefaultNemoInlineRunner(NemoInlineRunner):
    """Run NAT workflows in-process and write phase-1 artifacts."""

    async def run(self, request: NemoInlineRunnerInput) -> NemoInlineRunnerResult:
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.data_models.config import Config
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.runtime.session import SessionManager
        from nat.utils.io.yaml_tools import yaml_load

        discover_and_register_plugins(PluginTypes.COMPONENT)

        config_dict = yaml_load(request.config_file)
        config = Config(**config_dict)

        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = request.artifact_dir / "trajectory.json"
        output_path = request.artifact_dir / "nemo-agent-output.txt"

        result_text = ""
        intermediate_steps_dicts: list[dict[str, Any]] = []

        with _temporary_environment(request.env):
            async with WorkflowBuilder.from_config(config) as builder:
                session_manager = await SessionManager.create(config=config, shared_builder=builder)
                try:
                    async with session_manager.session(user_id="harbor") as session:
                        async with session.run(request.instruction) as runner:
                            intermediate_task = None
                            try:
                                from nat.builder.runtime_event_subscriber import pull_intermediate
                                intermediate_task = asyncio.ensure_future(pull_intermediate())
                            except Exception:
                                intermediate_task = None

                            result = await runner.result()
                            result_text = normalize_result_text(str(result))
                            if intermediate_task is not None:
                                intermediate_steps_dicts = await intermediate_task
                finally:
                    await session_manager.shutdown()

        output_path.write_text(result_text, encoding="utf-8")
        if intermediate_steps_dicts:
            _write_trajectory(intermediate_steps_dicts, str(trajectory_path))

        return NemoInlineRunnerResult(
            output_text=result_text,
            trajectory_path=trajectory_path,
            steps_count=len(intermediate_steps_dicts),
            runner_details={
                "library_mode": True, "artifact_dir": str(request.artifact_dir)
            },
        )

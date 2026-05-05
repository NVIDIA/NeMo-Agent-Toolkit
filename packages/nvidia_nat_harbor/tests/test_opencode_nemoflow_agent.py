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
"""Unit tests for the experimental OpenCode NeMo-Flow Harbor wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from harbor.models.agent.context import AgentContext

from nat_harbor.agents.installed.opencode_nemoflow import OpenCodeNeMoFlow


def _make_agent(tmp_path: Path, **kwargs) -> OpenCodeNeMoFlow:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir(parents=True)
    return OpenCodeNeMoFlow(
        logs_dir=logs_dir,
        model_name="nvidia-frontier/opus-frontier",
        nemo_flow_repo=str(tmp_path / "nemo-flow"),
        **kwargs,
    )


def _write_minimal_opencode_stdout(logs_dir: Path) -> None:
    (logs_dir / "opencode.txt").write_text(
        "\n".join([
            '{"type":"step_start","sessionID":"sid","timestamp":1}',
            '{"type":"step_finish","part":{"cost":0,"tokens":{"input":0,"output":0}}}',
            "",
        ]),
        encoding="utf-8",
    )


def test_populate_context_converts_nemoflow_atof_to_atif(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    _write_minimal_opencode_stdout(agent.logs_dir)
    atof_path = agent.logs_dir / "nemo-flow-atof" / "events.jsonl"
    atof_path.parent.mkdir(parents=True)
    atof_path.write_text('{"kind":"mark","uuid":"event-1","name":"sample","category":"unknown"}\n', encoding="utf-8")
    converted_atif_path = agent.logs_dir / "nemo-flow-atof-atif" / "trajectory.json"
    context = AgentContext()

    with patch.object(agent, "_convert_atof_to_atif", return_value=converted_atif_path) as convert_mock:
        converted_atif_path.parent.mkdir(parents=True)
        converted_atif_path.write_text('{"schema_version":"ATIF-v1.7","steps":[]}\n', encoding="utf-8")

        agent.populate_context_post_run(context)

    convert_mock.assert_called_once_with(atof_path, converted_atif_path)
    assert context.metadata is not None
    assert context.metadata["nemo_flow_atof_path"] == str(atof_path)
    assert context.metadata["nemo_flow_atof_exists"] is True
    assert context.metadata["nemo_flow_converted_atif_path"] == str(converted_atif_path)
    assert context.metadata["nemo_flow_converted_atif_exists"] is True


def test_populate_context_can_treat_atof_conversion_as_best_effort(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, fail_nemoflow_atof_conversion=False)
    _write_minimal_opencode_stdout(agent.logs_dir)
    atof_path = agent.logs_dir / "nemo-flow-atof" / "events.jsonl"
    atof_path.parent.mkdir(parents=True)
    atof_path.write_text('{"kind":"mark","uuid":"event-1","name":"sample","category":"unknown"}\n', encoding="utf-8")
    context = AgentContext()

    with patch.object(agent, "_convert_atof_to_atif", side_effect=ValueError("bad stream")):
        agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["nemo_flow_atof_exists"] is True
    assert context.metadata["nemo_flow_converted_atif_exists"] is False

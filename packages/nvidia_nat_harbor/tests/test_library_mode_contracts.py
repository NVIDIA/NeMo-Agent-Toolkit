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
"""Tests for phase-1 library mode contracts."""

from __future__ import annotations

import json
from pathlib import Path

from nat_harbor.agents.installed.inline_runner import _write_minimal_trajectory
from nat_harbor.agents.installed.library_mode import NemoInlineRunnerInput
from nat_harbor.agents.installed.library_mode import NemoInlineRunnerResult
from nat_harbor.verifier.inline_verifier import InlineVerifierRequest
from nat_harbor.verifier.inline_verifier import InlineVerifierResult
from nat_harbor.verifier.inline_verifier import build_inline_verifier_metadata


def test_nemo_inline_runner_contracts_roundtrip(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "agent"
    config_path = tmp_path / "config.yml"
    result_path = artifact_dir / "trajectory.json"
    request = NemoInlineRunnerInput(
        instruction="hello",
        config_file=config_path,
        artifact_dir=artifact_dir,
        env={"NVIDIA_API_KEY": "stub"},
    )
    result = NemoInlineRunnerResult(
        output_text="hi",
        trajectory_path=result_path,
        steps_count=3,
        runner_details={"lane": "inline"},
    )
    assert request.config_file == config_path
    assert request.artifact_dir == artifact_dir
    assert result.trajectory_path == result_path
    assert result.steps_count == 3


def test_inline_verifier_contracts_roundtrip(tmp_path: Path) -> None:
    verifier_dir = tmp_path / "verifier"
    request = InlineVerifierRequest(
        trajectory_path=tmp_path / "agent" / "trajectory.json",
        evaluator_kind="trajectory",
        evaluator_ref=None,
        config_file=str(tmp_path / "config.yml"),
        evaluator_name="trajectory_eval",
        verifier_output_dir=verifier_dir,
    )
    result = InlineVerifierResult(
        reward=1.0,
        rewards={"reward": 1.0},
        details={"mode": "inline"},
        reward_json_path=verifier_dir / "reward.json",
        reward_txt_path=verifier_dir / "reward.txt",
        details_json_path=verifier_dir / "details.json",
    )
    assert request.evaluator_kind == "trajectory"
    assert result.rewards["reward"] == 1.0


def test_build_inline_verifier_metadata_has_phase1_defaults(tmp_path: Path) -> None:
    metadata = build_inline_verifier_metadata(
        evaluator_mode="builtin",
        trajectory_path=tmp_path / "agent" / "trajectory.json",
        reward_json_path=tmp_path / "verifier" / "reward.json",
        details_json_path=tmp_path / "verifier" / "details.json",
    )
    assert metadata["inline_mode"] is True
    assert metadata["evaluator_mode"] == "builtin"
    assert metadata["trajectory_path"].endswith("trajectory.json")


def test_write_minimal_trajectory_creates_empty_atif_artifact(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.json"

    _write_minimal_trajectory(trajectory_path)

    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "ATIF-v1.6"
    assert payload["agent"]["name"] == "nemo-agent"
    assert payload["steps"] == []

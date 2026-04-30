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
"""Tests for inline verifier driver behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nat_harbor.verifier.inline_verifier import DefaultInlineVerifierDriver
from nat_harbor.verifier.inline_verifier import InlineVerifierError
from nat_harbor.verifier.inline_verifier import InlineVerifierRequest


def _write_minimal_atif(path: Path) -> None:
    payload = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-1",
        "agent": {
            "name": "bridge-test-agent",
            "version": "1.0.0",
        },
        "steps": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inline_verifier_success_writes_harbor_compatible_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "trajectory.json"
    verifier_dir = tmp_path / "verifier"
    _write_minimal_atif(artifact_path)

    async def _eval_stub(**kwargs):
        assert kwargs["evaluator_kind"] == "trajectory"
        return 0.75, {"evaluator_mode": "builtin", "items": 1}

    monkeypatch.setattr("nat_harbor.verifier.inline_verifier.evaluate_artifact", _eval_stub)
    request = InlineVerifierRequest(
        trajectory_path=artifact_path,
        evaluator_kind="trajectory",
        evaluator_ref=None,
        config_file=str(tmp_path / "config.yml"),
        evaluator_name="trajectory",
        verifier_output_dir=verifier_dir,
    )
    result = asyncio.run(DefaultInlineVerifierDriver().verify(request))
    assert result.reward == pytest.approx(0.75)
    assert result.rewards == {"reward": pytest.approx(0.75)}
    details = json.loads(result.details_json_path.read_text(encoding="utf-8"))
    assert details["result"] == "evaluated"
    assert details["evaluator_details"]["items"] == 1
    assert details["inline_metadata"]["inline_mode"] is True
    reward_payload = json.loads(result.reward_json_path.read_text(encoding="utf-8"))
    assert reward_payload == {"reward": pytest.approx(0.75)}


def test_inline_verifier_missing_artifact_fail_raises(tmp_path: Path) -> None:
    request = InlineVerifierRequest(
        trajectory_path=tmp_path / "missing.json",
        evaluator_kind="trajectory",
        evaluator_ref=None,
        config_file=str(tmp_path / "config.yml"),
        evaluator_name="trajectory",
        verifier_output_dir=tmp_path / "verifier",
        fallback_mode="fail",
    )
    with pytest.raises(InlineVerifierError, match="missing"):
        asyncio.run(DefaultInlineVerifierDriver().verify(request))
    details = json.loads((request.verifier_output_dir / "details.json").read_text(encoding="utf-8"))
    assert details["result"] == "missing_artifact_fail"
    assert (request.verifier_output_dir / "reward.json").exists() is False


def test_inline_verifier_missing_artifact_raw_fallback_writes_details_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier_dir = tmp_path / "verifier"
    raw_output_path = tmp_path / "nemo-agent-output.txt"
    raw_output_path.write_text("agent output", encoding="utf-8")
    driver = DefaultInlineVerifierDriver()
    write_calls = 0
    write_details = driver._write_details

    def _write_details_once(output_dir: Path, details: dict[str, object]) -> Path:
        nonlocal write_calls
        write_calls += 1
        return write_details(output_dir, details)

    monkeypatch.setattr(driver, "_write_details", _write_details_once)
    request = InlineVerifierRequest(
        trajectory_path=tmp_path / "missing.json",
        evaluator_kind="trajectory",
        evaluator_ref=None,
        config_file=str(tmp_path / "config.yml"),
        evaluator_name="trajectory",
        verifier_output_dir=verifier_dir,
        fallback_mode="raw_output",
        raw_output_path=raw_output_path,
    )

    result = asyncio.run(driver.verify(request))

    assert write_calls == 1
    assert result.reward == pytest.approx(0.0)
    details = json.loads(result.details_json_path.read_text(encoding="utf-8"))
    assert details["result"] == "raw_fallback_missing_artifact"
    assert details["raw_output_exists"] is True
    assert details["inline_metadata"]["evaluator_mode"] == "raw_output_fallback"


def test_inline_verifier_raw_output_fallback_on_evaluator_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "trajectory.json"
    verifier_dir = tmp_path / "verifier"
    raw_output_path = tmp_path / "nemo-agent-output.txt"
    _write_minimal_atif(artifact_path)
    raw_output_path.write_text("agent output", encoding="utf-8")

    async def _eval_error(**kwargs):
        del kwargs
        raise RuntimeError("stub evaluator failure")

    monkeypatch.setattr("nat_harbor.verifier.inline_verifier.evaluate_artifact", _eval_error)
    request = InlineVerifierRequest(
        trajectory_path=artifact_path,
        evaluator_kind="trajectory",
        evaluator_ref=None,
        config_file=str(tmp_path / "config.yml"),
        evaluator_name="trajectory",
        verifier_output_dir=verifier_dir,
        fallback_mode="raw_output",
        raw_output_path=raw_output_path,
    )
    result = asyncio.run(DefaultInlineVerifierDriver().verify(request))
    assert result.reward == pytest.approx(0.0)
    details = json.loads(result.details_json_path.read_text(encoding="utf-8"))
    assert details["result"] == "raw_fallback_evaluator_error"
    assert details["raw_output_exists"] is True
    assert details["inline_metadata"]["evaluator_mode"] == "raw_output_fallback"

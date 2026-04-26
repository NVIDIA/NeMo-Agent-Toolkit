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
"""Tests for NAT Harbor ATIF bridge runner and evaluator adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nat_harbor.verifier import bridge_runner
from nat_harbor.verifier.evaluator_adapter import BridgeEvaluatorError
from nat_harbor.verifier.evaluator_adapter import evaluate_artifact_sync


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


def test_artifact_found_builtin_evaluator_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "trajectory.json"
    _write_minimal_atif(artifact_path)

    async def _builtin_stub(**kwargs):
        del kwargs
        return 0.8, {"lane": "trajectory"}

    monkeypatch.setattr("nat_harbor.verifier.evaluator_adapter._run_builtin_evaluator", _builtin_stub)
    reward, details = evaluate_artifact_sync(
        artifact_path=artifact_path,
        evaluator_kind="trajectory",
        config_file="/tmp/fake-config.yml",
    )
    assert reward == pytest.approx(0.8)
    assert details["lane"] == "trajectory"


def test_custom_evaluator_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "trajectory.json"
    _write_minimal_atif(artifact_path)

    def _custom_fn(atif_samples):
        assert len(atif_samples) == 1
        return {"reward": 1.0, "details": {"items": len(atif_samples)}}

    monkeypatch.setattr("nat_harbor.verifier.evaluator_adapter._load_callable", lambda _: _custom_fn)
    reward, details = evaluate_artifact_sync(
        artifact_path=artifact_path,
        evaluator_kind="custom",
        evaluator_ref="dummy.module:fn",
    )
    assert reward == pytest.approx(1.0)
    assert details["items"] == 1
    assert details["evaluator_mode"] == "custom"


def test_missing_artifact_fallback_fail(tmp_path: Path) -> None:
    output_dir = tmp_path / "verifier"
    code = bridge_runner.run_bridge(
        artifact_path="does-not-exist.json",
        evaluator_kind="trajectory",
        evaluator_ref=None,
        output_dir=str(output_dir),
        fallback_mode="fail",
        config_file=None,
        evaluator_name=None,
    )
    assert code == 1
    details = json.loads((output_dir / "details.json").read_text(encoding="utf-8"))
    assert details["result"] == "missing_artifact_fail"
    assert (output_dir / "reward.json").exists() is False


def test_missing_artifact_fallback_raw_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "verifier"
    monkeypatch.setattr(
        "nat_harbor.verifier.bridge_runner._raw_output_details",
        lambda: {
            "raw_output_path": "/logs/agent/nemo-agent-output.txt",
            "raw_output_exists": True,
            "raw_output_prefix": "stub", },
    )
    code = bridge_runner.run_bridge(
        artifact_path="still-missing.json",
        evaluator_kind="trajectory",
        evaluator_ref=None,
        output_dir=str(output_dir),
        fallback_mode="raw_output",
        config_file=None,
        evaluator_name=None,
    )
    assert code == 0
    reward_payload = json.loads((output_dir / "reward.json").read_text(encoding="utf-8"))
    assert reward_payload == {"reward": 0.0}
    details = json.loads((output_dir / "details.json").read_text(encoding="utf-8"))
    assert details["result"] == "raw_fallback_missing_artifact"


def test_invalid_evaluator_ref(tmp_path: Path) -> None:
    artifact_path = tmp_path / "trajectory.json"
    _write_minimal_atif(artifact_path)
    with pytest.raises(BridgeEvaluatorError, match="Invalid evaluator ref"):
        evaluate_artifact_sync(
            artifact_path=artifact_path,
            evaluator_kind="custom",
            evaluator_ref="not-a-valid-ref",
        )


def test_reward_details_output_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_path = tmp_path / "trajectory.json"
    output_dir = tmp_path / "verifier"
    _write_minimal_atif(artifact_path)

    def _eval_stub(**kwargs):
        del kwargs
        return 0.42, {"items": 1}

    monkeypatch.setattr("nat_harbor.verifier.bridge_runner.evaluate_artifact_sync", _eval_stub)
    code = bridge_runner.run_bridge(
        artifact_path=str(artifact_path),
        evaluator_kind="trajectory",
        evaluator_ref=None,
        output_dir=str(output_dir),
        fallback_mode="fail",
        config_file="/tmp/fake-config.yml",
        evaluator_name="trajectory",
    )
    assert code == 0
    reward_payload = json.loads((output_dir / "reward.json").read_text(encoding="utf-8"))
    assert list(reward_payload.keys()) == ["reward"]
    assert reward_payload["reward"] == pytest.approx(0.42)
    details = json.loads((output_dir / "details.json").read_text(encoding="utf-8"))
    assert details["result"] == "evaluated"
    assert details["evaluator_details"]["items"] == 1


def test_builtin_arbitrary_kind_dispatches_via_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "trajectory.json"
    _write_minimal_atif(artifact_path)

    async def _builtin_stub(**kwargs):
        assert kwargs["evaluator_kind"] == "ragas"
        return 0.6, {"lane": "ragas"}

    monkeypatch.setattr("nat_harbor.verifier.evaluator_adapter._run_builtin_evaluator", _builtin_stub)
    reward, details = evaluate_artifact_sync(
        artifact_path=artifact_path,
        evaluator_kind="ragas",
        config_file="/tmp/fake-config.yml",
    )
    assert reward == pytest.approx(0.6)
    assert details["lane"] == "ragas"

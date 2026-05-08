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
"""Tests for post-run ATIF trajectory scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nat_harbor.smoke import score_atif_trajectories
from nat_harbor.smoke.score_atif_trajectories import NEMOFLOW_HIGHER
from nat_harbor.smoke.score_atif_trajectories import NOT_SCORED
from nat_harbor.smoke.score_atif_trajectories import ScoringConfig
from nat_harbor.smoke.score_atif_trajectories import main
from nat_harbor.smoke.score_atif_trajectories import score_trial


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _trajectory(*, include_user: bool, tools: list[str], final_message: str) -> dict:
    steps = []
    step_id = 1
    if include_user:
        steps.append({"step_id": step_id, "source": "user", "message": "Fix the failing test."})
        step_id += 1
    for tool in tools:
        steps.append({
            "step_id":
                step_id,
            "source":
                "agent",
            "message":
                "(tool use)",
            "tool_calls": [{
                "tool_call_id": f"call-{step_id}",
                "function_name": tool,
                "arguments": {
                    "path": "file.py"
                },
            }],
        })
        step_id += 1
    steps.append({"step_id": step_id, "source": "agent", "message": final_message})
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "agent": {
            "name": "opencode",
            "version": "test",
        },
        "steps": steps,
    }


def _write_trial(tmp_path: Path) -> Path:
    trial_dir = tmp_path / "django__django-13741__abc123"
    _write_json(
        trial_dir / "agent" / "trajectory.json",
        _trajectory(include_user=False, tools=["read", "edit"], final_message="Done."),
    )
    _write_json(
        trial_dir / "agent" / "nemo-flow-atof-atif" / "trajectory.json",
        _trajectory(include_user=True, tools=["glob", "read", "edit"], final_message="Done."),
    )
    _write_json(trial_dir / "result.json", {"verifier_result": {"rewards": {"reward": 1.0}}})
    return trial_dir


def test_score_trial_no_llm_reports_deterministic_comparison(tmp_path: Path) -> None:
    trial_dir = _write_trial(tmp_path)

    row = score_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        scoring_config=ScoringConfig(
            evaluator_kind="trajectory",
            evaluator_ref=None,
            config_file=None,
            evaluator_name=None,
            score_threshold=0.05,
            score_timeout_sec=120,
            no_llm=True,
        ),
    )

    assert row.task_id == "django__django-13741"
    assert row.swebench_reward == pytest.approx(1.0)
    assert row.deterministic_comparison == "match (richer)"
    assert row.score_category == NOT_SCORED
    assert row.native_score is None
    assert row.nemoflow_score is None


def test_score_trial_scores_temp_samples_with_shared_user_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trial_dir = _write_trial(tmp_path)
    seen_samples: list[dict] = []

    def _fake_evaluate_artifact_sync(**kwargs):
        sample = json.loads(Path(kwargs["artifact_path"]).read_text(encoding="utf-8"))
        seen_samples.append(sample)
        lane = sample["metadata"]["lane"]
        return (0.9 if lane == "nemoflow" else 0.8), {"lane": lane}

    monkeypatch.setattr(score_atif_trajectories, "evaluate_artifact_sync", _fake_evaluate_artifact_sync)

    row = score_trial(
        trial_dir,
        native_rel=Path("agent/trajectory.json"),
        candidate_rel=Path("agent/nemo-flow-atof-atif/trajectory.json"),
        scoring_config=ScoringConfig(
            evaluator_kind="trajectory",
            evaluator_ref=None,
            config_file="config.yml",
            evaluator_name="trajectory_eval",
            score_threshold=0.05,
            score_timeout_sec=120,
            no_llm=False,
        ),
    )

    assert row.native_score == pytest.approx(0.8)
    assert row.nemoflow_score == pytest.approx(0.9)
    assert row.score_delta == pytest.approx(0.1)
    assert row.score_category == NEMOFLOW_HIGHER
    assert [sample["metadata"]["lane"] for sample in seen_samples] == ["native", "nemoflow"]
    native_sample = seen_samples[0]
    assert native_sample["trajectory"]["steps"][0]["source"] == "user"
    assert native_sample["trajectory"]["steps"][0]["message"] == "Fix the failing test."
    assert native_sample["output_obj"] == "Done."


def test_main_writes_csv_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_trial(tmp_path)
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "score_atif_trajectories",
            "--job-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--no-llm",
        ],
    )

    main()

    csv_text = (output_dir / "atif-trajectory-scores.csv").read_text(encoding="utf-8")
    markdown_text = (output_dir / "atif-trajectory-scores.md").read_text(encoding="utf-8")
    assert "django__django-13741" in csv_text
    assert "match (richer)" in markdown_text

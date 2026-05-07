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
"""Post-run scoring for native and ATOF-derived ATIF trajectory artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nat_harbor.smoke.compare_atif_tools import ToolSequenceComparison
from nat_harbor.smoke.compare_atif_tools import compare_atif_tool_sequences
from nat_harbor.verifier.evaluator_adapter import evaluate_artifact_sync

DEFAULT_NATIVE_REL = Path("agent/trajectory.json")
DEFAULT_CANDIDATE_REL = Path("agent/nemo-flow-atof-atif/trajectory.json")
SCORE_SAME = "score_same"
NEMOFLOW_HIGHER = "nemoflow_higher"
NATIVE_HIGHER = "native_higher"
NOT_SCORED = "not_scored"


@dataclass(frozen=True)
class ScoringConfig:
    """Evaluator settings for one scoring pass."""

    evaluator_kind: str
    evaluator_ref: str | None
    config_file: str | None
    evaluator_name: str | None
    score_threshold: float
    score_timeout_sec: int | None
    no_llm: bool


@dataclass(frozen=True)
class TrialScoreRow:
    """Report row for one Harbor trial directory."""

    task_id: str
    trial_name: str
    swebench_reward: float | None
    deterministic_comparison: str
    native_score: float | None
    nemoflow_score: float | None
    score_delta: float | None
    score_category: str
    native_error: str | None
    nemoflow_error: str | None
    native_path: str
    nemoflow_path: str

    def as_csv_row(self) -> dict[str, str]:
        """Return a CSV-friendly representation."""

        def _format_optional_number(value: float | None) -> str:
            return "" if value is None else f"{value:.6g}"

        return {
            "task_id": self.task_id,
            "trial_name": self.trial_name,
            "swebench_reward": _format_optional_number(self.swebench_reward),
            "deterministic_comparison": self.deterministic_comparison,
            "native_score": _format_optional_number(self.native_score),
            "nemoflow_score": _format_optional_number(self.nemoflow_score),
            "score_delta": _format_optional_number(self.score_delta),
            "score_category": self.score_category,
            "native_error": self.native_error or "",
            "nemoflow_error": self.nemoflow_error or "",
            "native_path": self.native_path,
            "nemoflow_path": self.nemoflow_path,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _message_to_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        parts = message.get("parts")
        if parts is None:
            text = message.get("text")
            return text if isinstance(text, str) else ""
    else:
        parts = message

    if not isinstance(parts, list):
        return ""

    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if part.get("type") == "text" and isinstance(text, str):
                text_parts.append(text)
    return "\n".join(text_parts)


def _first_user_message(*trajectories: dict[str, Any]) -> str:
    for trajectory in trajectories:
        for step in trajectory.get("steps", []):
            if step.get("source") == "user":
                text = _message_to_text(step.get("message")).strip()
                if text:
                    return text
    return ""


def _final_agent_message(trajectory: dict[str, Any]) -> str:
    for step in reversed(trajectory.get("steps", [])):
        if step.get("source") != "agent":
            continue
        text = _message_to_text(step.get("message")).strip()
        if text and text != "(tool use)":
            return text
    return ""


def _ensure_user_prompt(trajectory: dict[str, Any], prompt: str) -> dict[str, Any]:
    if not prompt:
        return trajectory
    if any(step.get("source") == "user" for step in trajectory.get("steps", [])):
        return trajectory

    patched = dict(trajectory)
    synthetic_user_step = {
        "step_id": 1,
        "source": "user",
        "message": prompt,
    }
    renumbered_steps: list[dict[str, Any]] = []
    for index, step in enumerate(trajectory.get("steps", []), start=2):
        renumbered_step = dict(step)
        renumbered_step["step_id"] = index
        renumbered_steps.append(renumbered_step)
    patched["steps"] = [synthetic_user_step, *renumbered_steps]
    return patched


def _write_eval_sample(
    *,
    path: Path,
    item_id: str,
    trajectory: dict[str, Any],
    prompt: str,
    output_obj: str,
    source_artifact: Path,
    lane: str,
) -> None:
    sample = {
        "item_id": f"{item_id}:{lane}",
        "trajectory": _ensure_user_prompt(trajectory, prompt),
        "output_obj": output_obj,
        "metadata": {
            "source_artifact": str(source_artifact),
            "lane": lane,
        },
    }
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")


def _trial_task_id(trial_dir: Path) -> str:
    return trial_dir.name.rsplit("__", 1)[0]


def _trial_dirs(path: Path) -> list[Path]:
    if (path / "agent").is_dir():
        return [path]
    return sorted(child for child in path.iterdir() if child.is_dir() and (child / "agent").is_dir())


def _trial_reward(trial_dir: Path) -> float | None:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = _load_json(result_path)
    except (OSError, json.JSONDecodeError):
        return None

    reward = (((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
    return float(reward) if isinstance(reward, int | float) else None


def _score_category(delta: float | None, threshold: float) -> str:
    if delta is None:
        return NOT_SCORED
    if abs(delta) <= threshold:
        return SCORE_SAME
    return NEMOFLOW_HIGHER if delta > 0 else NATIVE_HIGHER


def _score_artifact(
    *,
    artifact_path: Path,
    scoring_config: ScoringConfig,
) -> tuple[float | None, str | None]:
    if scoring_config.no_llm:
        return None, None

    def _timeout_handler(_signum, _frame) -> None:
        raise TimeoutError(f"Trajectory evaluator timed out after {scoring_config.score_timeout_sec} seconds.")

    previous_handler = None
    if scoring_config.score_timeout_sec is not None:
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(scoring_config.score_timeout_sec)
    try:
        reward, _details = evaluate_artifact_sync(
            artifact_path=artifact_path,
            evaluator_kind=scoring_config.evaluator_kind,
            evaluator_ref=scoring_config.evaluator_ref,
            config_file=scoring_config.config_file,
            evaluator_name=scoring_config.evaluator_name,
        )
    except Exception as exc:  # noqa: BLE001 - keep batch scoring resilient.
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if scoring_config.score_timeout_sec is not None:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
    return reward, None


def score_trial(
    trial_dir: Path,
    *,
    native_rel: Path,
    candidate_rel: Path,
    scoring_config: ScoringConfig,
) -> TrialScoreRow:
    native_path = trial_dir / native_rel
    candidate_path = trial_dir / candidate_rel
    task_id = _trial_task_id(trial_dir)
    reward = _trial_reward(trial_dir)

    deterministic_comparison = "missing"
    native_score: float | None = None
    nemoflow_score: float | None = None
    native_error: str | None = None
    nemoflow_error: str | None = None

    native_trajectory: dict[str, Any] | None = None
    candidate_trajectory: dict[str, Any] | None = None

    if not native_path.exists():
        native_error = "missing native trajectory"
    if not candidate_path.exists():
        nemoflow_error = "missing NeMo-Flow trajectory"

    if native_path.exists() and candidate_path.exists():
        try:
            comparison: ToolSequenceComparison = compare_atif_tool_sequences(native_path, candidate_path)
            deterministic_comparison = comparison.classification
        except Exception as exc:  # noqa: BLE001 - report and continue.
            deterministic_comparison = "comparison_error"
            native_error = native_error or f"{type(exc).__name__}: {exc}"

        try:
            native_trajectory = _load_json(native_path)
            candidate_trajectory = _load_json(candidate_path)
        except (OSError, json.JSONDecodeError) as exc:
            native_error = native_error or f"{type(exc).__name__}: {exc}"

    if native_trajectory is not None and candidate_trajectory is not None:
        prompt = _first_user_message(candidate_trajectory, native_trajectory)
        native_output = _final_agent_message(native_trajectory)
        candidate_output = _final_agent_message(candidate_trajectory)
        with tempfile.TemporaryDirectory(prefix="nat-harbor-atif-score-") as temp_dir:
            temp_path = Path(temp_dir)
            native_sample_path = temp_path / "native.json"
            candidate_sample_path = temp_path / "nemoflow.json"
            _write_eval_sample(
                path=native_sample_path,
                item_id=task_id,
                trajectory=native_trajectory,
                prompt=prompt,
                output_obj=native_output,
                source_artifact=native_path,
                lane="native",
            )
            _write_eval_sample(
                path=candidate_sample_path,
                item_id=task_id,
                trajectory=candidate_trajectory,
                prompt=prompt,
                output_obj=candidate_output,
                source_artifact=candidate_path,
                lane="nemoflow",
            )
            native_score, native_score_error = _score_artifact(
                artifact_path=native_sample_path,
                scoring_config=scoring_config,
            )
            nemoflow_score, nemoflow_score_error = _score_artifact(
                artifact_path=candidate_sample_path,
                scoring_config=scoring_config,
            )
            native_error = native_error or native_score_error
            nemoflow_error = nemoflow_error or nemoflow_score_error

    score_delta = (nemoflow_score - native_score) if native_score is not None and nemoflow_score is not None else None
    return TrialScoreRow(
        task_id=task_id,
        trial_name=trial_dir.name,
        swebench_reward=reward,
        deterministic_comparison=deterministic_comparison,
        native_score=native_score,
        nemoflow_score=nemoflow_score,
        score_delta=score_delta,
        score_category=_score_category(score_delta, scoring_config.score_threshold),
        native_error=native_error,
        nemoflow_error=nemoflow_error,
        native_path=str(native_path),
        nemoflow_path=str(candidate_path),
    )


def _write_csv(rows: list[TrialScoreRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].as_csv_row().keys()) if rows else list(TrialScoreRow.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_markdown(rows: list[TrialScoreRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_optional_number(value: float | None) -> str:
        return "" if value is None else f"{value:g}"

    lines = [
        "# ATIF Trajectory Scores",
        "",
        "| Task | Reward | Deterministic | Native | NeMo-Flow | Delta | Category | Errors |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        errors = "; ".join(error for error in [row.native_error, row.nemoflow_error] if error)
        lines.append(
            "| "
            f"{row.task_id} | "
            f"{_format_optional_number(row.swebench_reward)} | "
            f"{row.deterministic_comparison} | "
            f"{_format_optional_number(row.native_score)} | "
            f"{_format_optional_number(row.nemoflow_score)} | "
            f"{_format_optional_number(row.score_delta)} | "
            f"{row.score_category} | "
            f"{errors} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score native and ATOF-derived ATIF trajectories from Harbor trials.")
    parser.add_argument("--job-dir", required=True, type=Path, help="Harbor job dir or a single Harbor trial dir.")
    parser.add_argument("--output-dir", type=Path, help="Directory for report outputs. Defaults to --job-dir.")
    parser.add_argument(
        "--native-rel",
        type=Path,
        default=DEFAULT_NATIVE_REL,
        help="Native ATIF path relative to trial dir.",
    )
    parser.add_argument(
        "--candidate-rel",
        type=Path,
        default=DEFAULT_CANDIDATE_REL,
        help="ATOF-derived ATIF path relative to trial dir.",
    )
    parser.add_argument("--csv", type=Path, default=Path("atif-trajectory-scores.csv"), help="CSV report path.")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("atif-trajectory-scores.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of trials scored.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM scoring and emit deterministic comparison only.",
    )
    parser.add_argument(
        "--evaluator-kind",
        default="trajectory",
        help="Evaluator kind for Harbor ATIF evaluator bridge.",
    )
    parser.add_argument("--evaluator-ref", help="Custom evaluator ref in module:function format.")
    parser.add_argument("--config-file", help="NAT config path for builtin evaluator mode.")
    parser.add_argument("--evaluator-name", help="Evaluator name inside --config-file.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Absolute delta threshold for score_same classification.",
    )
    parser.add_argument(
        "--score-timeout-sec",
        type=int,
        default=120,
        help="Per-trajectory evaluator timeout in seconds. Use 0 to disable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.no_llm and not args.evaluator_ref and not args.config_file:
        raise SystemExit("Provide --config-file, --evaluator-ref, or --no-llm.")

    scoring_config = ScoringConfig(
        evaluator_kind=args.evaluator_kind,
        evaluator_ref=args.evaluator_ref,
        config_file=args.config_file,
        evaluator_name=args.evaluator_name,
        score_threshold=args.score_threshold,
        score_timeout_sec=args.score_timeout_sec or None,
        no_llm=args.no_llm,
    )
    trial_dirs = _trial_dirs(args.job_dir)
    if args.limit is not None:
        trial_dirs = trial_dirs[:args.limit]

    rows = [
        score_trial(
            trial_dir,
            native_rel=args.native_rel,
            candidate_rel=args.candidate_rel,
            scoring_config=scoring_config,
        ) for trial_dir in trial_dirs
    ]

    output_dir = args.output_dir or args.job_dir
    csv_path = args.csv if args.csv.is_absolute() else output_dir / args.csv
    markdown_path = args.markdown if args.markdown.is_absolute() else output_dir / args.markdown
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {len(rows)} rows to {markdown_path}")


if __name__ == "__main__":
    main()

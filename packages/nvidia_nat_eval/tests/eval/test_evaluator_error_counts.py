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
"""Evaluator failures are counted on `EvalOutput` and shown in the `nat eval` summary."""

from pathlib import Path
from types import SimpleNamespace

from nat.data_models.evaluate_runtime import EvaluationRunOutput
from nat.data_models.evaluate_runtime import ProfilerResults
from nat.data_models.evaluator import EvalInput
from nat.plugins.eval.cli.evaluate import write_tabular_output
from nat.plugins.eval.data_models.evaluator_io import EvalOutput
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem
from nat.plugins.eval.data_models.evaluator_io import count_evaluator_errors


def _output_with_one_failure() -> EvalOutput:
    """Three perfect answers and one judge failure, as in the reproduction for #2478."""
    items = [EvalOutputItem(id=i, score=1.0, reasoning={}) for i in range(3)]
    items.append(
        EvalOutputItem(id=3,
                       score=0.0,
                       reasoning={"error": "Evaluator error: judge timed out"},
                       error="Evaluator error: judge timed out"))
    return EvalOutput(average_score=0.75, eval_output_items=items)


def test_eval_output_counts_items_and_errors():
    output = _output_with_one_failure()

    assert output.num_items == 4
    assert output.num_errors == 1


def test_eval_output_without_failures_has_no_errors():
    output = EvalOutput(average_score=1.0, eval_output_items=[EvalOutputItem(id=1, score=1.0, reasoning={})])

    assert output.num_errors == 0
    assert EvalOutput(average_score=None, eval_output_items=[]).num_errors == 0


def test_counts_are_not_serialized():
    """The counts are derived, so the evaluator output files keep their existing shape."""
    dumped = _output_with_one_failure().model_dump()

    assert set(dumped) == {"average_score", "eval_output_items"}


def test_count_evaluator_errors_accepts_duck_typed_outputs():
    duck = SimpleNamespace(eval_output_items=[SimpleNamespace(error=None), SimpleNamespace(error="boom")])

    assert count_evaluator_errors(duck) == 1
    assert count_evaluator_errors(SimpleNamespace(average_score=0.5)) == 0


def _run_output(evaluation_results, tmp_path: Path) -> EvaluationRunOutput:
    return EvaluationRunOutput(
        workflow_output_file=tmp_path / "workflow_output.json",
        evaluator_output_files=[tmp_path / f"{name}_output.json" for name, _ in evaluation_results],
        workflow_interrupted=False,
        eval_input=EvalInput(eval_input_items=[]),
        evaluation_results=evaluation_results,
        profiler_results=ProfilerResults(),
    )


def test_summary_table_shows_errors_next_to_the_average(tmp_path: Path, capsys):
    clean = EvalOutput(average_score=1.0, eval_output_items=[EvalOutputItem(id=1, score=1.0, reasoning={})])
    run_output = _run_output([("accuracy", _output_with_one_failure()), ("relevance", clean)], tmp_path)

    write_tabular_output(run_output)
    printed = capsys.readouterr().out

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    header, _separator, accuracy, relevance = [line for line in printed.splitlines() if line.startswith("|")]
    assert cells(header) == ["Evaluator", "Avg Score", "Errors", "Output File"]
    assert cells(accuracy) == ["accuracy", "0.75", "1/4", "accuracy_output.json"]
    assert cells(relevance) == ["relevance", "1", "0/1", "relevance_output.json"]

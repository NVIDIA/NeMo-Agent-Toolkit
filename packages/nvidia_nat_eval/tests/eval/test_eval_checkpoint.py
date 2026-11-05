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
"""`nat eval` keeps each finished item on disk so an interrupted run can be resumed."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.config import Config
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.evaluate_config import EvalConfig
from nat.data_models.evaluate_config import EvalOutputConfig
from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.data_models.evaluate_runtime import ProfilerResults
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.token_usage import TokenUsageBaseModel
from nat.plugins.eval.runtime.checkpoint import CHECKPOINT_FILE_NAME
from nat.plugins.eval.runtime.checkpoint import EvalCheckpoint
from nat.plugins.eval.runtime.evaluate import EvaluationRun
from nat.runtime.session import SessionManager


def _llm_end_step(question: str, answer: str, total_tokens: int = 7) -> IntermediateStep:
    return IntermediateStep(parent_id="root",
                            function_ancestry=InvocationNode(function_name="llm", function_id="llm-1"),
                            payload=IntermediateStepPayload(
                                event_type=IntermediateStepType.LLM_END,
                                name="llm",
                                data=StreamEventData(input=question, output=answer),
                                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(
                                    prompt_tokens=total_tokens - 2, completion_tokens=2, total_tokens=total_tokens))))


def _item(item_id, question: str) -> EvalInputItem:
    return EvalInputItem(id=item_id,
                         input_obj=question,
                         expected_output_obj=f"expected {question}",
                         output_obj=None,
                         trajectory=[],
                         expected_trajectory=[],
                         full_dataset_entry={
                             "id": item_id, "question": question
                         })


def _answered(item_id, question: str) -> EvalInputItem:
    item = _item(item_id, question)
    item.output_obj = f"answer {question}"
    item.trajectory = [_llm_end_step(question, item.output_obj)]
    return item


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- the checkpoint file on its own ---------------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_output_and_trajectory(tmp_path):
    checkpoint = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()

    items = [_item(1, "q1"), _item(2, "q2")]
    resumed = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    restored = resumed.restore(items)

    assert restored == [items[0]]
    assert items[0].output_obj == "answer q1"
    assert len(items[0].trajectory) == 1
    step = items[0].trajectory[0]
    assert isinstance(step, IntermediateStep)
    assert step.event_type == IntermediateStepType.LLM_END
    assert step.usage_info.token_usage.total_tokens == 7
    assert items[1].output_obj is None


def test_checkpoint_skips_a_torn_last_line(tmp_path):
    """A process killed mid-write leaves half a line; the finished records before it still count."""
    path = tmp_path / CHECKPOINT_FILE_NAME
    checkpoint = EvalCheckpoint(path)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"id": 2, "input": "q2", "out')

    items = [_item(1, "q1"), _item(2, "q2")]
    restored = EvalCheckpoint(path).restore(items)

    assert restored == [items[0]]
    assert items[1].output_obj is None


def test_checkpoint_does_not_restore_an_item_whose_input_changed(tmp_path):
    checkpoint = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "old question"))
    checkpoint.close()

    items = [_item(1, "edited question")]
    restored = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME).restore(items)

    assert not restored
    assert items[0].output_obj is None


def test_checkpoint_matches_items_without_ids_by_input(tmp_path):
    """Datasets without an id column give every item the same empty id."""
    checkpoint = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    first = _answered("", "same question")
    first.output_obj = "first answer"
    second = _answered("", "same question")
    second.output_obj = "second answer"
    checkpoint.record(first)
    checkpoint.record(second)
    checkpoint.record(_answered("", "other question"))
    checkpoint.close()

    items = [_item("", "same question"), _item("", "other question"), _item("", "same question"), _item("", "new")]
    restored = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME).restore(items)

    assert restored == items[:3]
    assert [item.output_obj for item in items] == ["first answer", "answer other question", "second answer", None]


def test_resume_without_a_checkpoint_restores_nothing(tmp_path):
    items = [_item(1, "q1")]
    assert EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME).restore(items) == []


def test_fresh_start_clears_an_old_checkpoint(tmp_path):
    path = tmp_path / CHECKPOINT_FILE_NAME
    old = EvalCheckpoint(path)
    old.start(resume=False)
    old.record(_answered(1, "q1"))
    old.close()

    fresh = EvalCheckpoint(path)
    fresh.start(resume=False)
    fresh.close()

    assert _read_records(path) == []


def test_resumed_start_keeps_old_records_and_appends(tmp_path):
    path = tmp_path / CHECKPOINT_FILE_NAME
    old = EvalCheckpoint(path)
    old.start(resume=False)
    old.record(_answered(1, "q1"))
    old.close()

    resumed = EvalCheckpoint(path)
    resumed.start(resume=True)
    resumed.record(_answered(2, "q2"))
    resumed.close()

    assert [record["id"] for record in _read_records(path)] == [1, 2]


def test_checkpoint_skips_a_torn_line_that_splits_a_character(tmp_path):
    """A kill can land inside a multi-byte character; resume must still read the finished records."""
    path = tmp_path / CHECKPOINT_FILE_NAME
    checkpoint = EvalCheckpoint(path)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()
    with open(path, "ab") as f:
        f.write('{"id": 2, "input": "caf\u00e9'.encode()[:-1])

    items = [_item(1, "q1"), _item(2, "café")]
    restored = EvalCheckpoint(path).restore(items)

    assert restored == [items[0]]


def test_resuming_after_a_torn_line_keeps_the_next_record(tmp_path):
    path = tmp_path / CHECKPOINT_FILE_NAME
    checkpoint = EvalCheckpoint(path)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"id": 2, "input": "q2", "out')

    resumed = EvalCheckpoint(path)
    resumed.start(resume=True)
    resumed.record(_answered(2, "q2"))
    resumed.close()

    items = [_item(1, "q1"), _item(2, "q2")]
    assert EvalCheckpoint(path).restore(items) == items


def test_checkpoint_keeps_items_with_the_same_input_apart_by_id(tmp_path):
    checkpoint = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(2, "same question"))
    checkpoint.close()

    items = [_item(1, "same question"), _item(2, "same question")]
    restored = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME).restore(items)

    assert restored == [items[1]]
    assert items[0].output_obj is None


def test_checkpoint_uses_a_later_record_when_an_earlier_one_is_unreadable(tmp_path):
    path = tmp_path / CHECKPOINT_FILE_NAME
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": 1, "input": "q1", "output": "old", "trajectory": [{"not": "a step"}]}) + "\n")
        f.write(json.dumps({"id": 1, "input": "q1", "output": "answer q1", "trajectory": []}) + "\n")
        f.write(json.dumps({"id": 2, "input": "q2"}) + "\n")  # missing fields

    items = [_item(1, "q1"), _item(2, "q2")]
    restored = EvalCheckpoint(path).restore(items)

    assert restored == [items[0]]
    assert items[0].output_obj == "answer q1"


# --- local workflow runs --------------------------------------------------------------------------------------------


class _Workflow:
    has_single_output = True


def _session_manager(answer_for, on_call=None):
    """A SessionManager whose workflow answers `answer_for(question)`; raising inside it fails that item."""
    session_manager = MagicMock(spec=SessionManager)
    workflow = _Workflow()
    session_manager.workflow = workflow

    @asynccontextmanager
    async def run(question, runtime_type=None):
        runner = MagicMock()

        async def result():
            if on_call is not None:
                called = on_call(question)
                if asyncio.iscoroutine(called):
                    await called
            return answer_for(question)

        runner.result = AsyncMock(side_effect=result)
        runner.convert = MagicMock(side_effect=lambda value, to_type: value)
        yield runner

    @asynccontextmanager
    async def session(http_connection=None, user_id=None):
        s = MagicMock()
        s.run = run
        s.workflow = workflow
        yield s

    session_manager.session = session
    return session_manager


def _eval_config(output_dir: Path) -> EvalConfig:
    eval_config = EvalConfig()
    eval_config.general.dataset = EvalDatasetJsonConfig()
    eval_config.general.output = EvalOutputConfig(dir=output_dir)
    eval_config.general.max_concurrency = 1
    eval_config.evaluators = {}
    return eval_config


def _evaluation_run(tmp_path: Path, items: list[EvalInputItem], **run_config) -> EvaluationRun:
    config = EvaluationRunConfig(config_file=Path("config.yml"), **run_config)
    evaluation_run = EvaluationRun(config)
    evaluation_run.eval_config = _eval_config(tmp_path / "out")
    evaluation_run.eval_input = EvalInput(eval_input_items=items)
    evaluation_run.checkpoint = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    evaluation_run.checkpoint.start(resume=False)
    return evaluation_run


@pytest.fixture(name="no_intermediate_steps", autouse=True)
def fixture_no_intermediate_steps():
    with patch("nat.builder.runtime_event_subscriber.pull_intermediate", AsyncMock(return_value=[])):
        yield


async def test_each_item_is_on_disk_as_soon_as_it_finishes(tmp_path):
    """While q2 is still running, q1's answer is already in the checkpoint file."""
    seen_on_disk_during_q2: list[dict] = []

    async def on_call(question):
        if question == "q2":
            # items run concurrently, so give q1 a moment to finish first
            for _ in range(100):
                if _read_records(tmp_path / CHECKPOINT_FILE_NAME):
                    break
                await asyncio.sleep(0.01)
            seen_on_disk_during_q2.extend(_read_records(tmp_path / CHECKPOINT_FILE_NAME))

    evaluation_run = _evaluation_run(tmp_path, [_item(1, "q1"), _item(2, "q2")])
    await evaluation_run.run_workflow_local(_session_manager(lambda q: f"answer {q}", on_call=on_call))
    evaluation_run.checkpoint.close()

    assert [(r["id"], r["output"]) for r in seen_on_disk_during_q2] == [(1, "answer q1")]
    assert [(r["id"], r["output"]) for r in _read_records(tmp_path / CHECKPOINT_FILE_NAME)] == [(1, "answer q1"),
                                                                                                (2, "answer q2")]


async def test_an_item_is_checkpointed_before_the_callbacks_run(tmp_path):
    """Ctrl-C can land while an exporter callback awaits; the item's answer is already paid for."""
    evaluation_run = _evaluation_run(tmp_path, [_item(1, "q1")])

    async def cancelled_while_exporting(item, usage_stats_item):
        raise asyncio.CancelledError

    evaluation_run.callback_manager.a_on_usage_stats = cancelled_while_exporting
    with pytest.raises(asyncio.CancelledError):
        await evaluation_run.run_workflow_local(_session_manager(lambda q: f"answer {q}"))
    evaluation_run.checkpoint.close()

    assert [(r["id"], r["output"]) for r in _read_records(tmp_path / CHECKPOINT_FILE_NAME)] == [(1, "answer q1")]


async def test_failed_items_are_not_checkpointed(tmp_path):

    def answer_for(question):
        if question == "q2":
            raise RuntimeError("model endpoint down")
        return f"answer {question}"

    evaluation_run = _evaluation_run(tmp_path, [_item(1, "q1"), _item(2, "q2")])
    await evaluation_run.run_workflow_local(_session_manager(answer_for))
    evaluation_run.checkpoint.close()

    assert evaluation_run.workflow_interrupted
    assert [r["id"] for r in _read_records(tmp_path / CHECKPOINT_FILE_NAME)] == [1]


# --- run_and_evaluate with --resume ---------------------------------------------------------------------------------


def _write_dataset(path: Path, n: int) -> Path:
    path.write_text(json.dumps([{
        "id": i, "question": f"q{i}", "answer": f"expected q{i}"
    } for i in range(1, n + 1)]),
                    encoding="utf-8")
    return path


def _nat_config(dataset: Path, output_dir: Path, **output) -> Config:
    # built from a dict like a loaded YAML file, so output_dir follows output.dir
    config = Config()
    config.eval = EvalConfig.model_validate({
        "general": {
            "max_concurrency": 1,
            "dataset": {
                "_type": "json", "file_path": str(dataset)
            },
            "output": {
                "dir": str(output_dir), **output
            },
        },
        "evaluators": {},
    })
    return config


async def _run_and_evaluate(config: Config, session_manager, job_id: str | None = None, **run_config):
    evaluation_run = EvaluationRun(EvaluationRunConfig(config_file=config, **run_config))
    eval_builder = MagicMock()

    @asynccontextmanager
    async def from_config(config):
        yield eval_builder

    uploader = MagicMock()
    uploader.upload_directory = AsyncMock()
    with patch("nat.plugins.eval.runtime.builder.WorkflowEvalBuilder.from_config", side_effect=from_config), \
         patch("nat.plugins.eval.runtime.evaluate._get_output_uploader_cls",
               return_value=MagicMock(return_value=uploader)), \
         patch.object(evaluation_run, "profile_workflow", AsyncMock(return_value=ProfilerResults())):
        output = await evaluation_run.run_and_evaluate(session_manager=session_manager, job_id=job_id)
    return evaluation_run, output


async def test_resume_runs_only_the_unfinished_items(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 4)
    output_dir = tmp_path / "out"
    calls: list[str] = []
    finished_first_pass: list[int] = []

    def first_pass_answer(question):
        if question == "q3":
            # Ctrl-C under asyncio.run cancels the running tasks
            raise asyncio.CancelledError
        finished_first_pass.append(int(question[1:]))
        return f"answer {question}"

    with pytest.raises(asyncio.CancelledError):
        await _run_and_evaluate(_nat_config(dataset, output_dir),
                                _session_manager(first_pass_answer),
                                write_checkpoint=True)

    # items run concurrently, so every item that finished before the cancel is kept, and only those
    assert sorted(r["id"] for r in _read_records(output_dir / CHECKPOINT_FILE_NAME)) == sorted(finished_first_pass)
    assert 3 not in finished_first_pass

    evaluation_run, output = await _run_and_evaluate(_nat_config(dataset, output_dir),
                                                     _session_manager(lambda q: f"answer {q}", on_call=calls.append),
                                                     resume=True)

    assert sorted(calls) == sorted(f"q{i}" for i in range(1, 5) if i not in finished_first_pass)
    assert [item.output_obj for item in output.eval_input.eval_input_items] == [f"answer q{i}" for i in range(1, 5)]
    workflow_output = json.loads((output_dir / "workflow_output.json").read_text(encoding="utf-8"))
    assert [row["generated_answer"] for row in workflow_output] == [f"answer q{i}" for i in range(1, 5)]
    assert sorted(r["id"] for r in _read_records(output_dir / CHECKPOINT_FILE_NAME)) == [1, 2, 3, 4]


async def test_resume_restores_usage_stats_for_finished_items(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    checkpoint = EvalCheckpoint(output_dir / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()

    evaluation_run, _ = await _run_and_evaluate(_nat_config(dataset, output_dir),
                                                _session_manager(lambda q: pytest.fail("q1 already finished")),
                                                resume=True)

    assert evaluation_run.usage_stats.usage_stats_items[1].total_tokens == 7


async def test_a_fresh_run_ignores_and_replaces_an_old_checkpoint(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 2)
    output_dir = tmp_path / "out"
    calls: list[str] = []
    output_dir.mkdir()
    checkpoint = EvalCheckpoint(output_dir / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()

    # cleanup=False keeps the directory, so only the fresh start can clear the old records
    await _run_and_evaluate(_nat_config(dataset, output_dir, cleanup=False),
                            _session_manager(lambda q: f"new {q}", on_call=calls.append),
                            write_checkpoint=True)

    assert calls == ["q1", "q2"]
    assert [r["output"] for r in _read_records(output_dir / CHECKPOINT_FILE_NAME)] == ["new q1", "new q2"]


@pytest.mark.parametrize("write_checkpoint", [True, False])
async def test_interrupted_warning_points_to_resume_only_with_a_checkpoint(tmp_path, caplog, write_checkpoint):
    dataset = _write_dataset(tmp_path / "dataset.json", 2)

    def answer_for(question):
        if question == "q2":
            raise RuntimeError("model endpoint down")
        return f"answer {question}"

    with caplog.at_level("WARNING"):
        await _run_and_evaluate(_nat_config(dataset, tmp_path / "out"),
                                _session_manager(answer_for),
                                write_checkpoint=write_checkpoint)

    interrupted = [r.getMessage() for r in caplog.records if "interrupted" in r.getMessage()]
    assert interrupted
    if write_checkpoint:
        assert all("--resume" in message for message in interrupted)
    else:
        assert not any("--resume" in message for message in interrupted)
        assert all("--skip_completed_entries" in message for message in interrupted)


async def test_resume_with_skip_workflow_is_rejected(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    with pytest.raises(ValueError, match="resume"):
        await _run_and_evaluate(_nat_config(dataset, tmp_path / "out"),
                                _session_manager(lambda q: q),
                                resume=True,
                                skip_workflow=True)


async def test_resume_with_a_new_job_directory_per_run_is_rejected(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    config = _nat_config(dataset, tmp_path / "out")
    config.eval.general.output.job_management.append_job_id_to_output_dir = True
    with pytest.raises(ValueError, match="append_job_id_to_output_dir"):
        await _run_and_evaluate(config, _session_manager(lambda q: q), resume=True)


async def test_resume_with_an_explicit_job_id_is_allowed(tmp_path):
    """FastAPI passes a job_id, so the output directory is the same on every run."""
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    config = _nat_config(dataset, tmp_path / "out")
    config.eval.general.output.job_management.append_job_id_to_output_dir = True
    job_dir = tmp_path / "out" / "jobs" / "job_1"
    job_dir.mkdir(parents=True)
    checkpoint = EvalCheckpoint(job_dir / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()

    _, output = await _run_and_evaluate(config,
                                        _session_manager(lambda q: pytest.fail("q1 already finished")),
                                        job_id="job_1",
                                        resume=True)

    assert output.eval_input.eval_input_items[0].output_obj == "answer q1"


async def test_resume_without_writing_output_is_rejected(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    with pytest.raises(ValueError, match="write_output"):
        await _run_and_evaluate(_nat_config(dataset, tmp_path / "out"),
                                _session_manager(lambda q: q),
                                resume=True,
                                write_output=False)


async def test_no_checkpoint_unless_asked(tmp_path):
    """Programmatic runs (optimizer, profiler) share output directories, so they don't write one by default."""
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    output_dir = tmp_path / "out"
    await _run_and_evaluate(_nat_config(dataset, output_dir), _session_manager(lambda q: f"answer {q}"))
    assert not (output_dir / CHECKPOINT_FILE_NAME).exists()


async def test_checkpoint_is_closed_when_the_run_fails_before_the_workflow(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.json", 1)
    config = _nat_config(dataset, tmp_path / "out")
    config.eval.general.validate_llm_endpoints = True
    evaluation_run = EvaluationRun(EvaluationRunConfig(config_file=config, write_checkpoint=True))
    with patch("nat.plugins.eval.runtime.evaluate.validate_llm_endpoints", AsyncMock(side_effect=RuntimeError("404"))):
        with pytest.raises(RuntimeError, match="404"):
            await evaluation_run.run_and_evaluate(session_manager=_session_manager(lambda q: q))
    assert evaluation_run.checkpoint is None or evaluation_run.checkpoint._file is None


async def test_restored_items_reach_the_eval_callbacks(tmp_path):
    """Exporters such as Weave log every prediction of the run, so restored items are reported too."""
    dataset = _write_dataset(tmp_path / "dataset.json", 2)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    checkpoint = EvalCheckpoint(output_dir / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(_answered(1, "q1"))
    checkpoint.close()

    evaluation_run = EvaluationRun(EvaluationRunConfig(config_file=_nat_config(dataset, output_dir), resume=True))
    predictions: list[str] = []
    usage_items: list = []
    evaluation_run.callback_manager.on_prediction = lambda item, output: predictions.append(output)

    async def on_usage(item, usage_stats_item):
        usage_items.append(item.id)

    evaluation_run.callback_manager.a_on_usage_stats = on_usage
    evaluation_run.eval_input = EvalInput(eval_input_items=[_item(1, "q1"), _item(2, "q2")])
    evaluation_run.eval_config = _eval_config(output_dir)
    evaluation_run.checkpoint = EvalCheckpoint(output_dir / CHECKPOINT_FILE_NAME)
    await evaluation_run.restore_from_checkpoint()

    assert predictions == ["answer q1"]
    assert usage_items == [1]


async def test_total_runtime_counts_only_this_run_after_a_resume(tmp_path):
    """Restored items finished in an earlier run; the gap until the resume is not runtime."""
    dataset = _write_dataset(tmp_path / "dataset.json", 2)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    old = _answered(1, "q1")
    for step in old.trajectory:
        step.payload.event_timestamp = 1000.0  # finished long before this run
    checkpoint = EvalCheckpoint(output_dir / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(old)
    checkpoint.close()

    now_steps = [_llm_end_step("q2", "answer q2")]
    with patch("nat.builder.runtime_event_subscriber.pull_intermediate", AsyncMock(return_value=now_steps)):
        evaluation_run, _ = await _run_and_evaluate(_nat_config(dataset, output_dir),
                                                    _session_manager(lambda q: f"answer {q}"),
                                                    resume=True)

    assert evaluation_run.usage_stats.total_runtime < 60


async def test_total_runtime_after_a_resume_survives_a_pre_eval_function_that_copies_items(tmp_path):
    """A custom pre-eval process function may return new EvalInputItem objects instead of the restored ones."""
    from nat.plugins.eval.dataset_handler.dataset_handler import DatasetHandler

    dataset = _write_dataset(tmp_path / "dataset.json", 2)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    old = _answered(1, "q1")
    for step in old.trajectory:
        step.payload.event_timestamp = 1000.0
    checkpoint = EvalCheckpoint(output_dir / CHECKPOINT_FILE_NAME)
    checkpoint.start(resume=False)
    checkpoint.record(old)
    checkpoint.close()

    def copy_items(self, eval_input):
        return EvalInput(eval_input_items=[item.model_copy() for item in eval_input.eval_input_items])

    now_steps = [_llm_end_step("q2", "answer q2")]
    with patch("nat.builder.runtime_event_subscriber.pull_intermediate", AsyncMock(return_value=now_steps)), \
         patch.object(DatasetHandler, "pre_eval_process_eval_input", copy_items):
        evaluation_run, _ = await _run_and_evaluate(_nat_config(dataset, output_dir),
                                                    _session_manager(lambda q: f"answer {q}"),
                                                    resume=True)

    assert evaluation_run.usage_stats.total_runtime < 60


# --- remote workflow runs -------------------------------------------------------------------------------------------


async def test_remote_run_checkpoints_answered_items_only(tmp_path):
    items = [_item(1, "q1"), _item(2, "q2")]
    evaluation_run = _evaluation_run(tmp_path, items, endpoint="http://localhost:8000")

    async def fake_single(self, session, item):
        item.output_obj = None if item.input_obj == "q2" else f"answer {item.input_obj}"
        item.trajectory = []

    with patch("nat.plugins.eval.runtime.remote_workflow.EvaluationRemoteWorkflowHandler.run_workflow_remote_single",
               fake_single):
        await evaluation_run.run_workflow_remote()
    evaluation_run.checkpoint.close()

    assert [(r["id"], r["output"]) for r in _read_records(tmp_path / CHECKPOINT_FILE_NAME)] == [(1, "answer q1")]


async def test_remote_resume_sends_only_unfinished_items(tmp_path):
    items = [_item(1, "q1"), _item(2, "q2")]
    evaluation_run = _evaluation_run(tmp_path, items, endpoint="http://localhost:8000")
    evaluation_run.checkpoint.record(_answered(1, "q1"))
    evaluation_run.checkpoint.close()
    evaluation_run.checkpoint = EvalCheckpoint(tmp_path / CHECKPOINT_FILE_NAME)
    await evaluation_run.restore_from_checkpoint()
    sent: list[str] = []

    async def fake_single(self, session, item):
        sent.append(item.input_obj)
        item.output_obj = f"answer {item.input_obj}"

    with patch("nat.plugins.eval.runtime.remote_workflow.EvaluationRemoteWorkflowHandler.run_workflow_remote_single",
               fake_single):
        await evaluation_run.run_workflow_remote()

    assert sent == ["q2"]
    assert [item.output_obj for item in items] == ["answer q1", "answer q2"]

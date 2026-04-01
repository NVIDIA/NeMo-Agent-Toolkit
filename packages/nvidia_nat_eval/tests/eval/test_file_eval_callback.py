# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for FileEvalCallback."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nat.data_models.config import Config
from nat.data_models.evaluate_config import EvalConfig
from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem
from nat.eval.eval_callbacks import EvalResult
from nat.eval.eval_callbacks import EvalResultItem
from nat.plugins.eval.exporters.file_eval_callback import FileEvalCallback


@pytest.fixture(name="eval_result_item")
def fixture_eval_result_item():
    """Fixture for a single eval result item."""
    return EvalResultItem(
        item_id=1,
        input_obj="User input",
        expected_output="Golden answer",
        actual_output="Generated answer",
        scores={"MockEvaluator": 0.9},
        reasoning={"MockEvaluator": "All is well"},
    )


@pytest.fixture(name="eval_output")
def fixture_eval_output():
    """Fixture for an EvalOutput instance."""
    return EvalOutput(average_score=0.9, eval_output_items=[EvalOutputItem(id=1, score=0.9, reasoning="All is well")])


@pytest.fixture(name="run_config")
def fixture_run_config():
    """Fixture for EvaluationRunConfig."""
    return EvaluationRunConfig(config_file=Path("config.yml"), dataset="dummy_dataset")


@pytest.fixture(name="effective_config")
def fixture_effective_config():
    """Fixture for the effective config."""
    config = Config()
    config.eval = EvalConfig()
    return config


@pytest.fixture(name="eval_result")
def fixture_eval_result(eval_result_item, eval_output, run_config, effective_config, tmp_path):
    """Fixture for a fully populated EvalResult."""
    return EvalResult(
        metric_scores={"MockEvaluator": 0.9},
        items=[eval_result_item],
        evaluation_outputs=[("MockEvaluator", eval_output)],
        workflow_output_json='[{"id": 1, "output": "Generated answer"}]',
        run_config=run_config,
        effective_config=effective_config,
        output_dir=tmp_path / "output",
    )


def test_file_eval_callback_writes_workflow_output(eval_result, tmp_path):
    """Test that FileEvalCallback writes workflow_output.json."""
    callback = FileEvalCallback()
    callback.on_eval_complete(eval_result)

    output_file = tmp_path / "output" / "workflow_output.json"
    assert output_file.exists()
    assert output_file.read_text() == eval_result.workflow_output_json
    assert callback.workflow_output_file == output_file


def test_file_eval_callback_writes_atif_workflow_output(eval_result, tmp_path):
    """Test that FileEvalCallback writes workflow_output_atif.json when provided."""
    eval_result.atif_workflow_output_json = '[{"item_id": 1, "trajectory": {"steps": []}}]'

    callback = FileEvalCallback()
    callback.on_eval_complete(eval_result)

    output_file = tmp_path / "output" / "workflow_output_atif.json"
    assert output_file.exists()
    assert output_file.read_text() == eval_result.atif_workflow_output_json
    assert callback.atif_workflow_output_file == output_file


def test_file_eval_callback_writes_evaluator_outputs(eval_result, tmp_path):
    """Test that FileEvalCallback writes per-evaluator output files."""
    callback = FileEvalCallback()
    callback.on_eval_complete(eval_result)

    evaluator_file = tmp_path / "output" / "MockEvaluator_output.json"
    assert evaluator_file.exists()

    content = json.loads(evaluator_file.read_text())
    assert content["average_score"] == 0.9
    assert len(callback.evaluator_output_files) == 1
    assert callback.evaluator_output_files[0] == evaluator_file


@pytest.mark.filterwarnings("ignore:.*Pydantic serializer warnings.*:UserWarning")
def test_file_eval_callback_writes_config_from_path(eval_result, tmp_path):
    """Test that FileEvalCallback copies original config when config_file is a Path."""
    config_file = tmp_path / "test_config.yml"
    config_file.write_text("workflow:\n  type: test\n")
    eval_result.run_config.config_file = config_file
    eval_result.run_config.override = (("eval.general.max_concurrency", "5"), )

    callback = FileEvalCallback()
    callback.on_eval_complete(eval_result)

    output_dir = tmp_path / "output"
    config_original = output_dir / "config_original.yml"
    config_effective = output_dir / "config_effective.yml"
    config_metadata = output_dir / "config_metadata.json"

    assert config_original.exists()
    assert config_effective.exists()
    assert config_metadata.exists()

    metadata = json.loads(config_metadata.read_text())
    assert metadata["config_file"] == str(config_file)
    assert metadata["config_file_type"] == "Path"
    assert len(metadata["overrides"]) == 1
    assert metadata["overrides"][0]["path"] == "eval.general.max_concurrency"


@pytest.mark.filterwarnings("ignore:.*Pydantic serializer warnings.*:UserWarning")
def test_file_eval_callback_writes_config_from_basemodel(eval_result, tmp_path):
    """Test that FileEvalCallback serializes config when config_file is a BaseModel."""
    eval_result.run_config.config_file = Config()
    eval_result.run_config.override = ()

    callback = FileEvalCallback()
    callback.on_eval_complete(eval_result)

    output_dir = tmp_path / "output"
    assert (output_dir / "config_original.yml").exists()
    assert (output_dir / "config_effective.yml").exists()

    metadata = json.loads((output_dir / "config_metadata.json").read_text())
    assert metadata["config_file_type"] == "BaseModel"
    assert len(metadata["overrides"]) == 0


def test_file_eval_callback_handles_missing_effective_config(eval_result, tmp_path):
    """Test that FileEvalCallback handles None effective_config gracefully."""
    config_file = tmp_path / "test_config.yml"
    config_file.write_text("workflow:\n  type: test\n")
    eval_result.run_config.config_file = config_file
    eval_result.effective_config = None

    callback = FileEvalCallback()
    with patch("nat.plugins.eval.exporters.file_eval_callback.logger.warning") as mock_warning:
        callback.on_eval_complete(eval_result)

    mock_warning.assert_any_call("Effective config not available, skipping config_effective.yml")

    output_dir = tmp_path / "output"
    assert (output_dir / "config_original.yml").exists()
    assert not (output_dir / "config_effective.yml").exists()
    assert (output_dir / "config_metadata.json").exists()


def test_file_eval_callback_skips_when_no_output_dir(eval_result_item, eval_output, run_config, effective_config):
    """Test that FileEvalCallback does nothing when output_dir is None."""
    result = EvalResult(
        metric_scores={"MockEvaluator": 0.9},
        items=[eval_result_item],
        evaluation_outputs=[("MockEvaluator", eval_output)],
        workflow_output_json='[]',
        run_config=run_config,
        effective_config=effective_config,
        output_dir=None,
    )
    callback = FileEvalCallback()
    callback.on_eval_complete(result)

    assert callback.workflow_output_file is None
    assert callback.evaluator_output_files == []


def test_file_eval_callback_skips_workflow_output_when_none(eval_result, tmp_path):
    """Test that FileEvalCallback skips workflow_output.json when workflow_output_json is None."""
    eval_result.workflow_output_json = None

    callback = FileEvalCallback()
    callback.on_eval_complete(eval_result)

    assert not (tmp_path / "output" / "workflow_output.json").exists()
    assert callback.workflow_output_file is None


def test_file_eval_callback_handles_none_output_config(eval_result_item, eval_output, tmp_path):
    """Test FileEvalCallback when run_config is None (no config to write)."""
    result = EvalResult(
        metric_scores={"MockEvaluator": 0.9},
        items=[eval_result_item],
        evaluation_outputs=[("MockEvaluator", eval_output)],
        workflow_output_json='[]',
        run_config=None,
        effective_config=None,
        output_dir=tmp_path / "output",
    )
    callback = FileEvalCallback()
    callback.on_eval_complete(result)

    output_dir = tmp_path / "output"
    assert (output_dir / "workflow_output.json").exists()
    assert not (output_dir / "config_original.yml").exists()
    assert not (output_dir / "config_metadata.json").exists()


def test_on_dataset_loaded_is_noop():
    """Test that on_dataset_loaded does not fail."""
    callback = FileEvalCallback()
    callback.on_dataset_loaded(dataset_name="test", items=[])

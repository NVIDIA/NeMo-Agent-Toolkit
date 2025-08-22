# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nat.data_models.dataset_handler import EvalDatasetCustomConfig
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.dataset_handler import EvalDatasetStructureConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode
from nat.eval.dataset_handler.dataset_handler import DatasetHandler
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem


@pytest.fixture
def dataset_structure():
    """Fixture for dataset structure configuration"""
    return EvalDatasetStructureConfig(question_key="question",
                                      answer_key="answer",
                                      generated_answer_key="generated",
                                      trajectory_key="trajectory",
                                      expected_trajectory_key="expected_trajectory")


@pytest.fixture
def dataset_id_key():
    """Fixture for dataset id key."""
    return "id"


@pytest.fixture
def dataset_handler(dataset_config):
    """
    While setting this up we intentionally use default key names. They are compared with keys dataset_structure.
    This ensures that the defaults are not changed (easily or accidentally).
    """
    return DatasetHandler(dataset_config, reps=1, concurrency=1)


@pytest.fixture
def input_entry_one(dataset_id_key, dataset_structure):
    """Mock input entry."""
    return {
        dataset_id_key: "1",
        dataset_structure.question_key: "What is AI?",
        dataset_structure.answer_key: "Artificial Intelligence",
        dataset_structure.generated_answer_key: "AI",
        dataset_structure.trajectory_key: [],
        dataset_structure.expected_trajectory_key: []
    }


@pytest.fixture
def input_entry_two(dataset_id_key, dataset_structure):
    """Mock input entry."""
    return {
        dataset_id_key: "2",
        dataset_structure.question_key: "What is ML?",
        dataset_structure.answer_key: "Machine Learning",
        dataset_structure.generated_answer_key: "AI subset",
        dataset_structure.trajectory_key: [],
        dataset_structure.expected_trajectory_key: []
    }


@pytest.fixture
def input_entry_with_extras(dataset_id_key, dataset_structure):
    """Mock input entry with additional fields."""
    return {
        dataset_id_key: "3",
        dataset_structure.question_key: "What is NLP?",
        dataset_structure.answer_key: "Natural Language Processing",
        dataset_structure.generated_answer_key: "NLP",
        dataset_structure.trajectory_key: [],
        dataset_structure.expected_trajectory_key: [],
        "additional_field": "additional_value",
        "additional_field_2": 123,
        "additional_field_3": True,
        "additional_field_4": [1, 2, 3],
        "additional_field_5": {
            "key": "value"
        }
    }


@pytest.fixture
def mock_input_df_with_extras(input_entry_with_extras):
    """Mock DataFrame with additional fields."""
    return pd.DataFrame([input_entry_with_extras])


@pytest.fixture
def mock_input_df(input_entry_one, input_entry_two):
    """Mock DataFrame with sample dataset."""
    return pd.DataFrame([input_entry_one, input_entry_two])


@pytest.fixture
def dataset_config():
    """Fixture for dataset configuration."""
    return EvalDatasetJsonConfig()


@pytest.fixture
def dataset_swe_bench_id_key():
    """
    Fixture for swe dataset id key. swe_bench uses 'unstructured' data i.e.
    the nat-lib doesn't look beyond the id.
    """
    return "instance_id"


@pytest.fixture
def dataset_swe_bench_config(dataset_swe_bench_id_key):
    """Fixture for unstructured dataset configuration."""
    return EvalDatasetJsonConfig(id_key=dataset_swe_bench_id_key, structure=EvalDatasetStructureConfig(disable=True))


@pytest.fixture
def dataset_swe_bench_handler(dataset_swe_bench_config):
    return DatasetHandler(dataset_swe_bench_config, reps=1, concurrency=1)


@pytest.fixture
def mock_swe_bench_input_df(dataset_swe_bench_id_key):
    """Mock DataFrame with unstructured data."""
    return pd.DataFrame([{
        dataset_swe_bench_id_key: "foo_1", "problem": "Divide by zero", "repo": "foo"
    }, {
        dataset_swe_bench_id_key: "bar_2", "problem": "Overflow", "repo": "bar"
    }])


@pytest.fixture
def sample_nested_data():
    """Fixture providing sample nested JSON data for testing."""
    return {
        "metadata": {
            "dataset_name": "simple_calculator_test",
            "version": "1.0",
            "description": "Test dataset for calculator operations"
        },
        "configuration": {
            "format": "nested", "encoding": "utf-8"
        },
        "questions": [{
            "id": 1,
            "question": "What is 2 + 3?",
            "answer": "5",
            "category": "addition",
            "difficulty": "easy",
            "tags": ["basic", "arithmetic"]
        },
                      {
                          "id": 2,
                          "question": "What is 12 * 7?",
                          "answer": "84",
                          "category": "multiplication",
                          "difficulty": "medium",
                          "tags": ["multiplication", "arithmetic"]
                      },
                      {
                          "id": 3,
                          "question": "What is 144 / 12?",
                          "answer": "12",
                          "category": "division",
                          "difficulty": "medium",
                          "tags": ["division", "arithmetic"]
                      },
                      {
                          "id": 4,
                          "question": "What is 15 - 8?",
                          "answer": "7",
                          "category": "subtraction",
                          "difficulty": "easy",
                          "tags": ["subtraction", "arithmetic"]
                      },
                      {
                          "id": 5,
                          "question": "What is 25 * 25?",
                          "answer": "625",
                          "category": "multiplication",
                          "difficulty": "hard",
                          "tags": ["multiplication", "square"]
                      }]
    }


@pytest.fixture
def temp_nested_json_file(sample_nested_data):
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_nested_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


def sample_custom_parser(file_path: Path, difficulty: str = "") -> EvalInput:
    """
    Test implementation of a custom dataset parser that:
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract questions array from the nested structure
    questions = data.get('questions', [])

    # Apply filtering if specified
    if difficulty:
        filtered_questions = []
        for question in questions:
            # Check if filter_by_tag matches category or difficulty (case insensitive)
            if question.get('difficulty', '').lower() == difficulty.lower():
                filtered_questions.append(question)
        questions = filtered_questions

    eval_items = []

    for item in questions:
        eval_item = EvalInputItem(id=item['id'],
                                  input_obj=item['question'],
                                  expected_output_obj=item['answer'],
                                  full_dataset_entry=item)
        eval_items.append(eval_item)

    return EvalInput(eval_input_items=eval_items)


@pytest.fixture
def custom_dataset_config():
    """Fixture for dataset configuration."""
    function_str = f"{__name__}.sample_custom_parser"
    return EvalDatasetCustomConfig(function=function_str, kwargs={"difficulty": "medium"})


def test_get_eval_input_from_df_with_additional_fields(mock_input_df_with_extras,
                                                       input_entry_with_extras,
                                                       dataset_id_key,
                                                       dataset_structure):
    """
    Test that additional fields are always passed to the evaluator as full_dataset_entry.
    """
    dataset_config = EvalDatasetJsonConfig()
    dataset_handler = DatasetHandler(dataset_config, reps=1, concurrency=1)
    eval_input = dataset_handler.get_eval_input_from_df(mock_input_df_with_extras)

    # check core fields
    assert eval_input.eval_input_items[0].id == input_entry_with_extras[dataset_id_key]
    assert eval_input.eval_input_items[0].input_obj == input_entry_with_extras[dataset_structure.question_key]
    assert eval_input.eval_input_items[0].expected_output_obj == input_entry_with_extras[dataset_structure.answer_key]
    assert eval_input.eval_input_items[0].expected_trajectory == input_entry_with_extras[
        dataset_structure.expected_trajectory_key]

    # full_dataset_entry should always be provided
    assert eval_input.eval_input_items[0].full_dataset_entry == input_entry_with_extras


def test_get_eval_input_from_df(dataset_handler,
                                mock_input_df,
                                input_entry_one,
                                input_entry_two,
                                dataset_structure,
                                dataset_id_key):
    """
    Test DataFrame conversion to EvalInput for structured data.
    1. Ensure that default key names have not changed
    2. All rows are converted to EvalInputItems
    3. Each EvalInputItem has the correct values
    """
    eval_input = dataset_handler.get_eval_input_from_df(mock_input_df)

    assert isinstance(eval_input, EvalInput), "Should return an EvalInput instance"
    assert len(eval_input.eval_input_items) == len(mock_input_df), "Number of items should match DataFrame rows"

    def assert_input_item_valid(item, input_entry):
        assert item.id == input_entry[dataset_id_key], f"Expected id '{input_entry['id']}', got '{item.id}'"
        assert item.input_obj == input_entry[dataset_structure.question_key], \
            f"Expected input '{input_entry[dataset_structure.question_key]}', got '{item.input_obj}'"
        assert item.expected_output_obj == input_entry[dataset_structure.answer_key], \
            f"Expected answer '{input_entry[dataset_structure.answer_key]}', got '{item.expected_output_obj}'"

    first_item = eval_input.eval_input_items[0]
    second_item = eval_input.eval_input_items[1]
    assert_input_item_valid(first_item, input_entry_one)
    assert_input_item_valid(second_item, input_entry_two)


def test_get_eval_input_from_swe_bench_df(dataset_swe_bench_handler, mock_swe_bench_input_df):
    """
    Test DataFrame conversion to EvalInput for unstructured data.
    1. Ensure that entire row is passed as input_obj
    """
    eval_input = dataset_swe_bench_handler.get_eval_input_from_df(mock_swe_bench_input_df)

    assert isinstance(eval_input, EvalInput), "Should return an EvalInput instance"
    assert len(eval_input.eval_input_items) == len(mock_swe_bench_input_df), "Number of items must match DataFrame rows"

    first_item = eval_input.eval_input_items[0]
    second_item = eval_input.eval_input_items[1]
    assert first_item.input_obj == mock_swe_bench_input_df.iloc[0].to_json(), \
        f"Expected input '{mock_swe_bench_input_df.iloc[0].to_json()}', got '{first_item.input_obj}'"
    assert second_item.input_obj == mock_swe_bench_input_df.iloc[1].to_json(), \
        f"Expected input '{mock_swe_bench_input_df.iloc[1].to_json()}', got '{second_item.input_obj}'"


def test_get_eval_input_from_df_ignore_invalid_rows(dataset_handler, mock_input_df):
    """
    Test that
    1. Unknown columns are ignored.
    2. Rows missing `question_key` or having empty `question_key` (for structured data) are filtered out.
    This test is only applicable for structured data. For unstructured data there is no validation.
    """

    # Append bad rows to mock_input_df
    new_valid_row_id = "5"
    bad_rows = pd.DataFrame([
        {
            "id": "3",  # This row is missing "question" (row should be ignored)
            "answer": "Deep Learning",
            "generated": "DL",
            "trajectory": [],
            "expected_trajectory": []
        },
        {
            "id": "4",
            "question": "",  # Empty question (row should be ignored)
            "answer": "Machine Learning",
            "generated": "AI subset",
            "trajectory": [],
            "expected_trajectory": []
        },
        {
            "id": f"{new_valid_row_id}",
            "question": "What is NLP?",
            "answer": "Natural Language Processing",
            "generated": "NLP",
            "trajectory": [],
            "expected_trajectory": [],
            "extra_info": "This should be ignored"  # Extra column (row should be processed)
        },
        {
            "id": "6",
            "question": "  ",  # Empty question (row should be ignored)
            "answer": "Machine Learning",
            "generated": "AI subset",
            "trajectory": [],
            "expected_trajectory": []
        },
    ])
    test_df = pd.concat([mock_input_df, bad_rows], ignore_index=True)

    # Run function
    eval_input = dataset_handler.get_eval_input_from_df(test_df)

    assert isinstance(eval_input, EvalInput), "Should return an EvalInput instance"

    #  Check that invalid rows (missing or empty questions) are filtered out
    assert len(eval_input.eval_input_items) == len(mock_input_df) + 1, \
        f"Expected {len(mock_input_df) + 1} valid rows, but got {len(eval_input.eval_input_items)}"

    valid_ids = {item.id for item in eval_input.eval_input_items}
    expected_ids = {row["id"] for _, row in mock_input_df.iterrows()} | {new_valid_row_id}  # Include new valid row

    assert valid_ids == expected_ids, f"Expected valid IDs {expected_ids}, but got {valid_ids}"


def test_setup_reps(dataset_handler, mock_input_df, dataset_id_key):
    """Test that dataset repetitions are correctly applied."""
    replicated_df = dataset_handler.setup_reps(mock_input_df)

    assert len(replicated_df) == len(mock_input_df) * dataset_handler.reps, "Dataset should be replicated correctly"
    assert all("_rep" in str(i) for i in replicated_df[dataset_id_key]), "IDs should be suffixed with `_repX`"


@pytest.fixture
def mock_intermediate_steps():
    """Create a list of mock intermediate steps with different event types."""
    steps = []
    # Add LLM_START step
    steps.append(
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llm_start", function_id="test-llm-start"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START, name="llm_start")))
    # Add LLM_END step
    steps.append(
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llm_end", function_id="test-llm-end"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_END, name="llm_end")))
    # Add TOOL_START step
    steps.append(
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="tool_start", function_id="test-tool-start"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                         name="tool_start")))
    # Add TOOL_END step
    steps.append(
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="tool_end", function_id="test-tool-end"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_END, name="tool_end")))
    return steps


def test_filter_intermediate_steps(dataset_handler, mock_intermediate_steps):
    """Test that filter_intermediate_steps correctly filters steps based on event types."""
    # Define the filter to include only LLM_END, TOOL_START, and TOOL_END
    event_filter = [IntermediateStepType.LLM_END, IntermediateStepType.TOOL_START, IntermediateStepType.TOOL_END]

    # Get the filtered steps
    filtered_steps = dataset_handler.filter_intermediate_steps(mock_intermediate_steps, event_filter)

    # Verify that only the specified event types are included (LLM_START is filtered out)
    event_types = [step["payload"]["event_type"] for step in filtered_steps]
    assert IntermediateStepType.LLM_START not in event_types, "LLM_START should be filtered out"
    assert IntermediateStepType.LLM_END in event_types, "LLM_END should be included"
    assert IntermediateStepType.TOOL_START in event_types, "TOOL_START should be included"
    assert IntermediateStepType.TOOL_END in event_types, "TOOL_END should be included"

    # Verify the order of steps is preserved
    assert len(filtered_steps) == 3, "Should have exactly 3 steps after filtering"
    assert filtered_steps[0]["payload"]["event_type"] == IntermediateStepType.LLM_END, "First step should be LLM_END"
    assert filtered_steps[1]["payload"]["event_type"] == IntermediateStepType.TOOL_START, \
        "Second step should be TOOL_START"
    assert filtered_steps[2]["payload"]["event_type"] == IntermediateStepType.TOOL_END, "Third step should be TOOL_END"


def make_eval_input_item(**overrides):
    defaults = {
        "id": "default_id",
        "input_obj": None,
        "expected_output_obj": None,
        "output_obj": None,
        "trajectory": [],
        "expected_trajectory": [],
        "full_dataset_entry": {},
    }
    defaults.update(overrides)
    return EvalInputItem(**defaults)


def test_publish_eval_input_unstructured_string_and_json():
    """Test that unstructured input handles plain strings, JSON strings, and Python objects correctly."""

    config = EvalDatasetJsonConfig(id_key="id", structure=EvalDatasetStructureConfig(disable=True))
    handler = DatasetHandler(config, reps=1, concurrency=1)

    items = [
        make_eval_input_item(id="1", output_obj="plain string output"),
        make_eval_input_item(id="2", output_obj='{"result": 42, "ok": true}'),
        make_eval_input_item(id="3", output_obj=EvalOutputItem(id="3", score=42, reasoning="The answer is 42")),
        make_eval_input_item(id="4", output_obj=42),
    ]
    eval_input = EvalInput(eval_input_items=items)
    output_json = handler.publish_eval_input(eval_input)
    output = json.loads(output_json)

    assert isinstance(output, list)
    assert output[0] == "plain string output"
    assert isinstance(output[1], dict)
    assert output[1] == {"result": 42, "ok": True}
    assert isinstance(output[2], dict)
    assert output[2] == {"id": "3", "score": 42, "reasoning": "The answer is 42"}
    assert output[3] == 42


def test_custom_dataset_config(custom_dataset_config, temp_nested_json_file):
    dataset_handler = DatasetHandler(custom_dataset_config, reps=1, concurrency=1)

    eval_input = dataset_handler.get_eval_input_from_dataset(temp_nested_json_file)

    # check that there are two medium entries in the eval_input
    assert len(eval_input.eval_input_items) == 2
    assert all(item.full_dataset_entry['difficulty'] == 'medium' for item in eval_input.eval_input_items)

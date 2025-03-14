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

import pandas as pd
import pytest

from aiq.data_models.dataset_handler import EvalDatasetJsonConfig
from aiq.data_models.dataset_handler import EvalDatasetStructureConfig
from aiq.eval.dataset_handler.dataset_handler import DatasetHandler
from aiq.eval.evaluator.evaluator_model import EvalInput

# pylint: disable=redefined-outer-name


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
    return DatasetHandler(dataset_config, reps=1)


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
    the aiq-lib doesn't look beyond the id.
    """
    return "instance_id"


@pytest.fixture
def dataset_swe_bench_config(dataset_swe_bench_id_key):
    """Fixture for unstructured dataset configuration."""
    return EvalDatasetJsonConfig(id_key=dataset_swe_bench_id_key, structure=EvalDatasetStructureConfig(disable=True))


@pytest.fixture
def dataset_swe_bench_handler(dataset_swe_bench_config):
    return DatasetHandler(dataset_swe_bench_config, reps=1)


@pytest.fixture
def mock_swe_bench_input_df(dataset_swe_bench_id_key):
    """Mock DataFrame with unstructured data."""
    return pd.DataFrame([{
        dataset_swe_bench_id_key: "foo_1", "problem": "Divide by zero", "repo": "foo"
    }, {
        dataset_swe_bench_id_key: "bar_2", "problem": "Overflow", "repo": "bar"
    }])


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


def test_get_eval_input_from_df_ignore_invalid_rows(dataset_handler, mock_input_df, dataset_id_key):
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

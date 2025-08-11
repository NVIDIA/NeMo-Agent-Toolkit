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
from pathlib import Path

from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem


def extract_nested_questions(input_path: Path, filter_by_tag: str = None, max_rows: int = None, **kwargs) -> EvalInput:
    """
    Extract questions from a nested JSON structure with optional filtering.

    Expects JSON format:
    {
        "metadata": {...},
        "configuration": {...},
        "questions": [
            {"id": 1, "question": "...", "answer": "...", "category": "...", "difficulty": "...", ...},
            ...
        ]
    }

    Args:
        input_path: Path to the nested JSON file
        filter_by_tag: Optional tag to filter questions by (matches against category or difficulty)
        max_rows: Optional maximum number of questions to return
        **kwargs: Additional parameters (unused in this example)

    Returns:
        EvalInput object containing the extracted questions
    """

    # Load the nested JSON
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Extract questions array from the nested structure
    questions = data.get('questions', [])
    metadata = data.get('metadata', {})
    configuration = data.get('configuration', {})

    # Apply filtering if specified
    if filter_by_tag:
        filtered_questions = []
        for question in questions:
            # Check if filter_by_tag matches category, difficulty, or any other field
            if (question.get('category', '').lower() == filter_by_tag.lower()
                    or question.get('difficulty', '').lower() == filter_by_tag.lower()
                    or filter_by_tag.lower() in str(question).lower()):
                filtered_questions.append(question)
        questions = filtered_questions

    # Apply max_rows limit if specified
    if max_rows and max_rows > 0:
        questions = questions[:max_rows]

    eval_items = []

    for item in questions:
        # Create EvalInputItem with additional metadata in full_dataset_entry
        full_entry = {
            **item,  # Include original question data
            'dataset_metadata': metadata,
            'dataset_configuration': configuration,
            'processing_info': {
                'filtered_by_tag': filter_by_tag,
                'max_rows_applied': max_rows,
                'total_questions_in_dataset': len(data.get('questions', []))
            }
        }

        eval_item = EvalInputItem(
            id=item['id'],
            input_obj=item['question'],
            expected_output_obj=item['answer'],
            output_obj="",  # Will be filled by workflow
            expected_trajectory=[],
            trajectory=[],  # Will be filled by workflow
            full_dataset_entry=full_entry)
        eval_items.append(eval_item)

    return EvalInput(eval_input_items=eval_items)

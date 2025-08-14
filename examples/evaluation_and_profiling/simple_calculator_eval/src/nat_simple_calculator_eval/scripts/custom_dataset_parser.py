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

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem


def extract_nested_questions(file_path: Path, difficulty: str | None = None, max_rows: int | None = None) -> EvalInput:
    """
    This is a sample custom dataset parser that:
    1. Loads a nested JSON file
    2. Extracts the questions array from the nested structure
    3. Applies optional filtering by difficulty (hard, medium, easy)
    4. Applies an optional maximum number of questions to return
    5. Creates an EvalInput object with the extracted questions and returns it

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
        file_path: Path to the nested JSON file
        difficulty: Optional difficulty to filter questions by
        max_rows: Optional maximum number of questions to return

    Returns:
        EvalInput object containing the extracted questions
    """

    # Load the nested JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract questions array from the nested structure
    questions = data.get('questions', [])

    # Apply filtering if specified
    if difficulty:
        filtered_questions = []
        for question in questions:
            # Check if difficulty matches difficulty (hard, medium, easy)
            if question.get('difficulty', '').lower() == difficulty.lower():
                filtered_questions.append(question)
        questions = filtered_questions

    # Apply max_rows limit if specified
    if max_rows and max_rows > 0:
        questions = questions[:max_rows]

    eval_items = []

    for item in questions:
        eval_item = EvalInputItem(id=item['id'],
                                  input_obj=item['question'],
                                  expected_output_obj=item['answer'],
                                  full_dataset_entry=item)
        eval_items.append(eval_item)

    return EvalInput(eval_input_items=eval_items)

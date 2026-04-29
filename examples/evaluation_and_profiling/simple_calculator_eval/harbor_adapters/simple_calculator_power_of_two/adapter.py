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
"""Adapter for converting power-of-two examples to Harbor tasks."""

from __future__ import annotations

import json
from pathlib import Path

from common import HarborTaskAdapter
from common import TaskData
from common import parse_power_of_two


class SimpleCalculatorPowerOfTwoAdapter(HarborTaskAdapter):
    """Convert power-of-two JSON rows to Harbor task directories."""

    NAME = "simple-calculator-power-of-two"
    DEFAULT_SOURCE_DATA = (Path(__file__).resolve().parents[2] / "src" / "nat_simple_calculator_eval" / "data" /
                           "simple_calculator_power_of_two.json")

    def _load_benchmark_data(self) -> dict[str, TaskData]:
        records = json.loads(self.source_file.read_text(encoding="utf-8"))
        tasks: dict[str, TaskData] = {}
        for row in records:
            source_id = str(row["id"])
            question = str(row["question"])
            number = parse_power_of_two(question)
            tasks[source_id] = {
                "source_id": source_id,
                "question": question,
                "rubric": str(row.get("answer", "")),
                "number": number,
                "expected_value": number * number,
            }
        return tasks

    def _customize_task(self, output_dir: Path, task: TaskData, local_task_id: str) -> None:
        instruction = ("Solve the calculator question and write your final answer to /workspace/answer.txt.\n\n"
                       "Guidelines:\n"
                       "- Include the squared value in your answer.\n"
                       "- You can include additional explanation, but the numeric result must be explicit.\n\n"
                       f"Question:\n{task['question']}\n")
        self._write_task_files(
            output_dir,
            local_task_id=local_task_id,
            source_id=str(task["source_id"]),
            instruction=instruction,
            expected_value=float(task["expected_value"]),
            ground_truth={
                "id": task["source_id"],
                "question": task["question"],
                "rubric": task["rubric"],
                "operation": "power_of_two",
                "number": task["number"],
                "expected_value": task["expected_value"],
                "tolerance": 1e-4,
            },
            task_toml_replacements={'category = "math"': 'category = "power-of-two"'},
        )

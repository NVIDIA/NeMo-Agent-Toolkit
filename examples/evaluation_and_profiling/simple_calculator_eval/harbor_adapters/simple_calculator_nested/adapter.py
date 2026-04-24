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

"""Adapter for converting nested calculator examples to Harbor tasks."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common import copy_template
from common import number_to_string
from common import parse_basic_arithmetic
from common import write_ground_truth

DEFAULT_SOURCE_DATA = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "nat_simple_calculator_eval"
    / "data"
    / "simple_calculator_nested.json"
)


class SimpleCalculatorNestedAdapter:
    """Convert nested simple calculator JSON rows to Harbor task directories."""

    NAME = "simple-calculator-nested"

    def __init__(self, task_dir: Path, source_file: Path | None = None):
        self.task_dir = Path(task_dir)
        self.source_file = source_file or DEFAULT_SOURCE_DATA
        self.tasks = self._load_benchmark_data()

    @staticmethod
    def make_local_task_id(source_id: str) -> str:
        normalized = str(source_id).strip().lower().replace("_", "-")
        return f"simple-calculator-nested-{normalized}"

    def _load_benchmark_data(self) -> dict[str, dict]:
        payload = json.loads(self.source_file.read_text(encoding="utf-8"))
        rows = payload.get("questions", [])
        tasks: dict[str, dict] = {}
        for row in rows:
            source_id = str(row["id"])
            question = str(row["question"])
            operation, lhs, rhs, expected_value = parse_basic_arithmetic(question)
            tasks[source_id] = {
                "source_id": source_id,
                "question": question,
                "rubric": str(row.get("answer", "")),
                "category": str(row.get("category", "nested-calculator")),
                "difficulty": str(row.get("difficulty", "unknown")),
                "operation": operation,
                "lhs": lhs,
                "rhs": rhs,
                "expected_value": expected_value,
            }
        return tasks

    def list_available_tasks(self) -> list[str]:
        return sorted(self.tasks.keys(), key=lambda item: int(item))

    def generate_task(self, source_id: str, local_task_id: str, *, overwrite: bool = True) -> bool:
        if source_id not in self.tasks:
            raise KeyError(f"Unknown source_id: {source_id}")

        output_dir = self.task_dir / local_task_id
        if output_dir.exists():
            if not overwrite:
                return False
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        copy_template(output_dir)
        self._customize_task(output_dir, self.tasks[source_id], local_task_id)
        return True

    def generate_many(self, source_ids: list[str], *, overwrite: bool = True) -> tuple[int, int]:
        generated = 0
        for source_id in source_ids:
            local_task_id = self.make_local_task_id(source_id)
            if self.generate_task(source_id, local_task_id, overwrite=overwrite):
                generated += 1
        return generated, len(source_ids)

    def _customize_task(self, output_dir: Path, task: dict, local_task_id: str) -> None:
        instruction = (
            "Solve the calculator question and write your final answer to /workspace/answer.txt.\n\n"
            "Guidelines:\n"
            "- Include the computed arithmetic value in your answer.\n"
            "- Include any comparison requested by the question.\n"
            "- You can include additional explanation, but the numeric result must be explicit.\n\n"
            f"Question:\n{task['question']}\n"
        )
        (output_dir / "instruction.md").write_text(instruction, encoding="utf-8")

        task_toml = (output_dir / "task.toml").read_text(encoding="utf-8")
        task_toml = task_toml.replace("__TASK_NAME__", f"nvidia/{local_task_id}")
        task_toml = task_toml.replace("__TASK_ID__", task["source_id"])
        task_toml = task_toml.replace('category = "math"', f'category = "{task["category"]}"')
        task_toml = task_toml.replace('difficulty = "easy"', f'difficulty = "{task["difficulty"]}"')
        (output_dir / "task.toml").write_text(task_toml, encoding="utf-8")

        solution_script = (output_dir / "solution" / "solve.sh").read_text(encoding="utf-8")
        solution_script = solution_script.replace(
            "__EXPECTED_VALUE__", number_to_string(task["expected_value"])
        )
        (output_dir / "solution" / "solve.sh").write_text(solution_script, encoding="utf-8")
        (output_dir / "solution" / "solve.sh").chmod(0o755)
        (output_dir / "tests" / "test.sh").chmod(0o755)

        ground_truth = {
            "id": task["source_id"],
            "question": task["question"],
            "rubric": task["rubric"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "operation": task["operation"],
            "lhs": task["lhs"],
            "rhs": task["rhs"],
            "expected_value": task["expected_value"],
            "tolerance": 1e-4,
        }
        write_ground_truth(output_dir / "tests" / "ground_truth.json", ground_truth)


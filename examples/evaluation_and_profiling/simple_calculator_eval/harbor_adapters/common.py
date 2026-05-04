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
"""Shared helpers for simple calculator Harbor adapters."""

from __future__ import annotations

import json
import re
import shutil
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import ClassVar

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
_TEMPLATE_IGNORE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
TaskData = dict[str, Any]


def _is_ignored_template_item(path: Path) -> bool:
    """Return whether a top-level template item is generated/cache output."""
    return path.name in {"__pycache__", ".pytest_cache"} or path.suffix in {".pyc", ".pyo"}


def copy_template(output_dir: Path) -> None:
    """Copy shared adapter template into task output directory."""
    for item in TEMPLATE_DIR.iterdir():
        if _is_ignored_template_item(item):
            continue

        destination = output_dir / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*_TEMPLATE_IGNORE_PATTERNS),
            )
        else:
            shutil.copy2(item, destination)


class HarborTaskAdapter(ABC):
    """Shared generation flow for simple calculator Harbor task adapters."""

    NAME: ClassVar[str]
    DEFAULT_SOURCE_DATA: ClassVar[Path]

    def __init__(self, task_dir: Path, source_file: Path | None = None) -> None:
        self.task_dir = Path(task_dir)
        self.source_file = source_file or self.DEFAULT_SOURCE_DATA
        self.tasks = self._load_benchmark_data()

    @classmethod
    def make_local_task_id(cls, source_id: str) -> str:
        normalized = str(source_id).strip().lower().replace("_", "-")
        return f"{cls.NAME}-{normalized}"

    @abstractmethod
    def _load_benchmark_data(self) -> dict[str, TaskData]:
        """Load source benchmark data keyed by source ID."""

    def list_available_tasks(self) -> list[str]:
        return sorted(self.tasks.keys(), key=lambda item: int(item))

    def generate_task(self, source_id: str, local_task_id: str, *, overwrite: bool = True) -> bool:
        if source_id not in self.tasks:
            raise KeyError(f"Unknown source_id: {source_id}")

        output_dir = resolve_safe_task_dir(self.task_dir, local_task_id)
        if output_dir.exists():
            if not overwrite:
                return False
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        copy_template(output_dir)
        self._customize_task(output_dir, self.tasks[source_id], local_task_id)
        return True

    def generate_many(self, source_ids: Sequence[str], *, overwrite: bool = True) -> tuple[int, int]:
        generated = 0
        for source_id in source_ids:
            local_task_id = self.make_local_task_id(source_id)
            if self.generate_task(source_id, local_task_id, overwrite=overwrite):
                generated += 1
        return generated, len(source_ids)

    @abstractmethod
    def _customize_task(self, output_dir: Path, task: TaskData, local_task_id: str) -> None:
        """Write task-specific prompt, metadata, solution, and ground truth."""

    def _write_task_files(
        self,
        output_dir: Path,
        *,
        local_task_id: str,
        source_id: str,
        instruction: str,
        expected_value: float,
        ground_truth: TaskData,
        task_toml_replacements: Mapping[str, str] | None = None,
    ) -> None:
        (output_dir / "instruction.md").write_text(instruction, encoding="utf-8")

        task_toml = (output_dir / "task.toml").read_text(encoding="utf-8")
        task_toml = task_toml.replace("__TASK_NAME__", f"nvidia/{local_task_id}")
        task_toml = task_toml.replace("__TASK_ID__", source_id)
        for old, new in (task_toml_replacements or {}).items():
            task_toml = task_toml.replace(old, new)
        (output_dir / "task.toml").write_text(task_toml, encoding="utf-8")

        solution_script = (output_dir / "solution" / "solve.sh").read_text(encoding="utf-8")
        solution_script = solution_script.replace("__EXPECTED_VALUE__", number_to_string(expected_value))
        (output_dir / "solution" / "solve.sh").write_text(solution_script, encoding="utf-8")
        (output_dir / "solution" / "solve.sh").chmod(0o755)
        (output_dir / "tests" / "test.sh").chmod(0o755)

        write_ground_truth(output_dir / "tests" / "ground_truth.json", ground_truth)


def parse_basic_arithmetic(question: str) -> tuple[str, float, float, float]:
    """Extract operation and operands from simple arithmetic questions."""
    question_lower = question.lower()
    patterns: list[tuple[str, str]] = [
        ("product", r"product of (\d+) and (\d+)"),
        ("multiply", r"multiply (\d+) by (\d+)"),
        ("multiply", r"multiplying (\d+) by (\d+)"),
        ("multiply", r"(\d+) multiplied by (\d+)"),
        ("times", r"(\d+) times (\d+)"),
        ("sum", r"sum of (\d+) and (\d+)"),
        ("difference", r"difference between (\d+) and (\d+)"),
        ("divide", r"(\d+) divided by (\d+)"),
    ]
    for operation, pattern in patterns:
        match = re.search(pattern, question_lower)
        if not match:
            continue
        lhs = float(match.group(1))
        rhs = float(match.group(2))
        if operation in {"product", "multiply", "times"}:
            expected = lhs * rhs
        elif operation == "sum":
            expected = lhs + rhs
        elif operation == "difference":
            expected = lhs - rhs
        else:
            if rhs == 0:
                raise ValueError(f"Division by zero is not supported in question: {question}")
            expected = lhs / rhs
        return operation, lhs, rhs, expected
    raise ValueError(f"Unsupported question format: {question}")


def parse_power_of_two(question: str) -> float:
    """Extract base integer from power-of-two wording variants."""
    lowered = question.lower()
    patterns = [
        r"what is (\d+) to the power of 2",
        r"compute (\d+) squared",
        r"calculate (\d+) raised to the power of 2",
        r"evaluate (\d+)\^2",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return float(match.group(1))
    raise ValueError(f"Unsupported power-of-two question format: {question}")


def write_ground_truth(path: Path, payload: dict) -> None:
    """Write formatted ground truth JSON payload."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_safe_task_dir(task_dir: Path, local_task_id: str) -> Path:
    """Resolve a task output directory without allowing path traversal."""
    local_path = Path(local_task_id)
    if local_path.is_absolute() or ".." in local_path.parts:
        raise ValueError(f"Invalid local_task_id: {local_task_id}")

    root = task_dir.resolve()
    output_dir = (root / local_path).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Invalid local_task_id: {local_task_id}") from exc
    return output_dir


def number_to_string(value: float) -> str:
    """Format numbers for textual template replacement."""
    return str(int(value)) if value.is_integer() else str(value)

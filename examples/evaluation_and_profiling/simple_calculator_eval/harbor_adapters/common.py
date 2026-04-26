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
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"


def copy_template(output_dir: Path) -> None:
    """Copy shared adapter template into task output directory."""
    for item in TEMPLATE_DIR.iterdir():
        destination = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


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


def number_to_string(value: float) -> str:
    """Format numbers for textual template replacement."""
    return str(int(value)) if value.is_integer() else str(value)

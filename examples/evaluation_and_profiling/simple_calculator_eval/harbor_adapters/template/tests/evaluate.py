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
"""Strict numeric verifier for simple calculator tasks."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


def _load_ground_truth() -> dict:
    ground_truth_path = Path("/tests/ground_truth.json")
    if not ground_truth_path.exists():
        raise FileNotFoundError("Missing /tests/ground_truth.json")
    return json.loads(ground_truth_path.read_text(encoding="utf-8"))


def _extract_numbers(answer_text: str) -> list[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return [float(match) for match in matches]


def _write_details(details: dict[str, object]) -> None:
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


def main() -> int:
    ground_truth = _load_ground_truth()
    expected_value = float(ground_truth["expected_value"])
    tolerance = float(ground_truth.get("tolerance", 1e-4))
    details: dict[str, object] = {
        "expected_value": expected_value,
        "tolerance": tolerance,
        "candidate_values": [],
        "passed": False,
    }

    answer_path = Path("/workspace/answer.txt")
    if not answer_path.exists():
        print("Missing /workspace/answer.txt")
        details["error"] = "missing_answer_file"
        _write_details(details)
        return 1

    answer_text = answer_path.read_text(encoding="utf-8").strip()
    if not answer_text:
        print("Answer is empty")
        details["error"] = "empty_answer"
        _write_details(details)
        return 1

    candidate_values = _extract_numbers(answer_text)
    details["candidate_values"] = candidate_values
    if not candidate_values:
        print("No numeric values found in answer")
        details["error"] = "no_numeric_values"
        _write_details(details)
        return 1

    passed = any(math.isclose(value, expected_value, abs_tol=tolerance) for value in candidate_values)
    details["passed"] = passed
    _write_details(details)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

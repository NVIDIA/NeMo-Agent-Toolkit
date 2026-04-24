# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


def main() -> int:
    ground_truth = _load_ground_truth()
    answer_path = Path("/workspace/answer.txt")
    if not answer_path.exists():
        print("Missing /workspace/answer.txt")
        return 1

    answer_text = answer_path.read_text(encoding="utf-8").strip()
    if not answer_text:
        print("Answer is empty")
        return 1

    expected_value = float(ground_truth["expected_value"])
    tolerance = float(ground_truth.get("tolerance", 1e-4))
    candidate_values = _extract_numbers(answer_text)
    if not candidate_values:
        print("No numeric values found in answer")
        return 1

    passed = any(math.isclose(value, expected_value, abs_tol=tolerance) for value in candidate_values)
    details = {
        "expected_value": expected_value,
        "tolerance": tolerance,
        "candidate_values": candidate_values,
        "passed": passed,
    }
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())


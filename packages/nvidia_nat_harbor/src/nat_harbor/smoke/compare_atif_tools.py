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
"""Compare tool-call sequences between two ATIF trajectory artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATCH_SAME = "match (same)"
MATCH_RICHER = "match (richer)"
MATCH_POORER = "match (poorer)"
MISMATCH = "mismatch"


@dataclass(frozen=True)
class ToolSequenceComparison:
    """Tool sequence comparison between native and candidate ATIF artifacts."""

    classification: str
    native_tools: list[str]
    candidate_tools: list[str]

    @property
    def native_counts(self) -> Counter[str]:
        return Counter(self.native_tools)

    @property
    def candidate_counts(self) -> Counter[str]:
        return Counter(self.candidate_tools)


def load_atif(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_tool_sequence(trajectory: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for step in trajectory.get("steps", []):
        for tool_call in step.get("tool_calls") or []:
            tool_name = tool_call.get("function_name") or tool_call.get("name")
            if tool_name:
                tools.append(str(tool_name))
    return tools


def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return True

    index = 0
    for item in haystack:
        if item == needle[index]:
            index += 1
            if index == len(needle):
                return True
    return False


def classify_tool_sequences(native_tools: list[str], candidate_tools: list[str]) -> str:
    if native_tools == candidate_tools:
        return MATCH_SAME
    if is_subsequence(native_tools, candidate_tools):
        return MATCH_RICHER
    if is_subsequence(candidate_tools, native_tools):
        return MATCH_POORER
    return MISMATCH


def compare_atif_tool_sequences(native_path: Path, candidate_path: Path) -> ToolSequenceComparison:
    native_tools = extract_tool_sequence(load_atif(native_path))
    candidate_tools = extract_tool_sequence(load_atif(candidate_path))
    return ToolSequenceComparison(
        classification=classify_tool_sequences(native_tools, candidate_tools),
        native_tools=native_tools,
        candidate_tools=candidate_tools,
    )


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _format_sequence(tools: list[str]) -> str:
    return " -> ".join(tools) if tools else "(none)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare tool-call sequences between two ATIF trajectory files.")
    parser.add_argument("--native", required=True, type=Path, help="Native/reference ATIF trajectory path.")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate ATIF trajectory path.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = compare_atif_tool_sequences(args.native, args.candidate)

    if args.json:
        print(
            json.dumps(
                {
                    "classification": comparison.classification,
                    "native_tools": comparison.native_tools,
                    "candidate_tools": comparison.candidate_tools,
                    "native_counts": dict(comparison.native_counts),
                    "candidate_counts": dict(comparison.candidate_counts),
                },
                indent=2,
            ))
        return

    print(f"Classification: {comparison.classification}")
    print(f"Native tools ({len(comparison.native_tools)}): {_format_counts(comparison.native_counts)}")
    print(f"Candidate tools ({len(comparison.candidate_tools)}): {_format_counts(comparison.candidate_counts)}")
    print()
    print(f"Native sequence: {_format_sequence(comparison.native_tools)}")
    print(f"Candidate sequence: {_format_sequence(comparison.candidate_tools)}")


if __name__ == "__main__":
    main()

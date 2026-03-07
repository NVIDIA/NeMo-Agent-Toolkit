# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BFCL dataset loader — no LLM required."""

import json
import os

import pandas as pd
import pytest

from nat.plugins.benchmarks.bfcl.dataset import load_bfcl_dataset


def _make_bfcl_entry(entry_id: str, question: str = "Calculate area", func_name: str = "calc") -> dict:
    return {
        "id": entry_id,
        "question": [[{"role": "user", "content": question}]],
        "function": [{
            "name": func_name,
            "description": "A test function",
            "parameters": {
                "type": "dict",
                "properties": {"x": {"type": "integer", "description": "value"}},
                "required": ["x"],
            },
        }],
    }


def _make_bfcl_answer(entry_id: str) -> dict:
    return {
        "id": entry_id,
        "ground_truth": [{"calc": {"x": [10]}}],
    }


class TestBFCLDatasetLoader:

    def test_loads_jsonl_file(self, tmp_path):
        """Each line in the JSONL becomes one DataFrame row."""
        test_file = tmp_path / "BFCL_v3_simple.json"
        with open(test_file, "w") as f:
            for i in range(3):
                f.write(json.dumps(_make_bfcl_entry(f"simple_{i}")) + "\n")

        df = load_bfcl_dataset(str(test_file))
        assert len(df) == 3
        assert set(df.columns) >= {"id", "question", "answer"}

    def test_question_column_contains_full_entry(self, tmp_path):
        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            f.write(json.dumps(_make_bfcl_entry("simple_0", "Find area")) + "\n")

        df = load_bfcl_dataset(str(test_file))
        loaded = json.loads(df.iloc[0]["question"])
        assert loaded["id"] == "simple_0"
        assert loaded["function"][0]["name"] == "calc"
        assert loaded["question"][0][0]["content"] == "Find area"

    def test_answer_column_contains_ground_truth(self, tmp_path):
        """When possible_answer file exists, answers are loaded."""
        test_file = tmp_path / "BFCL_v3_simple.json"
        answer_dir = tmp_path / "possible_answer"
        answer_dir.mkdir()
        answer_file = answer_dir / "BFCL_v3_simple.json"

        with open(test_file, "w") as f:
            f.write(json.dumps(_make_bfcl_entry("simple_0")) + "\n")
        with open(answer_file, "w") as f:
            f.write(json.dumps(_make_bfcl_answer("simple_0")) + "\n")

        df = load_bfcl_dataset(str(test_file))
        answer = json.loads(df.iloc[0]["answer"])
        assert answer["ground_truth"] == [{"calc": {"x": [10]}}]

    def test_missing_answer_file_still_loads(self, tmp_path):
        """Dataset loads even without possible_answer file."""
        test_file = tmp_path / "test.json"
        with open(test_file, "w") as f:
            f.write(json.dumps(_make_bfcl_entry("test_0")) + "\n")

        df = load_bfcl_dataset(str(test_file))
        assert len(df) == 1
        answer = json.loads(df.iloc[0]["answer"])
        assert answer["ground_truth"] == []

    def test_raises_on_missing_file(self):
        with pytest.raises(ValueError, match="not found"):
            load_bfcl_dataset("/nonexistent/path.json")

    def test_loads_real_bfcl_simple_data(self):
        """Smoke test against installed bfcl package data."""
        try:
            from bfcl.constant import PROMPT_PATH
            simple_file = os.path.join(PROMPT_PATH, "BFCL_v3_simple.json")
            if not os.path.isfile(simple_file):
                pytest.skip("BFCL simple data file not found")
        except ImportError:
            pytest.skip("bfcl not installed")

        df = load_bfcl_dataset(simple_file, test_category="simple")
        assert len(df) > 0

        # Verify structure
        row = df.iloc[0]
        entry = json.loads(row["question"])
        assert "function" in entry
        assert "question" in entry

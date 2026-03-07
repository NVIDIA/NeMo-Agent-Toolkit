# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ToolTalk dataset loader — no LLM required."""

import json
import os
import tempfile

import pandas as pd
import pytest

from nat.plugins.benchmarks.tooltalk.dataset import load_tooltalk_dataset


def _make_conversation(conversation_id: str, user_text: str = "Hello") -> dict:
    """Create a minimal ToolTalk conversation for testing."""
    return {
        "name": f"test-{conversation_id}",
        "conversation_id": conversation_id,
        "suites_used": ["Alarm"],
        "apis_used": ["AddAlarm"],
        "scenario": "test scenario",
        "user": {"username": "testuser", "session_token": "abc-123"},
        "metadata": {
            "location": "New York",
            "timestamp": "2023-09-11 13:00:00",
            "session_token": "abc-123",
            "username": "testuser",
        },
        "conversation": [
            {"index": 0, "role": "user", "text": user_text},
            {
                "index": 1,
                "role": "assistant",
                "text": "Done.",
                "apis": [{
                    "request": {"api_name": "AddAlarm", "parameters": {"time": "18:30:00"}},
                    "response": {"alarm_id": "1234-5678"},
                    "exception": None,
                }],
            },
        ],
    }


class TestLoadToolTalkDataset:

    def test_loads_directory_of_json_files(self, tmp_path):
        """Each JSON file in the directory becomes one DataFrame row."""
        for i in range(3):
            conv = _make_conversation(f"conv-{i}", f"Message {i}")
            (tmp_path / f"conv-{i}.json").write_text(json.dumps(conv))

        df = load_tooltalk_dataset(str(tmp_path))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df.columns) >= {"id", "question", "answer"}

    def test_question_column_contains_full_conversation_json(self, tmp_path):
        """The question column should be the serialized full conversation."""
        conv = _make_conversation("conv-0")
        (tmp_path / "conv-0.json").write_text(json.dumps(conv))

        df = load_tooltalk_dataset(str(tmp_path))
        loaded = json.loads(df.iloc[0]["question"])

        assert loaded["conversation_id"] == "conv-0"
        assert loaded["conversation"][0]["role"] == "user"
        assert loaded["metadata"]["location"] == "New York"

    def test_id_column_uses_conversation_id(self, tmp_path):
        conv = _make_conversation("my-unique-id")
        (tmp_path / "test.json").write_text(json.dumps(conv))

        df = load_tooltalk_dataset(str(tmp_path))
        assert df.iloc[0]["id"] == "my-unique-id"

    def test_id_falls_back_to_filename_stem(self, tmp_path):
        """If conversation_id is missing, use the filename stem."""
        conv = _make_conversation("ignored")
        del conv["conversation_id"]
        (tmp_path / "fallback-name.json").write_text(json.dumps(conv))

        df = load_tooltalk_dataset(str(tmp_path))
        assert df.iloc[0]["id"] == "fallback-name"

    def test_raises_on_file_path_not_directory(self, tmp_path):
        file_path = tmp_path / "not-a-dir.json"
        file_path.write_text("{}")

        with pytest.raises(ValueError, match="must be a directory"):
            load_tooltalk_dataset(str(file_path))

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No JSON files found"):
            load_tooltalk_dataset(str(tmp_path))

    def test_loads_real_tooltalk_easy_data(self):
        """Smoke test against the installed tooltalk package's easy split."""
        try:
            import tooltalk
            tooltalk_dir = os.path.dirname(tooltalk.__file__)
            easy_dir = os.path.join(tooltalk_dir, "data", "easy")
            if not os.path.isdir(easy_dir):
                pytest.skip("tooltalk easy data directory not found")
        except ImportError:
            pytest.skip("tooltalk not installed")

        df = load_tooltalk_dataset(easy_dir)
        assert len(df) > 0
        assert "id" in df.columns

        # Verify each row is valid JSON
        for _, row in df.iterrows():
            conv = json.loads(row["question"])
            assert "conversation" in conv
            assert "metadata" in conv

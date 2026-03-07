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
"""Tests for Agent Leaderboard dataset loader — no network required for unit tests."""

import json
import os

import pandas as pd
import pytest

from nat.plugins.benchmarks.agent_leaderboard.dataset import (
    load_agent_leaderboard_dataset,
    _derive_expected_tool_calls,
)


def _make_sample_tools():
    return [
        {"title": "get_account_balance", "description": "Get the balance of an account", "properties": {}, "required": []},
        {"title": "transfer_funds", "description": "Transfer money between accounts", "properties": {}, "required": []},
        {"title": "get_exchange_rates", "description": "Get currency exchange rates", "properties": {}, "required": []},
        {"title": "schedule_appointment", "description": "Schedule a branch appointment", "properties": {}, "required": []},
    ]


def _make_sample_entry(idx: int = 0, domain: str = "banking"):
    tools = _make_sample_tools()
    return {
        "id": f"{domain}_scenario_{idx:03d}",
        "question": "I need to check my balance and transfer some money",
        "ground_truth": "User goals:\n- Check account balance\n- Transfer funds",
        "user_goals": ["Check account balance", "Transfer funds"],
        "available_tools": tools,
        "expected_tool_calls": [
            {"tool": "get_account_balance", "parameters": {}},
            {"tool": "transfer_funds", "parameters": {}},
        ],
        "metadata": {"domain": domain, "persona_name": "Test User", "num_goals": 2},
    }


class TestDeriveExpectedToolCalls:

    def test_matches_balance_goal(self):
        tools = _make_sample_tools()
        expected = _derive_expected_tool_calls(["Check my account balance"], tools)
        tool_names = [tc["tool"] for tc in expected]
        assert "get_account_balance" in tool_names

    def test_matches_transfer_goal(self):
        tools = _make_sample_tools()
        expected = _derive_expected_tool_calls(["Transfer money to savings"], tools)
        tool_names = [tc["tool"] for tc in expected]
        assert "transfer_funds" in tool_names

    def test_matches_exchange_goal(self):
        tools = _make_sample_tools()
        expected = _derive_expected_tool_calls(["What are the exchange rates?"], tools)
        tool_names = [tc["tool"] for tc in expected]
        assert "get_exchange_rates" in tool_names

    def test_no_match_returns_empty(self):
        tools = _make_sample_tools()
        expected = _derive_expected_tool_calls(["Tell me a joke"], tools)
        assert expected == []

    def test_deduplicates(self):
        tools = _make_sample_tools()
        expected = _derive_expected_tool_calls(
            ["Check balance", "Also check my balance again"], tools
        )
        tool_names = [tc["tool"] for tc in expected]
        assert tool_names.count("get_account_balance") == 1


class TestLoadFromFile:

    def test_loads_json_file(self, tmp_path):
        """Loads pre-downloaded JSON file."""
        entries = [_make_sample_entry(i) for i in range(3)]
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w") as f:
            json.dump(entries, f)

        df = load_agent_leaderboard_dataset(str(data_file))
        assert len(df) == 3
        assert set(df.columns) >= {"id", "question", "answer"}

    def test_question_contains_full_entry(self, tmp_path):
        entries = [_make_sample_entry()]
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(entries, f)

        df = load_agent_leaderboard_dataset(str(data_file))
        loaded = json.loads(df.iloc[0]["question"])
        assert "available_tools" in loaded
        assert "expected_tool_calls" in loaded
        assert loaded["user_goals"] == ["Check account balance", "Transfer funds"]

    def test_limit_parameter(self, tmp_path):
        entries = [_make_sample_entry(i) for i in range(10)]
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(entries, f)

        df = load_agent_leaderboard_dataset(str(data_file), limit=3)
        assert len(df) == 3

    def test_answer_contains_expected_calls(self, tmp_path):
        entries = [_make_sample_entry()]
        data_file = tmp_path / "test.json"
        with open(data_file, "w") as f:
            json.dump(entries, f)

        df = load_agent_leaderboard_dataset(str(data_file))
        answer = json.loads(df.iloc[0]["answer"])
        assert len(answer) == 2
        assert answer[0]["tool"] == "get_account_balance"


class TestHuggingFaceDownload:

    @pytest.mark.integration
    @pytest.mark.slow
    def test_downloads_banking_domain(self):
        """Downloads real data from HuggingFace — requires network."""
        try:
            from datasets import load_dataset
        except ImportError:
            pytest.skip("datasets not installed")

        df = load_agent_leaderboard_dataset(
            "/nonexistent/path.json",  # Will trigger HF download
            domains=["banking"],
            limit=3,
        )
        assert len(df) == 3
        entry = json.loads(df.iloc[0]["question"])
        assert "available_tools" in entry
        assert entry["metadata"]["domain"] == "banking"

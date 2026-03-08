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
"""Tests for config enum validation across all benchmarks.

Verifies that StrEnum fields reject invalid values and accept valid ones,
and that configs with validators enforce their constraints.
"""

import pytest
from pydantic import ValidationError

from nat.plugins.benchmarks.agent_leaderboard.config import AgentLeaderboardDatasetConfig
from nat.plugins.benchmarks.agent_leaderboard.config import AgentLeaderboardDomain
from nat.plugins.benchmarks.agent_leaderboard.config import TSQEvaluatorConfig
from nat.plugins.benchmarks.bfcl.config import BFCLDatasetConfig
from nat.plugins.benchmarks.bfcl.config import BFCLEvaluatorConfig
from nat.plugins.benchmarks.bfcl.config import BFCLTestCategory
from nat.plugins.benchmarks.tooltalk.config import ToolTalkApiMode
from nat.plugins.benchmarks.tooltalk.config import ToolTalkWorkflowConfig


class TestToolTalkApiMode:
    """Tests for ToolTalkApiMode enum."""

    def test_valid_values(self):
        assert ToolTalkApiMode.EXACT == "exact"
        assert ToolTalkApiMode.SUITE == "suite"
        assert ToolTalkApiMode.ALL == "all"

    def test_config_accepts_valid_string(self):
        config = ToolTalkWorkflowConfig(
            llm_name="test_llm",
            database_dir="/tmp/db",
            api_mode="exact",
        )
        assert config.api_mode == ToolTalkApiMode.EXACT

    def test_config_accepts_enum_value(self):
        config = ToolTalkWorkflowConfig(
            llm_name="test_llm",
            database_dir="/tmp/db",
            api_mode=ToolTalkApiMode.SUITE,
        )
        assert config.api_mode == ToolTalkApiMode.SUITE

    def test_config_rejects_invalid_value(self):
        with pytest.raises(ValidationError, match="api_mode"):
            ToolTalkWorkflowConfig(
                llm_name="test_llm",
                database_dir="/tmp/db",
                api_mode="invalid_mode",
            )

    def test_default_is_all(self):
        config = ToolTalkWorkflowConfig(
            llm_name="test_llm",
            database_dir="/tmp/db",
        )
        assert config.api_mode == ToolTalkApiMode.ALL


class TestBFCLTestCategory:
    """Tests for BFCLTestCategory enum."""

    def test_all_categories_exist(self):
        expected = [
            "simple",
            "multiple",
            "parallel",
            "parallel_multiple",
            "java",
            "javascript",
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
            "irrelevance",
            "live_irrelevance",
            "live_relevance",
        ]
        assert sorted(BFCLTestCategory) == sorted(expected)

    def test_dataset_config_accepts_valid(self):
        config = BFCLDatasetConfig(
            file_path="/tmp/data.json",
            test_category="parallel_multiple",
        )
        assert config.test_category == BFCLTestCategory.PARALLEL_MULTIPLE

    def test_dataset_config_rejects_invalid(self):
        with pytest.raises(ValidationError, match="test_category"):
            BFCLDatasetConfig(
                file_path="/tmp/data.json",
                test_category="nonexistent_category",
            )

    def test_evaluator_config_accepts_valid(self):
        config = BFCLEvaluatorConfig(test_category="irrelevance", language="Python")
        assert config.test_category == BFCLTestCategory.IRRELEVANCE

    def test_evaluator_config_default(self):
        config = BFCLEvaluatorConfig()
        assert config.test_category == BFCLTestCategory.SIMPLE
        assert config.language == "Python"


class TestBFCLLanguage:
    """Tests for BFCLEvaluatorConfig language field (Literal type)."""

    def test_valid_languages(self):
        for lang in ("Python", "Java", "JavaScript"):
            config = BFCLEvaluatorConfig(language=lang)
            assert config.language == lang

    def test_rejects_invalid_language(self):
        with pytest.raises(ValidationError):
            BFCLEvaluatorConfig(language="Ruby")

    def test_rejects_lowercase(self):
        """Language values are case-sensitive (Literal, not StrEnum)."""
        with pytest.raises(ValidationError):
            BFCLEvaluatorConfig(language="python")


class TestAgentLeaderboardDomain:
    """Tests for AgentLeaderboardDomain enum."""

    def test_all_domains(self):
        expected = {"banking", "healthcare", "insurance", "investment", "telecom"}
        assert set(AgentLeaderboardDomain) == expected

    def test_config_accepts_valid_domains(self):
        config = AgentLeaderboardDatasetConfig(
            file_path="/tmp/data.json",
            domains=["banking", "telecom"],
        )
        assert config.domains == [AgentLeaderboardDomain.BANKING, AgentLeaderboardDomain.TELECOM]

    def test_config_rejects_invalid_domain(self):
        with pytest.raises(ValidationError, match="domains"):
            AgentLeaderboardDatasetConfig(
                file_path="/tmp/data.json",
                domains=["banking", "automotive"],
            )

    def test_config_rejects_empty_domains(self):
        with pytest.raises(ValidationError, match="At least one domain"):
            AgentLeaderboardDatasetConfig(
                file_path="/tmp/data.json",
                domains=[],
            )

    def test_default_is_banking(self):
        config = AgentLeaderboardDatasetConfig(file_path="/tmp/data.json")
        assert config.domains == [AgentLeaderboardDomain.BANKING]


class TestTSQEvaluatorConfigConstraints:
    """Tests for TSQEvaluatorConfig weight constraints."""

    def test_valid_weights(self):
        config = TSQEvaluatorConfig(tool_weight=0.7, parameter_weight=0.3)
        assert config.tool_weight == 0.7
        assert config.parameter_weight == 0.3

    def test_rejects_negative_weight(self):
        with pytest.raises(ValidationError):
            TSQEvaluatorConfig(tool_weight=-0.1)

    def test_rejects_weight_over_one(self):
        with pytest.raises(ValidationError):
            TSQEvaluatorConfig(tool_weight=1.5)

    def test_defaults(self):
        config = TSQEvaluatorConfig()
        assert config.tool_weight == 1.0
        assert config.parameter_weight == 0.0

    def test_boundary_values(self):
        config = TSQEvaluatorConfig(tool_weight=0.0, parameter_weight=1.0)
        assert config.tool_weight == 0.0
        assert config.parameter_weight == 1.0


class TestToolTalkWorkflowConfigConstraints:
    """Tests for ToolTalkWorkflowConfig PositiveInt constraints."""

    def test_rejects_zero_max_tool_calls(self):
        with pytest.raises(ValidationError):
            ToolTalkWorkflowConfig(
                llm_name="test",
                database_dir="/tmp",
                max_tool_calls_per_turn=0,
            )

    def test_rejects_negative_max_tool_calls(self):
        with pytest.raises(ValidationError):
            ToolTalkWorkflowConfig(
                llm_name="test",
                database_dir="/tmp",
                max_tool_calls_per_turn=-1,
            )

    def test_accepts_positive(self):
        config = ToolTalkWorkflowConfig(
            llm_name="test",
            database_dir="/tmp",
            max_tool_calls_per_turn=5,
        )
        assert config.max_tool_calls_per_turn == 5

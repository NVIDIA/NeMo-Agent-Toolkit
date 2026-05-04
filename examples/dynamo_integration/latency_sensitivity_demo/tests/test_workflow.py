# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the latency sensitivity demo workflow."""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "src" / "latency_sensitivity_demo" / "configs"


class TestConfigFiles:
    """Verify all config files exist and have the correct structure."""

    def test_config_profile_exists(self):
        assert (CONFIGS_DIR / "config_profile.yml").exists()

    def test_config_with_trie_exists(self):
        assert (CONFIGS_DIR / "config_with_trie.yml").exists()

    def test_config_profile_has_prediction_trie(self):
        with open(CONFIGS_DIR / "config_profile.yml") as f:
            config = yaml.safe_load(f)
        profiler = config["eval"]["general"]["profiler"]
        assert profiler["prediction_trie"]["enable"] is True
        assert profiler["prediction_trie"]["auto_sensitivity"] is True

    def test_config_profile_has_sensitivity_weights(self):
        with open(CONFIGS_DIR / "config_profile.yml") as f:
            config = yaml.safe_load(f)
        trie_cfg = config["eval"]["general"]["profiler"]["prediction_trie"]
        assert "w_critical" in trie_cfg
        assert "w_fanout" in trie_cfg
        assert "w_position" in trie_cfg
        assert "w_parallel" in trie_cfg

    def test_config_declares_sub_functions(self):
        """All configs should declare the 7 sub-functions in the functions section."""
        expected = {
            "classify_query",
            "research_context",
            "lookup_policy",
            "check_compliance",
            "analyze_sentiment",
            "draft_response",
            "review_response",
        }
        for config_name in ["config_profile.yml", "config_with_trie.yml"]:
            with open(CONFIGS_DIR / config_name) as f:
                config = yaml.safe_load(f)
            functions = config.get("functions", {})
            assert expected == set(functions.keys()), (
                f"{config_name} functions mismatch: "
                f"missing={expected - set(functions.keys())}, extra={set(functions.keys()) - expected}")

    def test_config_workflow_references_sub_functions(self):
        """Workflow section should reference the 7 sub-functions."""
        with open(CONFIGS_DIR / "config_profile.yml") as f:
            config = yaml.safe_load(f)
        workflow = config["workflow"]
        assert workflow["classify_fn"] == "classify_query"
        assert workflow["research_fn"] == "research_context"
        assert workflow["policy_fn"] == "lookup_policy"
        assert workflow["compliance_fn"] == "check_compliance"
        assert workflow["sentiment_fn"] == "analyze_sentiment"
        assert workflow["draft_fn"] == "draft_response"
        assert workflow["review_fn"] == "review_response"


class TestDataset:
    """Verify the customer queries dataset."""

    def test_dataset_exists(self):
        data_path = (Path(__file__).parent.parent / "src" / "latency_sensitivity_demo" / "data" /
                     "customer_queries.json")
        assert data_path.exists()

    def test_dataset_has_entries(self):
        import json

        data_path = (Path(__file__).parent.parent / "src" / "latency_sensitivity_demo" / "data" /
                     "customer_queries.json")
        with open(data_path) as f:
            data = json.load(f)
        assert len(data) >= 5
        for entry in data:
            assert "id" in entry
            assert "question" in entry


class TestWorkflowRegistration:
    """Verify the workflow module can be imported and is registered."""

    def test_module_imports(self):
        from latency_sensitivity_demo import workflow
        assert workflow is not None

    def test_orchestrator_config_exists(self):
        from latency_sensitivity_demo.workflow import LatencySensitivityDemoConfig
        assert LatencySensitivityDemoConfig is not None

    def test_orchestrator_function_exists(self):
        from latency_sensitivity_demo.workflow import latency_sensitivity_demo_function
        assert latency_sensitivity_demo_function is not None

    def test_sub_function_configs_exist(self):
        from latency_sensitivity_demo.workflow import AnalyzeSentimentConfig
        from latency_sensitivity_demo.workflow import CheckComplianceConfig
        from latency_sensitivity_demo.workflow import ClassifyConfig
        from latency_sensitivity_demo.workflow import DraftResponseConfig
        from latency_sensitivity_demo.workflow import LookupPolicyConfig
        from latency_sensitivity_demo.workflow import ResearchContextConfig
        from latency_sensitivity_demo.workflow import ReviewConfig
        assert ClassifyConfig is not None
        assert ResearchContextConfig is not None
        assert LookupPolicyConfig is not None
        assert CheckComplianceConfig is not None
        assert AnalyzeSentimentConfig is not None
        assert DraftResponseConfig is not None
        assert ReviewConfig is not None

    def test_sub_function_registrations_exist(self):
        from latency_sensitivity_demo.workflow import analyze_sentiment_function
        from latency_sensitivity_demo.workflow import check_compliance_function
        from latency_sensitivity_demo.workflow import classify_query_function
        from latency_sensitivity_demo.workflow import draft_response_function
        from latency_sensitivity_demo.workflow import lookup_policy_function
        from latency_sensitivity_demo.workflow import research_context_function
        from latency_sensitivity_demo.workflow import review_response_function
        assert classify_query_function is not None
        assert research_context_function is not None
        assert lookup_policy_function is not None
        assert check_compliance_function is not None
        assert analyze_sentiment_function is not None
        assert draft_response_function is not None
        assert review_response_function is not None

    def test_orchestrator_has_function_refs(self):
        from latency_sensitivity_demo.workflow import LatencySensitivityDemoConfig
        fields = LatencySensitivityDemoConfig.model_fields
        assert "classify_fn" in fields
        assert "research_fn" in fields
        assert "policy_fn" in fields
        assert "compliance_fn" in fields
        assert "sentiment_fn" in fields
        assert "draft_fn" in fields
        assert "review_fn" in fields


class TestSensitivityReport:
    """Verify the sensitivity report module."""

    def test_report_module_imports(self):
        from latency_sensitivity_demo import sensitivity_report
        assert sensitivity_report is not None

    def test_print_report_with_empty_trie(self, capsys):
        from latency_sensitivity_demo.sensitivity_report import print_report

        from nat.profiler.prediction_trie.data_models import PredictionTrieNode

        empty_root = PredictionTrieNode(name="root")
        print_report(empty_root)
        captured = capsys.readouterr()
        assert "No prediction data found" in captured.out

    def test_print_report_with_sample_trie(self, capsys):
        from latency_sensitivity_demo.sensitivity_report import print_report

        from nat.profiler.prediction_trie.data_models import LLMCallPrediction
        from nat.profiler.prediction_trie.data_models import PredictionMetrics
        from nat.profiler.prediction_trie.data_models import PredictionTrieNode

        child = PredictionTrieNode(
            name="classify",
            predictions_by_call_index={
                1:
                    LLMCallPrediction(
                        remaining_calls=PredictionMetrics(sample_count=8, mean=4.0),
                        interarrival_ms=PredictionMetrics(sample_count=8, mean=150.0),
                        output_tokens=PredictionMetrics(sample_count=8, mean=20.0),
                        latency_sensitivity=5,
                    )
            },
        )
        root = PredictionTrieNode(name="root", children={"classify": child})
        print_report(root)
        captured = capsys.readouterr()
        assert "LATENCY SENSITIVITY REPORT" in captured.out
        assert "classify" in captured.out
        assert "5/5" in captured.out

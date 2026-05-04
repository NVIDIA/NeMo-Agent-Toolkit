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
"""
Unit tests for the self-evaluating agent.

These tests verify that the self-evaluation wrapper correctly:
- Evaluates tool call sequences
- Retries when confidence is below threshold
- Passes feedback to the agent on retry
- Returns the best result after max retries
"""

from pathlib import Path

import pytest

# Get the configs directory relative to this test file
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestSelfEvaluatingAgentConfig:
    """Test self-evaluating agent configuration loading."""

    def test_config_file_exists(self):
        """Verify the self-evaluation config file exists."""
        config_path = CONFIGS_DIR / "eval_config_rethinking_full_test.yml"
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_profile_config_file_exists(self):
        """Verify the profile config with feedback exists."""
        config_path = CONFIGS_DIR / "profile_rethinking_full_test.yml"
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_contains_self_evaluating_agent(self):
        """Verify the config defines a self_evaluating_agent workflow."""
        import yaml

        config_path = CONFIGS_DIR / "eval_config_rethinking_full_test.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "workflow" in config, "Config must have workflow section"
        assert config["workflow"]["_type"] == "self_evaluating_agent_with_feedback", (
            "Workflow type must be self_evaluating_agent_with_feedback")

    def test_config_has_required_parameters(self):
        """Verify the config has all required self-evaluation parameters."""
        import yaml

        config_path = CONFIGS_DIR / "eval_config_rethinking_full_test.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        workflow = config["workflow"]
        assert "wrapped_agent" in workflow, "Must specify wrapped_agent"
        assert "evaluator_llm" in workflow, "Must specify evaluator_llm"
        assert "max_retries" in workflow, "Must specify max_retries"
        assert "min_confidence_threshold" in workflow, "Must specify min_confidence_threshold"

    def test_config_max_retries_in_range(self):
        """Verify max_retries is within acceptable range (0-10)."""
        import yaml

        config_path = CONFIGS_DIR / "eval_config_rethinking_full_test.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        max_retries = config["workflow"]["max_retries"]
        assert 0 <= max_retries <= 10, f"max_retries should be 0-10, got {max_retries}"

    def test_config_confidence_threshold_in_range(self):
        """Verify confidence threshold is within acceptable range (0.0-1.0)."""
        import yaml

        config_path = CONFIGS_DIR / "eval_config_rethinking_full_test.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        threshold = config["workflow"]["min_confidence_threshold"]
        assert 0.0 <= threshold <= 1.0, f"threshold should be 0.0-1.0, got {threshold}"


class TestSelfEvaluatingAgentModule:
    """Test self-evaluating agent module imports and registration."""

    def test_module_imports(self):
        """Verify the self-evaluating agent module can be imported."""
        from react_benchmark_agent import self_evaluating_agent_with_feedback

        assert self_evaluating_agent_with_feedback is not None

    def test_config_class_exists(self):
        """Verify SelfEvaluatingAgentWithFeedbackConfig class exists."""
        from react_benchmark_agent.self_evaluating_agent_with_feedback import SelfEvaluatingAgentWithFeedbackConfig

        assert SelfEvaluatingAgentWithFeedbackConfig is not None


class TestEvaluationResponseParsing:
    """Test parsing of evaluation responses."""

    @staticmethod
    def parse_evaluation_response(response_text: str) -> dict:
        """
        Parse evaluation response from LLM.
        Mirrors the logic in self_evaluating_agent_with_feedback.py.
        """
        import json

        # Find JSON in the response (it might have extra text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            try:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Default response if parsing fails
        return {
            "is_sufficient": False,
            "confidence": 0.0,
            "reasoning": "Failed to parse evaluation response",
            "missing_steps": [],
            "suggestions": "",
        }

    def test_parse_valid_json(self):
        """Test parsing a valid JSON evaluation response."""
        response = """
        Based on my analysis, here is the evaluation:
        {
            "is_sufficient": true,
            "confidence": 0.85,
            "reasoning": "All required tools were called",
            "missing_steps": [],
            "suggestions": ""
        }
        """
        result = self.parse_evaluation_response(response)
        assert result["is_sufficient"] is True
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "All required tools were called"

    def test_parse_insufficient_response(self):
        """Test parsing an insufficient evaluation response."""
        response = """
        {
            "is_sufficient": false,
            "confidence": 0.45,
            "reasoning": "Missing verification step",
            "missing_steps": ["get_transaction_history"],
            "suggestions": "Add verification after transfer"
        }
        """
        result = self.parse_evaluation_response(response)
        assert result["is_sufficient"] is False
        assert result["confidence"] == 0.45
        assert "get_transaction_history" in result["missing_steps"]

    def test_parse_malformed_json(self):
        """Test parsing returns default when JSON is malformed."""
        response = "This is not valid JSON at all"
        result = self.parse_evaluation_response(response)
        assert result["is_sufficient"] is False
        assert result["confidence"] == 0.0

    def test_parse_partial_json(self):
        """Test parsing partial JSON embedded in text."""
        response = """
        Let me evaluate this...

        {"is_sufficient": true, "confidence": 0.92, "reasoning": "Good", "missing_steps": [], "suggestions": ""}

        That concludes my evaluation.
        """
        result = self.parse_evaluation_response(response)
        assert result["is_sufficient"] is True
        assert result["confidence"] == 0.92


class TestDecisionLogic:
    """Test the self-evaluation decision logic."""

    @staticmethod
    def should_accept(is_sufficient: bool, confidence: float, threshold: float) -> bool:
        """
        Determine if the tool sequence should be accepted.
        Mimics the decision logic in self_evaluating_agent.py.
        """
        return is_sufficient and confidence >= threshold

    @staticmethod
    def should_retry(is_sufficient: bool, confidence: float, threshold: float, retries_left: int) -> bool:
        """
        Determine if the agent should retry.
        """
        if retries_left <= 0:
            return False
        return not (is_sufficient and confidence >= threshold)

    def test_accept_sufficient_and_confident(self):
        """Accept when sufficient and above threshold."""
        assert self.should_accept(True, 0.85, 0.70) is True

    def test_reject_sufficient_but_not_confident(self):
        """Reject when sufficient but below threshold."""
        assert self.should_accept(True, 0.50, 0.70) is False

    def test_reject_not_sufficient(self):
        """Reject when not sufficient regardless of confidence."""
        assert self.should_accept(False, 0.95, 0.70) is False

    def test_accept_at_exact_threshold(self):
        """Accept when confidence equals threshold exactly."""
        assert self.should_accept(True, 0.70, 0.70) is True

    def test_retry_when_not_sufficient(self):
        """Retry when not sufficient and retries available."""
        assert self.should_retry(False, 0.85, 0.70, 2) is True

    def test_retry_when_not_confident(self):
        """Retry when not confident enough and retries available."""
        assert self.should_retry(True, 0.50, 0.70, 2) is True

    def test_no_retry_when_accepted(self):
        """Don't retry when sequence is accepted."""
        assert self.should_retry(True, 0.85, 0.70, 2) is False

    def test_no_retry_when_exhausted(self):
        """Don't retry when no retries left."""
        assert self.should_retry(False, 0.50, 0.70, 0) is False


class TestFeedbackGeneration:
    """Test feedback message generation."""

    @staticmethod
    def generate_feedback(
        reasoning: str,
        missing_steps: list,
        suggestions: str,
        template: str | None = None,
    ) -> str:
        """
        Generate feedback message for retry.
        Mimics the logic in self_evaluating_agent_with_feedback.py.
        """
        if template is None:
            template = """
PREVIOUS ATTEMPT FEEDBACK:

Your previous tool selection was evaluated and found to be insufficient.

EVALUATION:
{reasoning}

MISSING STEPS:
{missing_steps}

SUGGESTIONS:
{suggestions}

Please try again, addressing the issues identified above.
"""
        missing_steps_str = "\n".join(f"- {step}" for step in missing_steps) if missing_steps else "None identified"

        return template.format(
            reasoning=reasoning,
            missing_steps=missing_steps_str,
            suggestions=suggestions or "None provided",
        )

    def test_generate_basic_feedback(self):
        """Test basic feedback generation."""
        feedback = self.generate_feedback(
            reasoning="Missing verification step",
            missing_steps=["get_transaction_history"],
            suggestions="Add verification after transfer",
        )
        assert "Missing verification step" in feedback
        assert "get_transaction_history" in feedback
        assert "Add verification" in feedback

    def test_generate_feedback_empty_missing_steps(self):
        """Test feedback with no missing steps."""
        feedback = self.generate_feedback(reasoning="Could be better", missing_steps=[], suggestions="Try harder")
        assert "None identified" in feedback

    def test_generate_feedback_multiple_missing_steps(self):
        """Test feedback with multiple missing steps."""
        feedback = self.generate_feedback(
            reasoning="Incomplete",
            missing_steps=["step_a", "step_b", "step_c"],
            suggestions="",
        )
        assert "step_a" in feedback
        assert "step_b" in feedback
        assert "step_c" in feedback

    def test_generate_feedback_custom_template(self):
        """Test feedback with custom template."""
        custom_template = "Issues: {reasoning}\nMissing: {missing_steps}\nTips: {suggestions}"
        feedback = self.generate_feedback(
            reasoning="Test reason",
            missing_steps=["test_step"],
            suggestions="Test tip",
            template=custom_template,
        )
        assert "Issues: Test reason" in feedback
        assert "test_step" in feedback
        assert "Tips: Test tip" in feedback


@pytest.mark.integration
class TestSelfEvaluatingAgentWithNIM:
    """
    Integration tests for self-evaluating agent using NVIDIA NIM API.

    These tests require NVIDIA_API_KEY environment variable to be set.
    Run with: pytest --run_slow --run_integration
    """

    @pytest.fixture(name="nim_self_eval_config")
    def fixture_nim_self_eval_config(self, nvidia_api_key, tmp_path):
        """Create a test config using NVIDIA NIM API for self-evaluation."""
        import yaml

        # Load the base config
        config_path = CONFIGS_DIR / "eval_config_rethinking_full_test.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Replace Dynamo LLM with NIM
        config["llms"]["dynamo_llm"] = {
            "_type": "nim",
            "model_name": "meta/llama-3.1-8b-instruct",
            "temperature": 0.0,
            "max_tokens": 2048,
            "stop": ["Observation:", "\nThought:"],
        }

        config["llms"]["eval_llm"] = {
            "_type": "nim",
            "model_name": "meta/llama-3.1-8b-instruct",
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        # Reduce max_retries for faster testing
        config["workflow"]["max_retries"] = 2
        config["workflow"]["min_confidence_threshold"] = 0.7

        # Write temp config
        temp_config = tmp_path / "nim_self_eval_config.yml"
        with open(temp_config, "w") as f:
            yaml.dump(config, f)

        return temp_config

    async def test_self_evaluation_workflow_loads_with_nim(self, nim_self_eval_config):
        """Test that the self-evaluation workflow can be loaded with NIM backend."""
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.runtime.loader import load_config

        config = load_config(str(nim_self_eval_config))

        async with WorkflowBuilder.from_config(config) as builder:
            workflow = builder.get_workflow()
            assert workflow is not None

    async def test_self_evaluation_rethinking_with_nim(self, nim_self_eval_config):
        """
        Test the self-evaluation re-thinking mechanism with NIM.

        This test verifies that:
        1. The agent can process a banking question
        2. The self-evaluator assesses the tool sequence
        3. The agent may retry if confidence is below threshold
        """
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.runtime.loader import load_config

        config = load_config(str(nim_self_eval_config))

        async with WorkflowBuilder.from_config(config) as builder:
            workflow = builder.get_workflow()

            # Use a simple question that should trigger tool selection
            question = "Check my account balance for account 12345"

            result = await workflow.ainvoke(question)

            # Verify we got a response
            assert result is not None
            assert len(result) > 0, "Expected non-empty response from self-evaluating agent"

    async def test_self_evaluation_complex_question_with_nim(self, nim_self_eval_config):
        """
        Test self-evaluation with a more complex multi-step question.

        This tests the re-thinking loop with a question that may require
        multiple tool calls, potentially triggering retries.
        """
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.runtime.loader import load_config

        config = load_config(str(nim_self_eval_config))

        async with WorkflowBuilder.from_config(config) as builder:
            workflow = builder.get_workflow()

            # A complex question requiring multiple tools
            question = """
            I need to:
            1. Check my checking account balance
            2. Transfer $500 to my savings account
            """

            result = await workflow.ainvoke(question)

            # Verify we got a response
            assert result is not None
            assert len(result) > 0, "Expected non-empty response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

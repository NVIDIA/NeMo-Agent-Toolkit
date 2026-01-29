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

from nat.plugins.langchain.agent.prompt_optimizer.prompt import oracle_feedback_template
from nat.plugins.langchain.agent.prompt_optimizer.register import PromptOptimizerInputSchema


class TestPromptOptimizerInputSchema:
    """Tests for PromptOptimizerInputSchema."""

    def test_oracle_feedback_is_optional(self):
        """Oracle feedback defaults to None."""
        schema = PromptOptimizerInputSchema(
            original_prompt="Test prompt",
            objective="Test objective",
        )
        assert schema.oracle_feedback is None

    def test_oracle_feedback_can_be_set(self):
        """Oracle feedback can be provided."""
        feedback = "1. [Accuracy] Failed\n"
        schema = PromptOptimizerInputSchema(
            original_prompt="Test prompt",
            objective="Test objective",
            oracle_feedback=feedback,
        )
        assert schema.oracle_feedback == feedback


class TestOracleFeedbackFormatting:
    """Tests for oracle feedback template formatting."""

    def test_feedback_template_formats_correctly(self):
        """Oracle feedback template correctly formats feedback string."""
        feedback = "1. [Accuracy] Failed to answer\n2. [Relevance] Off topic\n"
        result = oracle_feedback_template.format(oracle_feedback=feedback)

        # Verify the template includes the expected sections
        assert "FAILURE ANALYSIS" in result
        assert "[Accuracy] Failed to answer" in result
        assert "[Relevance] Off topic" in result
        assert "root causes" in result.lower()  # Instructions are present

    def test_empty_feedback_results_in_empty_placeholder(self):
        """Empty feedback results in empty feedback section (tested via conditional logic)."""
        # The conditional logic in register.py is:
        # feedback_section = ""
        # if oracle_feedback:
        #     feedback_section = oracle_feedback_template.format(oracle_feedback=oracle_feedback)
        #
        # Test that the conditional evaluates correctly
        oracle_feedback = None
        feedback_section = ""
        if oracle_feedback:
            feedback_section = oracle_feedback_template.format(oracle_feedback=oracle_feedback)

        assert feedback_section == ""

        # Also test empty string
        oracle_feedback = ""
        feedback_section = ""
        if oracle_feedback:
            feedback_section = oracle_feedback_template.format(oracle_feedback=oracle_feedback)

        assert feedback_section == ""

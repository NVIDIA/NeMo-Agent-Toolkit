# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.agent.prompt_optimizer.prompt import mutator_prompt
from nat.agent.prompt_optimizer.prompt import oracle_feedback_template


class TestPromptTemplates:
    """Tests for prompt optimizer templates."""

    def test_mutator_prompt_has_feedback_placeholder(self):
        """Mutator prompt includes oracle_feedback_section placeholder."""
        assert "{oracle_feedback_section}" in mutator_prompt

    def test_oracle_feedback_template_has_feedback_placeholder(self):
        """Oracle feedback template includes oracle_feedback placeholder."""
        assert "{oracle_feedback}" in oracle_feedback_template

    def test_oracle_feedback_template_formatting(self):
        """Oracle feedback template formats correctly."""
        feedback = "1. [Accuracy] Failed to answer\n2. [Relevance] Off topic\n"
        result = oracle_feedback_template.format(oracle_feedback=feedback)
        assert "FAILURE ANALYSIS" in result
        assert "[Accuracy] Failed to answer" in result
        assert "[Relevance] Off topic" in result

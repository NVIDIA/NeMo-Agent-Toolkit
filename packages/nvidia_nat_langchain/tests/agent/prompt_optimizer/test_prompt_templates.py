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

from nat.plugins.langchain.agent.prompt_optimizer.prompt import mutator_prompt
from nat.plugins.langchain.agent.prompt_optimizer.prompt import oracle_feedback_template


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

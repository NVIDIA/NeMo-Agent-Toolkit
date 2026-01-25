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
"""Tests for system prompts."""

from nat_sandbox_agent.prompts import SANDBOX_AGENT_SYSTEM_PROMPT
from nat_sandbox_agent.prompts import get_system_prompt


class TestSystemPromptV1:
    """Tests for v1 system prompt generation."""

    def test_default_prompt_exists(self):
        """Test that default system prompt is defined."""
        assert SANDBOX_AGENT_SYSTEM_PROMPT is not None
        assert len(SANDBOX_AGENT_SYSTEM_PROMPT) > 0

    def test_default_prompt_contains_capabilities(self):
        """Test that default prompt describes capabilities."""
        prompt = SANDBOX_AGENT_SYSTEM_PROMPT

        assert "shell" in prompt.lower()
        assert "python" in prompt.lower()
        assert "file" in prompt.lower()
        assert "web" in prompt.lower()

    def test_get_prompt_with_additional_instructions(self):
        """Test getting prompt with additional instructions."""
        additional = "Always respond in JSON format."
        prompt = get_system_prompt(additional_instructions=additional)

        assert SANDBOX_AGENT_SYSTEM_PROMPT in prompt
        assert additional in prompt

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest


class TestHaystackDeepResearchAgent:
    """Test suite for the Haystack Deep Research Agent workflow"""

    def test_import_modules(self):
        """Test that all modules can be imported without errors"""
        try:
            from aiq_haystack_deep_research_agent import register
            # Note: Individual tool modules are no longer used since everything is in register.py
        except ImportError as e:
            pytest.fail(f"Failed to import modules: {e}")

    def test_config_classes_exist(self):
        """Test that configuration classes are properly defined"""
        from aiq_haystack_deep_research_agent.register import HaystackDeepResearchWorkflowConfig

        # Verify that the configuration class can be instantiated
        workflow_config = HaystackDeepResearchWorkflowConfig(llm="test_llm")
        assert workflow_config.max_agent_steps == 20
        assert "deep research assistant" in workflow_config.system_prompt

    @pytest.mark.e2e
    async def test_workflow_integration(self):
        """End-to-end test of the workflow (requires API keys and OpenSearch)"""
        # This test requires:
        # - OPENAI_API_KEY environment variable
        # - SERPERDEV_API_KEY environment variable
        # - OpenSearch running on localhost:9200 (optional for search-only mode)
        # Run with: pytest -m e2e

        import os
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPERDEV_API_KEY"):
            pytest.skip("API keys not available for end-to-end testing")

        # This is a placeholder for a full end-to-end test
        # In a real implementation, you would:
        # 1. Start OpenSearch (or mock it) - workflow now gracefully degrades without it
        # 2. Create the workflow with proper configuration
        # 3. Test with a simple query like "What is artificial intelligence?"
        # 4. Verify the response format and that it includes web search results

        assert True  # Placeholder assertion
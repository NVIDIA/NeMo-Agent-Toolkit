# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for LLM endpoint validation before evaluation."""

import pytest
from unittest.mock import MagicMock, patch

from nat.data_models.config import Config
from nat.llm.openai_llm import OpenAIModelConfig
from nat.eval.llm_validator import validate_llm_endpoints


class TestLLMEndpointValidation:
    """Tests for LLM endpoint validation functionality."""

    @pytest.fixture
    def config_with_openai_llm(self):
        """Create config with OpenAI-compatible LLM."""
        config = Config()
        config.llms = {
            "test_llm": OpenAIModelConfig(
                model_name="test-model",
                base_url="http://localhost:8000/v1"
            )
        }
        return config

    @pytest.fixture
    def config_with_multiple_llms(self):
        """Create config with multiple LLMs."""
        config = Config()
        config.llms = {
            "llm1": OpenAIModelConfig(
                model_name="model1",
                base_url="http://localhost:8000/v1"
            ),
            "llm2": OpenAIModelConfig(
                model_name="model2",
                base_url="http://localhost:8001/v1"
            )
        }
        return config

    @pytest.fixture
    def config_without_llms(self):
        """Create config without any LLMs."""
        config = Config()
        config.llms = {}
        return config

    async def test_validation_with_no_llms_configured(self, config_without_llms):
        """Test validation succeeds when no LLMs are configured."""
        # Should not raise any error
        await validate_llm_endpoints(config_without_llms)

    async def test_validation_detects_unreachable_endpoint(self):
        """Test that validation detects unreachable endpoints."""
        config = Config()
        config.llms = {
            "unreachable_llm": OpenAIModelConfig(
                model_name="test-model",
                base_url="http://localhost:9999/v1"  # Non-existent endpoint
            )
        }

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config)

        error_msg = str(exc_info.value)
        assert "LLM endpoint validation failed" in error_msg
        assert "ACTION REQUIRED" in error_msg

    @patch('nat.eval.llm_validator.openai')
    async def test_validation_detects_model_not_found_404(self, mock_openai, config_with_openai_llm):
        """Test that validation detects 404 errors when model doesn't exist."""
        # Import after patching to get the right exception class
        import openai

        # Mock OpenAI client to raise NotFoundError (404)
        mock_client = MagicMock()
        mock_client.models.list.side_effect = openai.NotFoundError(
            message="Model not found",
            response=MagicMock(status_code=404),
            body=None
        )
        mock_openai.OpenAI.return_value = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config_with_openai_llm)

        error_msg = str(exc_info.value)
        assert "404" in error_msg
        assert "not found" in error_msg.lower()
        assert any(phrase in error_msg for phrase in [
            "This typically means",
            "ACTION REQUIRED",
            "model has not been deployed"
        ])

    @patch('nat.eval.llm_validator.openai')
    async def test_validation_succeeds_with_accessible_endpoint(self, mock_openai, config_with_openai_llm):
        """Test that validation succeeds when endpoint is accessible."""
        # Mock successful connection
        mock_client = MagicMock()
        mock_models_response = MagicMock()
        mock_models_response.data = [MagicMock(id="test-model")]
        mock_client.models.list.return_value = mock_models_response
        mock_openai.OpenAI.return_value = mock_client

        # Should not raise any error
        await validate_llm_endpoints(config_with_openai_llm)

    async def test_validation_skips_non_openai_llm_types(self):
        """Test that validation skips non-OpenAI compatible LLM types."""
        config = Config()
        # Mock a non-OpenAI LLM type
        mock_llm = MagicMock()
        mock_llm.type = "bedrock"  # Non-OpenAI type
        config.llms = {"bedrock_llm": mock_llm}

        # Should not raise error for non-OpenAI LLMs
        await validate_llm_endpoints(config)

    async def test_validation_handles_llm_without_base_url(self):
        """Test that validation handles LLMs without base_url gracefully."""
        config = Config()
        mock_llm = MagicMock()
        mock_llm.type = "openai"
        mock_llm.base_url = None  # No base_url
        config.llms = {"no_url_llm": mock_llm}

        # Should not raise error, just skip validation
        await validate_llm_endpoints(config)

    @patch('nat.eval.llm_validator.openai')
    async def test_validation_fails_on_first_bad_endpoint(self, mock_openai, config_with_multiple_llms):
        """Test that validation fails fast on first bad endpoint."""
        # First endpoint fails
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ConnectionError("Connection refused")
        mock_openai.OpenAI.return_value = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config_with_multiple_llms)

        error_msg = str(exc_info.value)
        # Should mention the first failing LLM
        assert "llm1" in error_msg or "LLM endpoint validation failed" in error_msg


class TestLLMValidationErrorMessages:
    """Tests for error message quality and actionability."""

    async def test_error_message_includes_endpoint_details(self):
        """Test that error messages include specific endpoint details."""
        config = Config()
        config.llms = {
            "training_llm": OpenAIModelConfig(
                model_name="custom-model-name",
                base_url="http://custom-host:8000/v1"
            )
        }

        try:
            await validate_llm_endpoints(config)
        except RuntimeError as e:
            error_msg = str(e)
            # Should include the LLM name
            assert "training_llm" in error_msg
            # Should include the base URL
            assert "http://custom-host:8000/v1" in error_msg

    async def test_error_message_provides_troubleshooting_steps(self):
        """Test that error messages include actionable troubleshooting steps."""
        config = Config()
        config.llms = {
            "test_llm": OpenAIModelConfig(
                model_name="test-model",
                base_url="http://localhost:9999/v1"
            )
        }

        try:
            await validate_llm_endpoints(config)
        except RuntimeError as e:
            error_msg = str(e)
            # Should include actionable guidance
            assert any(keyword in error_msg for keyword in [
                "ACTION REQUIRED",
                "Check",
                "Verify",
                "Ensure"
            ])

    @patch('nat.eval.llm_validator.openai')
    async def test_404_error_message_mentions_training_cancellation(self, mock_openai):
        """Test that 404 error message mentions potential training cancellation."""
        import openai

        config = Config()
        config.llms = {
            "finetuned_model": OpenAIModelConfig(
                model_name="finetuned-llama",
                base_url="http://localhost:8000/v1"
            )
        }

        # Mock 404 error
        mock_client = MagicMock()
        mock_client.models.list.side_effect = openai.NotFoundError(
            message="Not found",
            response=MagicMock(status_code=404),
            body=None
        )
        mock_openai.OpenAI.return_value = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config)

        error_msg = str(exc_info.value)
        # Should mention training-related causes
        assert any(phrase in error_msg.lower() for phrase in [
            "training",
            "deployed",
            "canceled",
            "model has not been deployed"
        ])


class TestLLMValidationIntegration:
    """Integration tests for LLM validation with evaluation flow."""

    @pytest.fixture
    def config_for_finetuned_model(self):
        """Create config simulating post-training scenario."""
        config = Config()
        config.llms = {
            "training_llm": OpenAIModelConfig(
                model_name="default/meta-llama-3.1-8b-instruct-nat-dpo",
                base_url="http://nim-endpoint:8000/v1"
            )
        }
        return config

    async def test_validation_scenario_after_canceled_training(self, config_for_finetuned_model):
        """
        Test validation behavior in the scenario that caused the original bug:
        Training was canceled, model never deployed, user tries to run eval.
        """
        # This simulates the exact bug scenario:
        # 1. Training was canceled at 93.3%
        # 2. Model was never deployed
        # 3. User tries to run evaluation

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config_for_finetuned_model)

        error_msg = str(exc_info.value)

        # Validation should:
        # 1. Detect the missing model early (before eval starts)
        # 2. Provide clear error about what went wrong
        # 3. Give actionable next steps

        assert any(check in error_msg for check in [
            "LLM endpoint validation failed",
            "not found",
            "not permitted"  # May get connection error instead
        ])


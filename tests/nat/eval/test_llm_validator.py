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
from unittest.mock import AsyncMock, MagicMock, patch

from nat.data_models.config import Config
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.eval.llm_validator import validate_llm_endpoints, _is_404_error


class TestLLMEndpointValidation:
    """Tests for LLM endpoint validation functionality using WorkflowBuilder."""

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
    def config_with_nim_llm(self):
        """Create config with NIM LLM."""
        config = Config()
        config.llms = {
            "nim_llm": NIMModelConfig(
                model_name="meta/llama-3.1-8b-instruct",
                base_url="http://localhost:8000/v1"
            )
        }
        return config

    @pytest.fixture
    def config_with_bedrock_llm(self):
        """Create config with AWS Bedrock LLM."""
        config = Config()
        config.llms = {
            "bedrock_llm": AWSBedrockModelConfig(
                model_name="anthropic.claude-v2",
                region_name="us-east-1"
            )
        }
        return config

    @pytest.fixture
    def config_with_multiple_llms(self):
        """Create config with multiple LLMs of different types."""
        config = Config()
        config.llms = {
            "openai_llm": OpenAIModelConfig(
                model_name="gpt-4",
                base_url="http://localhost:8000/v1"
            ),
            "nim_llm": NIMModelConfig(
                model_name="llama-3.1-8b-instruct",
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

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_succeeds_with_accessible_endpoint(self, mock_builder_class, config_with_openai_llm):
        """Test that validation succeeds when LLM endpoint is accessible."""
        # Mock the builder and LLM
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value="test response")
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        # Should not raise any error
        await validate_llm_endpoints(config_with_openai_llm)
        
        # Verify builder was used correctly
        mock_builder.add_llm.assert_called_once()
        mock_builder.get_llm.assert_called_once()
        mock_llm.ainvoke.assert_called_once()

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_detects_404_error(self, mock_builder_class, config_with_openai_llm):
        """Test that validation detects 404 errors when model doesn't exist."""
        # Mock 404 error from ainvoke
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        
        # Simulate NotFoundError (404) - create actual NotFoundError class
        class NotFoundError(Exception):
            pass
        
        error_404 = NotFoundError("404: Model not found")
        mock_llm.ainvoke = AsyncMock(side_effect=error_404)
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config_with_openai_llm)

        error_msg = str(exc_info.value)
        assert "404" in error_msg
        assert "not found" in error_msg.lower()
        assert "ACTION REQUIRED" in error_msg

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_handles_auth_errors_gracefully(self, mock_builder_class, config_with_openai_llm):
        """Test that validation warns but continues on auth errors (not 404s)."""
        # Mock auth error from ainvoke
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("401: Unauthorized"))
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        # Should not raise RuntimeError for non-404 errors
        # (just logs warning)
        await validate_llm_endpoints(config_with_openai_llm)

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_works_for_nim_llm(self, mock_builder_class, config_with_nim_llm):
        """Test that validation works for NIM LLM type."""
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value="test response")
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        # Should validate NIM LLMs (not skip them)
        await validate_llm_endpoints(config_with_nim_llm)
        
        mock_builder.add_llm.assert_called_once()
        mock_llm.ainvoke.assert_called_once()

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_works_for_bedrock_llm(self, mock_builder_class, config_with_bedrock_llm):
        """Test that validation works for AWS Bedrock LLM type (framework-agnostic)."""
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value="test response")
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        # Should validate Bedrock LLMs (framework-agnostic approach)
        await validate_llm_endpoints(config_with_bedrock_llm)
        
        mock_builder.add_llm.assert_called_once()
        mock_llm.ainvoke.assert_called_once()

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_with_multiple_llms(self, mock_builder_class, config_with_multiple_llms):
        """Test that validation checks all configured LLMs."""
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value="test response")
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        await validate_llm_endpoints(config_with_multiple_llms)
        
        # Should have validated both LLMs
        assert mock_builder.add_llm.call_count == 2
        assert mock_builder.get_llm.call_count == 2
        assert mock_llm.ainvoke.call_count == 2

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_collects_all_404_errors(self, mock_builder_class, config_with_multiple_llms):
        """Test that validation collects all 404 errors before failing."""
        # Create actual NotFoundError class
        class NotFoundError(Exception):
            pass
        
        mock_builder = AsyncMock()
        
        # First LLM succeeds, second LLM has 404
        mock_llm_success = AsyncMock()
        mock_llm_success.ainvoke = AsyncMock(return_value="ok")
        
        mock_llm_404 = AsyncMock()
        error_404 = NotFoundError("404: Model not found")
        mock_llm_404.ainvoke = AsyncMock(side_effect=error_404)
        
        # Return different LLMs for different calls
        mock_builder.get_llm = AsyncMock(side_effect=[mock_llm_success, mock_llm_404])
        mock_builder.add_llm = AsyncMock()
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config_with_multiple_llms)

        error_msg = str(exc_info.value)
        # Should mention the failing LLM
        assert "nim_llm" in error_msg or "404" in error_msg


class Test404ErrorDetection:
    """Tests for the _is_404_error helper function."""

    def test_detects_notfounderror_type(self):
        """Test detection of NotFoundError exception type."""
        class NotFoundError(Exception):
            pass
        
        error = NotFoundError("Model not found")
        assert _is_404_error(error)

    def test_detects_404_in_message(self):
        """Test detection of 404 in error message."""
        error = Exception("HTTP 404: Model not found")
        assert _is_404_error(error)

    def test_detects_not_found_phrase(self):
        """Test detection of 'not found' phrase."""
        error = Exception("The model does not exist")
        assert _is_404_error(error)

    def test_does_not_detect_other_errors(self):
        """Test that non-404 errors are not detected as 404s."""
        auth_error = Exception("401: Unauthorized")
        rate_limit_error = Exception("429: Rate limit exceeded")
        
        assert not _is_404_error(auth_error)
        assert not _is_404_error(rate_limit_error)


class TestLLMValidationErrorMessages:
    """Tests for error message quality and actionability."""

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_error_message_includes_endpoint_details(self, mock_builder_class):
        """Test that error messages include specific endpoint details."""
        # Create actual NotFoundError class
        class NotFoundError(Exception):
            pass
        
        config = Config()
        config.llms = {
            "training_llm": OpenAIModelConfig(
                model_name="custom-model-name",
                base_url="http://custom-host:8000/v1"
            )
        }

        # Mock 404 error
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        error_404 = NotFoundError("404: Not found")
        mock_llm.ainvoke = AsyncMock(side_effect=error_404)
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config)

        error_msg = str(exc_info.value)
        # Should include the LLM name
        assert "training_llm" in error_msg
        # Should include the base URL
        assert "http://custom-host:8000/v1" in error_msg
        # Should include model name
        assert "custom-model-name" in error_msg

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_404_error_message_mentions_training_cancellation(self, mock_builder_class):
        """Test that 404 error message mentions potential training cancellation."""
        # Create actual NotFoundError class
        class NotFoundError(Exception):
            pass
        
        config = Config()
        config.llms = {
            "finetuned_model": NIMModelConfig(
                model_name="finetuned-llama",
                base_url="http://localhost:8000/v1"
            )
        }

        # Mock 404 error
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        error_404 = NotFoundError("404: Model not found")
        mock_llm.ainvoke = AsyncMock(side_effect=error_404)
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

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
        # Should include actionable guidance
        assert "ACTION REQUIRED" in error_msg


class TestLLMValidationIntegration:
    """Integration tests for LLM validation with evaluation flow."""

    @pytest.fixture
    def config_for_finetuned_model(self):
        """Create config simulating post-training scenario."""
        config = Config()
        config.llms = {
            "training_llm": NIMModelConfig(
                model_name="default/meta-llama-3.1-8b-instruct-nat-dpo",
                base_url="http://nim-endpoint:8000/v1"
            )
        }
        return config

    @patch("nat.eval.llm_validator.WorkflowBuilder")
    async def test_validation_scenario_after_canceled_training(self, mock_builder_class, config_for_finetuned_model):
        """
        Test validation behavior in the scenario that caused NVBug 5789819:
        Training was canceled, model never deployed, user tries to run eval.
        
        This should:
        1. Detect the missing model BEFORE eval starts (0/24 cases)
        2. Provide clear error about what went wrong
        3. Give actionable next steps
        """
        # Create actual NotFoundError class
        class NotFoundError(Exception):
            pass
        
        # Mock the exact bug scenario: endpoint is up but model doesn't exist (404)
        mock_builder = AsyncMock()
        mock_llm = AsyncMock()
        
        error_404 = NotFoundError("404: Model not found - the model default/meta-llama-3.1-8b-instruct-nat-dpo does not exist")
        mock_llm.ainvoke = AsyncMock(side_effect=error_404)
        
        mock_builder.add_llm = AsyncMock()
        mock_builder.get_llm = AsyncMock(return_value=mock_llm)
        mock_builder.__aenter__ = AsyncMock(return_value=mock_builder)
        mock_builder.__aexit__ = AsyncMock(return_value=None)
        
        mock_builder_class.return_value = mock_builder

        # Validation should fail with detailed error
        with pytest.raises(RuntimeError) as exc_info:
            await validate_llm_endpoints(config_for_finetuned_model)

        error_msg = str(exc_info.value)

        # Validation should catch the issue BEFORE eval starts
        assert any(check in error_msg for check in [
            "LLM endpoint validation failed",
            "not found",
            "404"
        ])
        
        # Should mention training-related causes
        assert any(phrase in error_msg.lower() for phrase in [
            "training",
            "canceled",
            "deployed"
        ])

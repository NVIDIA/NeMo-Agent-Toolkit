# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pydantic import ConfigDict

from aiq.data_models.llm import LLMBaseConfig
from aiq.guardrails.providers.nemo.config import NemoGuardrailsConfig
from aiq.guardrails.providers.nemo.provider import NemoGuardrailsProvider

# Check if NeMo Guardrails is available for integration tests
try:
    import nemoguardrails  # noqa: F401
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False


class MockLLMConfig(LLMBaseConfig, name="mock_llm"):
    """Mock LLM config for testing."""
    model_config = ConfigDict(extra="allow")  # Allow extra fields for testing

    model_name: str = "mock-model"

    def static_type(self):
        """Override static_type method for testing."""
        return "openai"


class TestNemoGuardrailsProvider:
    """Test the NeMo Guardrails provider."""

    def test_provider_creation(self):
        """Test provider can be created with valid config."""
        config = NemoGuardrailsConfig(llm_name="test_llm")
        provider = NemoGuardrailsProvider(config)

        assert provider.config == config
        assert provider.llm_config is None
        assert not provider.is_initialized

    def test_provider_creation_with_llm_config(self):
        """Test provider creation with LLM config."""
        config = NemoGuardrailsConfig(llm_name="test_llm")
        llm_config = MockLLMConfig()
        provider = NemoGuardrailsProvider(config, llm_config)

        assert provider.config == config
        assert provider.llm_config == llm_config

    def test_provider_creation_invalid_config(self):
        """Test provider creation with invalid config type."""
        with pytest.raises(ValueError, match="Config must be NemoGuardrailsConfig"):
            NemoGuardrailsProvider("invalid_config")

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not installed")
    async def test_initialize_with_nemo_available(self):
        """Test initialization when NeMo Guardrails is available."""
        config = NemoGuardrailsConfig(rails={"input": {"flows": ["self check input"]}})
        provider = NemoGuardrailsProvider(config)

        # This should not raise an exception
        await provider.initialize()
        assert provider.is_initialized

    async def test_initialize_without_nemo_available(self):
        """Test initialization when NeMo Guardrails is not available."""
        config = NemoGuardrailsConfig()
        provider = NemoGuardrailsProvider(config)

        # Mock the import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'nemoguardrails'")):
            with pytest.raises(ImportError, match="No module named 'nemoguardrails'"):
                await provider.initialize()

    async def test_initialize_with_fallback_on_error_import_fails(self):
        """Test initialization with fallback enabled when import fails at module level."""
        config = NemoGuardrailsConfig(fallback_on_error=True)
        provider = NemoGuardrailsProvider(config)

        # Mock the import to fail - this will still raise ImportError because
        # fallback_on_error only handles errors after successful import
        with patch('builtins.__import__', side_effect=ImportError("No module named 'nemoguardrails'")):
            # The provider initialize method catches ImportError and reraises it
            # fallback_on_error is meant for runtime errors, not import errors
            with pytest.raises(ImportError):
                await provider.initialize()

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not installed")
    async def test_initialize_with_fallback_on_error_runtime_fails(self):
        """Test initialization with fallback enabled when runtime initialization fails."""
        config = NemoGuardrailsConfig(fallback_on_error=True)
        provider = NemoGuardrailsProvider(config)

        # Mock LLMRails constructor to fail after import succeeds
        with patch('nemoguardrails.LLMRails', side_effect=RuntimeError("Runtime initialization failed")):
            # Should not raise exception, but mark as initialized with rails=None
            await provider.initialize()
            assert provider.is_initialized
            assert provider._rails is None

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not installed")
    async def test_apply_input_guardrails_integration(self):
        """Integration test for input guardrails when NeMo is available."""
        config = NemoGuardrailsConfig(rails={"input": {"flows": ["self check input"]}})
        provider = NemoGuardrailsProvider(config)
        await provider.initialize()

        # Test with simple string input
        input_data = "Hello, how are you?"
        result, should_continue = await provider.apply_input_guardrails(input_data)

        # Should return some result and boolean
        assert isinstance(result, str)
        assert isinstance(should_continue, bool)

    async def test_apply_input_guardrails_disabled(self):
        """Test input guardrails when disabled."""
        config = NemoGuardrailsConfig(enabled=False)
        provider = NemoGuardrailsProvider(config)

        input_data = "Test input"
        result, should_continue = await provider.apply_input_guardrails(input_data)

        # Should pass through unchanged
        assert result == input_data
        assert should_continue is True

    async def test_apply_input_guardrails_no_rails(self):
        """Test input guardrails when rails not initialized."""
        config = NemoGuardrailsConfig()
        provider = NemoGuardrailsProvider(config)
        # Don't initialize - _rails stays None

        input_data = "Test input"
        result, should_continue = await provider.apply_input_guardrails(input_data)

        # Should pass through unchanged
        assert result == input_data
        assert should_continue is True

    async def test_apply_output_guardrails_disabled(self):
        """Test output guardrails when disabled."""
        config = NemoGuardrailsConfig(output_rails_enabled=False)
        provider = NemoGuardrailsProvider(config)

        output_data = "Test output"
        result = await provider.apply_output_guardrails(output_data)

        # Should pass through unchanged
        assert result == output_data

    def test_create_fallback_response_string(self):
        """Test fallback response creation for string input."""
        config = NemoGuardrailsConfig(fallback_response="Custom fallback")
        provider = NemoGuardrailsProvider(config)

        result = provider.create_fallback_response("test input")
        assert result == "Custom fallback"

    def test_create_fallback_response_default(self):
        """Test fallback response creation with default message."""
        config = NemoGuardrailsConfig()
        provider = NemoGuardrailsProvider(config)

        result = provider.create_fallback_response("test input")
        assert result == "I cannot provide a response to that request."

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not installed")
    def test_convert_llm_config_openai(self):
        """Test LLM config conversion for OpenAI."""
        config = NemoGuardrailsConfig()
        provider = NemoGuardrailsProvider(config)

        # Create mock OpenAI config with proper fields
        openai_config = MockLLMConfig(model_name="gpt-4", api_key="test-key", temperature=0.7)

        result = provider._convert_llm_config(openai_config)

        assert result["engine"] == "openai"
        assert result["model"] == "gpt-4"
        assert result["parameters"]["api_key"] == "test-key"
        assert result["parameters"]["temperature"] == 0.7

    def test_get_flows_from_config(self):
        """Test extracting flows from rails configuration."""
        config = NemoGuardrailsConfig(rails={
            "input": {
                "flows": ["self check input", "check jailbreak"]
            }, "output": {
                "flows": ["self check output"]
            }
        })
        provider = NemoGuardrailsProvider(config)

        input_flows = provider._get_flows_from_config(config, "input")
        output_flows = provider._get_flows_from_config(config, "output")

        assert input_flows == ["self check input", "check jailbreak"]
        assert output_flows == ["self check output"]

    def test_get_flows_from_config_empty(self):
        """Test extracting flows when config is empty."""
        config = NemoGuardrailsConfig()
        provider = NemoGuardrailsProvider(config)

        flows = provider._get_flows_from_config(config, "input")
        assert flows == []


# Integration tests that require NeMo Guardrails to be installed
@pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not installed")
class TestNemoGuardrailsIntegration:
    """Integration tests that require NeMo Guardrails to be installed."""

    async def test_full_workflow_basic(self):
        """Test a complete workflow with actual NeMo Guardrails."""
        config = NemoGuardrailsConfig(rails={
            "input": {
                "flows": ["self check input"]
            }, "output": {
                "flows": ["self check output"]
            }
        })
        provider = NemoGuardrailsProvider(config)
        await provider.initialize()

        # Test input guardrails
        input_result, allowed = await provider.apply_input_guardrails("Hello there!")
        assert isinstance(input_result, str)
        assert isinstance(allowed, bool)

        # Test output guardrails
        output_result = await provider.apply_output_guardrails("Hello back!", "Hello there!")
        assert isinstance(output_result, str)

        # Test fallback response
        fallback = provider.create_fallback_response("blocked input")
        assert isinstance(fallback, str)

    async def test_error_handling_in_rails(self):
        """Test error handling when guardrails encounter errors."""
        config = NemoGuardrailsConfig(fallback_on_error=True)
        provider = NemoGuardrailsProvider(config)

        # Initialize with minimal config that might cause issues
        try:
            await provider.initialize()

            # Even if initialization succeeds, test error handling in application
            result, allowed = await provider.apply_input_guardrails("test")
            assert isinstance(result, str)
            assert isinstance(allowed, bool)

        except Exception:
            # If initialization fails with fallback_on_error=True,
            # it should still mark as initialized
            assert provider.is_initialized

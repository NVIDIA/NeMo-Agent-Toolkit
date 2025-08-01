# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from aiq.data_models.llm import LLMBaseConfig
from aiq.guardrails.interface import GuardrailsProviderFactory
from aiq.guardrails.manager import GuardrailsManager
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
    model_name: str = "mock-model"


class TestGuardrailsIntegration:
    """Integration tests for the complete guardrails system."""

    def setup_method(self):
        """Setup for each test."""
        # Clear factory state
        GuardrailsProviderFactory._providers.clear()
        # Register the NeMo provider
        GuardrailsProviderFactory.register_provider(NemoGuardrailsConfig, NemoGuardrailsProvider)

    def test_factory_provider_creation(self):
        """Test that factory correctly creates providers."""
        config = NemoGuardrailsConfig(llm_name="test_llm")
        llm_config = MockLLMConfig()

        # Create provider through factory
        provider = GuardrailsProviderFactory.create_provider(config, llm_config)

        assert isinstance(provider, NemoGuardrailsProvider)
        assert provider.config == config
        assert provider.llm_config == llm_config

    async def test_manager_creation_and_basic_properties(self):
        """Test manager creation and basic property access."""
        config = NemoGuardrailsConfig(llm_name="test_llm")
        llm_config = MockLLMConfig()
        manager = GuardrailsManager(config, llm_config)

        # Check initial state
        assert manager.config == config
        assert manager.llm_config == llm_config
        assert manager._provider is None

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not available")
    async def test_full_workflow_with_nemo_available(self):
        """Test full workflow when NeMo Guardrails is available."""
        config = NemoGuardrailsConfig(rails={"input": {"flows": ["self check input"]}})
        llm_config = MockLLMConfig()
        manager = GuardrailsManager(config, llm_config)

        # Initialize manager (should create and initialize provider)
        await manager.initialize()

        # Verify provider was created
        assert manager._provider is not None
        assert isinstance(manager._provider, NemoGuardrailsProvider)
        assert manager._provider.config == config
        assert manager._provider.llm_config == llm_config

        # Test basic functionality
        input_result, allowed = await manager.apply_input_guardrails("Hello there!")
        assert isinstance(input_result, str)
        assert isinstance(allowed, bool)

        output_result = await manager.apply_output_guardrails("Hello back!")
        assert isinstance(output_result, str)

        # Test fallback response
        fallback = manager.create_fallback_response("blocked_input")
        assert fallback == "I cannot provide a response to that request."

    async def test_error_handling_with_fallback(self):
        """Test error handling with fallback enabled when imports fail."""
        config = NemoGuardrailsConfig(fallback_on_error=True)
        manager = GuardrailsManager(config)

        # Mock import failure during provider initialization
        with patch('aiq.guardrails.providers.nemo.provider.NemoGuardrailsProvider.initialize',
                   side_effect=ImportError("No module named 'nemoguardrails'")):
            # Should not raise exception during initialization
            await manager.initialize()

            # Provider should be None due to initialization failure with fallback
            assert manager._provider is None

            # Operations should pass through using manager's fallback behavior
            input_result, allowed = await manager.apply_input_guardrails("test_input")
            assert input_result == "test_input"
            assert allowed is True

            output_result = await manager.apply_output_guardrails("test_output")
            assert output_result == "test_output"

            # Fallback should use config default
            fallback = manager.create_fallback_response("test")
            assert fallback == "I cannot provide a response to that request."

    async def test_error_handling_without_fallback(self):
        """Test error handling without fallback when imports fail."""
        config = NemoGuardrailsConfig(fallback_on_error=False)
        manager = GuardrailsManager(config)

        # Mock import failure during provider initialization
        with patch('aiq.guardrails.providers.nemo.provider.NemoGuardrailsProvider.initialize',
                   side_effect=ImportError("No module named 'nemoguardrails'")):
            # Should raise exception during initialization
            with pytest.raises(ImportError, match="No module named 'nemoguardrails'"):
                await manager.initialize()

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not available")
    async def test_multiple_managers_independent(self):
        """Test that multiple managers work independently."""
        config1 = NemoGuardrailsConfig(llm_name="llm1", fallback_response="Fallback 1")
        config2 = NemoGuardrailsConfig(llm_name="llm2", fallback_response="Fallback 2")

        manager1 = GuardrailsManager(config1)
        manager2 = GuardrailsManager(config2)

        # Initialize both
        await manager1.initialize()
        await manager2.initialize()

        # Should have different provider instances
        assert manager1._provider is not manager2._provider

        # Check configurations are different
        assert manager1._provider.config.llm_name == "llm1"
        assert manager2._provider.config.llm_name == "llm2"

        # Check fallback responses are different
        fallback1 = manager1.create_fallback_response("test")
        fallback2 = manager2.create_fallback_response("test")

        assert fallback1 == "Fallback 1"
        assert fallback2 == "Fallback 2"

    def test_configuration_validation(self):
        """Test that configuration validation works."""
        # Valid config should work
        valid_config = NemoGuardrailsConfig(llm_name="test_llm")
        manager = GuardrailsManager(valid_config)

        # Should not raise during manager creation
        assert manager.config == valid_config

        # Config with all optional fields should work
        full_config = NemoGuardrailsConfig(enabled=True,
                                           input_rails_enabled=False,
                                           output_rails_enabled=True,
                                           config_path="/path/to/config",
                                           llm_name="test_llm",
                                           fallback_response="Custom message",
                                           fallback_on_error=False,
                                           verbose=True,
                                           max_retries=5,
                                           timeout_seconds=60.0,
                                           rails={"input": {
                                               "flows": ["self check input"]
                                           }})
        manager_full = GuardrailsManager(full_config)
        assert manager_full.config == full_config

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not available")
    async def test_custom_configuration_options(self):
        """Test that custom configuration options are respected."""
        # Use valid flow names that exist in NeMo Guardrails
        config = NemoGuardrailsConfig(
            enabled=True,
            input_rails_enabled=False,
            output_rails_enabled=True,
            config_path=None,  # Use None for generated config
            llm_name="custom_llm",
            fallback_response="Custom fallback message",
            fallback_on_error=False,
            verbose=True,
            max_retries=5,
            timeout_seconds=60.0,
            rails={"input": {
                "flows": ["self check input"]
            }}  # Use valid flow name
        )

        manager = GuardrailsManager(config)
        await manager.initialize()

        # Verify configuration was passed to provider
        provider = manager._provider
        assert provider.config.enabled is True
        assert provider.config.input_rails_enabled is False
        assert provider.config.output_rails_enabled is True
        assert provider.config.llm_name == "custom_llm"
        assert provider.config.fallback_response == "Custom fallback message"
        assert provider.config.fallback_on_error is False
        assert provider.config.verbose is True
        assert provider.config.max_retries == 5
        assert provider.config.timeout_seconds == 60.0
        assert provider.config.rails == {"input": {"flows": ["self check input"]}}

        # Test custom fallback message
        fallback = manager.create_fallback_response("test")
        assert fallback == "Custom fallback message"

    async def test_disabled_guardrails_behavior(self):
        """Test behavior when guardrails are disabled."""
        config = NemoGuardrailsConfig(enabled=False)
        manager = GuardrailsManager(config)

        # Even with disabled guardrails, manager should initialize
        await manager.initialize()

        # Operations should pass through unchanged
        input_result, allowed = await manager.apply_input_guardrails("test input")
        assert input_result == "test input"
        assert allowed is True

        output_result = await manager.apply_output_guardrails("test output")
        assert output_result == "test output"

    async def test_uninitialized_manager_behavior(self):
        """Test behavior when manager is not initialized."""
        config = NemoGuardrailsConfig()
        manager = GuardrailsManager(config)

        # Should raise RuntimeError for operations before initialization
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await manager.apply_input_guardrails("test")

        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await manager.apply_output_guardrails("test")

        with pytest.raises(RuntimeError, match="Provider not initialized"):
            manager.create_fallback_response("test")

    def test_factory_unregistered_provider(self):
        """Test factory behavior with unregistered provider type."""
        # Clear factory state to simulate unregistered provider
        GuardrailsProviderFactory._providers.clear()

        config = NemoGuardrailsConfig()
        with pytest.raises(ValueError, match="No provider registered for config type"):
            GuardrailsProviderFactory.create_provider(config)

    @pytest.mark.skipif(not NEMO_AVAILABLE, reason="NeMo Guardrails not available")
    async def test_llm_config_integration(self):
        """Test integration with LLM configuration."""
        llm_config = MockLLMConfig(model_name="test-model")
        config = NemoGuardrailsConfig(llm_name="test_llm")

        manager = GuardrailsManager(config, llm_config)
        await manager.initialize()

        # Verify LLM config was passed to provider
        assert manager._provider.llm_config == llm_config
        assert manager._provider.llm_config.model_name == "test-model"

    def test_config_serialization_compatibility(self):
        """Test that configs can be serialized and deserialized."""
        original_config = NemoGuardrailsConfig(llm_name="test_llm",
                                               fallback_response="Test message",
                                               enabled=True,
                                               rails={"input": {
                                                   "flows": ["test flow"]
                                               }})

        # Serialize to dict
        config_dict = original_config.model_dump()

        # Recreate from dict
        recreated_config = NemoGuardrailsConfig(**config_dict)

        # Should be equivalent
        assert recreated_config.llm_name == original_config.llm_name
        assert recreated_config.fallback_response == original_config.fallback_response
        assert recreated_config.enabled == original_config.enabled
        assert recreated_config.rails == original_config.rails

    async def test_fallback_behavior_through_manager(self):
        """Test fallback behavior is properly handled through manager."""
        config = NemoGuardrailsConfig(fallback_on_error=True, fallback_response="Custom fallback")
        manager = GuardrailsManager(config)

        # Mock provider initialization to fail
        with patch('aiq.guardrails.providers.nemo.provider.NemoGuardrailsProvider.initialize',
                   side_effect=ImportError("Mock failure")):
            await manager.initialize()

            # Manager should handle the fallback by setting provider to None
            assert manager._provider is None

            # Operations should use manager's own fallback logic
            fallback = manager.create_fallback_response("test")
            assert fallback == "Custom fallback"

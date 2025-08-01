# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from aiq.data_models.guardrails import GuardrailsBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.guardrails.interface import GuardrailsProvider
from aiq.guardrails.interface import GuardrailsProviderFactory
from aiq.guardrails.manager import GuardrailsManager


class MockGuardrailsConfig(GuardrailsBaseConfig, name="mock_guardrails"):
    """Mock config for testing."""
    test_param: str = "test_value"
    fallback_response: str = "test_value"
    fallback_on_error: bool = False


class MockGuardrailsProvider(GuardrailsProvider):
    """Mock provider for testing manager delegation."""

    def __init__(self, config: MockGuardrailsConfig, llm_config: LLMBaseConfig | None = None):
        self.config = config
        self.llm_config = llm_config
        self.initialized = False
        self.initialize_called = False
        self.input_calls = []
        self.output_calls = []
        self.fallback_calls = []

    async def initialize(self):
        """Mock initialization with tracking."""
        self.initialize_called = True
        self.initialized = True

    async def apply_input_guardrails(self, input_data: Any) -> Tuple[Any, bool]:
        """Mock input guardrails with tracking."""
        self.input_calls.append(input_data)
        return f"provider_input_{input_data}", True

    async def apply_output_guardrails(self, output_data: Any, input_data: Any = None) -> Any:
        """Mock output guardrails with tracking."""
        self.output_calls.append((output_data, input_data))
        return f"provider_output_{output_data}"

    def create_fallback_response(self, input_data: Any) -> Any:
        """Mock fallback response with tracking."""
        self.fallback_calls.append(input_data)
        return f"provider_fallback_{input_data}"


class TestGuardrailsManager:
    """Test the GuardrailsManager delegation pattern."""

    def setup_method(self):
        """Reset factory state before each test."""
        GuardrailsProviderFactory._providers.clear()
        # Register our mock provider
        GuardrailsProviderFactory.register_provider(MockGuardrailsConfig, MockGuardrailsProvider)

    def test_manager_initialization(self):
        """Test GuardrailsManager initialization."""
        config = MockGuardrailsConfig()
        llm_config = MagicMock(spec=LLMBaseConfig)

        manager = GuardrailsManager(config, llm_config)

        assert manager.config == config
        assert manager.llm_config == llm_config
        assert manager._provider is None  # Not initialized yet
        assert not manager._initialized  # Not initialized yet

    async def test_manager_initialize_creates_provider(self):
        """Test that manager initialization creates and initializes the provider."""
        config = MockGuardrailsConfig()
        llm_config = MagicMock(spec=LLMBaseConfig)

        manager = GuardrailsManager(config, llm_config)
        await manager.initialize()

        # Check that provider was created and initialized
        assert manager._provider is not None
        assert isinstance(manager._provider, MockGuardrailsProvider)
        assert manager._provider.config == config
        assert manager._provider.llm_config == llm_config
        assert manager._provider.initialize_called
        assert manager._provider.initialized
        assert manager._initialized

    async def test_manager_initialize_without_llm_config(self):
        """Test manager initialization without LLM config."""
        config = MockGuardrailsConfig()

        manager = GuardrailsManager(config)
        await manager.initialize()

        assert manager._provider is not None
        assert manager._provider.config == config
        assert manager._provider.llm_config is None
        assert manager._provider.initialized
        assert manager._initialized

    async def test_apply_input_guardrails_delegates(self):
        """Test that apply_input_guardrails delegates to provider."""
        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)
        await manager.initialize()

        # Call input guardrails
        result, allowed = await manager.apply_input_guardrails("test_input")

        # Check delegation occurred
        assert result == "provider_input_test_input"
        assert allowed is True
        assert manager._provider.input_calls == ["test_input"]

    async def test_apply_output_guardrails_delegates(self):
        """Test that apply_output_guardrails delegates to provider."""
        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)
        await manager.initialize()

        # Call output guardrails
        result = await manager.apply_output_guardrails("test_output", "test_input")

        # Check delegation occurred
        assert result == "provider_output_test_output"
        assert manager._provider.output_calls == [("test_output", "test_input")]

    async def test_apply_output_guardrails_without_input_data(self):
        """Test output guardrails without input data."""
        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)
        await manager.initialize()

        # Call output guardrails without input data
        result = await manager.apply_output_guardrails("test_output")

        # Check delegation occurred
        assert result == "provider_output_test_output"
        assert manager._provider.output_calls == [("test_output", None)]

    async def test_create_fallback_response_delegates(self):
        """Test that create_fallback_response delegates to provider."""
        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)
        await manager.initialize()

        # Call fallback response
        result = manager.create_fallback_response("test_input")

        # Check delegation occurred
        assert result == "provider_fallback_test_input"
        assert manager._provider.fallback_calls == ["test_input"]

    async def test_provider_not_initialized_error(self):
        """Test that calling methods before initialization raises error."""
        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)

        # Try to call methods before initialization
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await manager.apply_input_guardrails("test")

        with pytest.raises(RuntimeError, match="Provider not initialized"):
            await manager.apply_output_guardrails("test")

        with pytest.raises(RuntimeError, match="Provider not initialized"):
            manager.create_fallback_response("test")

    async def test_unregistered_config_type_error(self):
        """Test that unregistered config type raises error during initialization."""
        # Clear the registered provider
        GuardrailsProviderFactory._providers.clear()

        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)

        with pytest.raises(ValueError, match="No provider registered for config type"):
            await manager.initialize()

    async def test_multiple_calls_same_provider(self):
        """Test that multiple calls use the same provider instance."""
        config = MockGuardrailsConfig()
        manager = GuardrailsManager(config)
        await manager.initialize()

        provider1 = manager._provider

        # Call multiple methods
        await manager.apply_input_guardrails("input1")
        await manager.apply_output_guardrails("output1")
        manager.create_fallback_response("fallback1")

        await manager.apply_input_guardrails("input2")
        await manager.apply_output_guardrails("output2")
        manager.create_fallback_response("fallback2")

        provider2 = manager._provider

        # Should be the same provider instance
        assert provider1 is provider2

        # Check all calls were tracked
        assert provider1.input_calls == ["input1", "input2"]
        assert len(provider1.output_calls) == 2
        assert provider1.fallback_calls == ["fallback1", "fallback2"]

    async def test_manager_with_different_configs(self):
        """Test that different managers with different configs work independently."""
        config1 = MockGuardrailsConfig(test_param="value1")
        config2 = MockGuardrailsConfig(test_param="value2")

        manager1 = GuardrailsManager(config1)
        manager2 = GuardrailsManager(config2)

        await manager1.initialize()
        await manager2.initialize()

        # Should have different provider instances
        assert manager1._provider is not manager2._provider
        assert manager1._provider.config.test_param == "value1"
        assert manager2._provider.config.test_param == "value2"

        # Calls should be independent
        await manager1.apply_input_guardrails("manager1_input")
        await manager2.apply_input_guardrails("manager2_input")

        assert manager1._provider.input_calls == ["manager1_input"]
        assert manager2._provider.input_calls == ["manager2_input"]

    async def test_fallback_behavior_when_provider_fails(self):
        """Test fallback behavior when provider initialization fails but fallback_on_error=True."""
        config = MockGuardrailsConfig(fallback_on_error=True)
        manager = GuardrailsManager(config)

        # Clear providers to force failure
        GuardrailsProviderFactory._providers.clear()

        # Should not raise, but set provider to None
        await manager.initialize()
        assert manager._initialized
        assert manager._provider is None

        # Operations should use manager's fallback behavior
        input_result, allowed = await manager.apply_input_guardrails("test")
        assert input_result == "test"
        assert allowed is True

        output_result = await manager.apply_output_guardrails("test")
        assert output_result == "test"

        fallback = manager.create_fallback_response("test")
        assert fallback == "test_value"  # Uses config's fallback_response if set

    async def test_error_propagation_when_fallback_disabled(self):
        """Test that errors propagate when fallback_on_error=False."""
        config = MockGuardrailsConfig(fallback_on_error=False)
        manager = GuardrailsManager(config)

        # Clear providers to force failure
        GuardrailsProviderFactory._providers.clear()

        # Should raise the original error
        with pytest.raises(ValueError, match="No provider registered"):
            await manager.initialize()

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


class MockGuardrailsConfig(GuardrailsBaseConfig, name="mock_guardrails"):
    """Mock config for testing."""
    test_param: str = "test_value"


class MockGuardrailsProvider(GuardrailsProvider):
    """Mock provider for testing."""

    def __init__(self, config: MockGuardrailsConfig, llm_config: LLMBaseConfig | None = None):
        self.config = config
        self.llm_config = llm_config
        self.initialized = False

    async def initialize(self):
        """Mock initialization."""
        self.initialized = True

    async def apply_input_guardrails(self, input_data: Any) -> Tuple[Any, bool]:
        """Mock input guardrails that adds a prefix."""
        return f"guarded_{input_data}", True

    async def apply_output_guardrails(self, output_data: Any, input_data: Any = None) -> Any:
        """Mock output guardrails that adds a suffix."""
        return f"{output_data}_guarded"

    def create_fallback_response(self, input_data: Any) -> Any:
        """Mock fallback response."""
        return f"fallback_for_{input_data}"


class TestGuardrailsProviderInterface:
    """Test the GuardrailsProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that GuardrailsProvider cannot be instantiated directly."""
        config = MockGuardrailsConfig()

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GuardrailsProvider(config)

    def test_abstract_methods_exist(self):
        """Test that all required abstract methods exist."""
        abstract_methods = GuardrailsProvider.__abstractmethods__
        expected_methods = {
            'initialize', 'apply_input_guardrails', 'apply_output_guardrails', 'create_fallback_response'
        }
        assert abstract_methods == expected_methods

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""
        config = MockGuardrailsConfig()
        provider = MockGuardrailsProvider(config)

        assert provider.config == config
        assert provider.llm_config is None
        assert not provider.initialized

    async def test_concrete_implementation_methods(self):
        """Test that concrete implementation methods work."""
        config = MockGuardrailsConfig()
        provider = MockGuardrailsProvider(config)

        # Test initialization
        await provider.initialize()
        assert provider.initialized

        # Test input guardrails
        input_result, allowed = await provider.apply_input_guardrails("test_input")
        assert input_result == "guarded_test_input"
        assert allowed is True

        # Test output guardrails
        output_result = await provider.apply_output_guardrails("test_output", "test_input")
        assert output_result == "test_output_guarded"

        # Test fallback response
        fallback = provider.create_fallback_response("test_input")
        assert fallback == "fallback_for_test_input"


class TestGuardrailsProviderFactory:
    """Test the GuardrailsProviderFactory."""

    def setup_method(self):
        """Reset factory state before each test."""
        GuardrailsProviderFactory._providers.clear()

    def test_register_provider(self):
        """Test registering a provider."""
        GuardrailsProviderFactory.register_provider(MockGuardrailsConfig, MockGuardrailsProvider)

        assert MockGuardrailsConfig in GuardrailsProviderFactory._providers
        assert GuardrailsProviderFactory._providers[MockGuardrailsConfig] == MockGuardrailsProvider

    def test_create_provider_success(self):
        """Test creating a provider successfully."""
        # Register the provider
        GuardrailsProviderFactory.register_provider(MockGuardrailsConfig, MockGuardrailsProvider)

        # Create config and provider
        config = MockGuardrailsConfig()
        llm_config = MagicMock(spec=LLMBaseConfig)

        provider = GuardrailsProviderFactory.create_provider(config, llm_config)

        assert isinstance(provider, MockGuardrailsProvider)
        assert provider.config == config
        assert provider.llm_config == llm_config

    def test_create_provider_unregistered_config(self):
        """Test creating a provider with unregistered config type."""
        config = MockGuardrailsConfig()

        with pytest.raises(ValueError, match="No provider registered for config type"):
            GuardrailsProviderFactory.create_provider(config)

    def test_create_provider_without_llm_config(self):
        """Test creating a provider without LLM config."""
        GuardrailsProviderFactory.register_provider(MockGuardrailsConfig, MockGuardrailsProvider)

        config = MockGuardrailsConfig()
        provider = GuardrailsProviderFactory.create_provider(config)

        assert isinstance(provider, MockGuardrailsProvider)
        assert provider.config == config
        assert provider.llm_config is None

    def test_multiple_provider_registration(self):
        """Test registering multiple providers."""

        class AnotherMockConfig(GuardrailsBaseConfig, name="another_mock"):
            pass

        class AnotherMockProvider(GuardrailsProvider):

            def __init__(self, config, llm_config=None):
                self.config = config
                self.llm_config = llm_config

            async def initialize(self):
                pass

            async def apply_input_guardrails(self, input_data):
                return input_data, True

            async def apply_output_guardrails(self, output_data, input_data=None):
                return output_data

            def create_fallback_response(self, input_data):
                return "fallback"

        # Register both providers
        GuardrailsProviderFactory.register_provider(MockGuardrailsConfig, MockGuardrailsProvider)
        GuardrailsProviderFactory.register_provider(AnotherMockConfig, AnotherMockProvider)

        assert len(GuardrailsProviderFactory._providers) == 2

        # Test creating each provider
        config1 = MockGuardrailsConfig()
        provider1 = GuardrailsProviderFactory.create_provider(config1)
        assert isinstance(provider1, MockGuardrailsProvider)

        config2 = AnotherMockConfig()
        provider2 = GuardrailsProviderFactory.create_provider(config2)
        assert isinstance(provider2, AnotherMockProvider)

    def test_factory_is_singleton_like(self):
        """Test that factory maintains state across calls."""
        GuardrailsProviderFactory.register_provider(MockGuardrailsConfig, MockGuardrailsProvider)

        # Create multiple configs and providers
        config1 = MockGuardrailsConfig()
        config2 = MockGuardrailsConfig()

        provider1 = GuardrailsProviderFactory.create_provider(config1)
        provider2 = GuardrailsProviderFactory.create_provider(config2)

        # Both should be instances of the same provider class
        assert isinstance(provider1, MockGuardrailsProvider)
        assert isinstance(provider2, MockGuardrailsProvider)

        # But they should be different instances with different configs
        assert provider1 is not provider2
        assert provider1.config is config1
        assert provider2.config is config2

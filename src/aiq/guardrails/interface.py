# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base interfaces for guardrails providers."""

from abc import ABC
from abc import abstractmethod
from typing import Any

from aiq.data_models.guardrails import GuardrailsBaseConfig
from aiq.data_models.llm import LLMBaseConfig


class GuardrailsProvider(ABC):
    """Base interface for guardrails providers."""

    def __init__(self, config: GuardrailsBaseConfig, llm_config: LLMBaseConfig | None = None):
        self.config = config
        self.llm_config = llm_config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the guardrails provider."""
        pass

    @abstractmethod
    async def apply_input_guardrails(self, input_data: Any) -> tuple[Any, bool]:
        """
        Apply input guardrails to the input data.

        Returns:
            tuple: (processed_input, should_continue)
                - processed_input: The input after guardrails processing
                - should_continue: Whether to continue with function execution
        """
        pass

    @abstractmethod
    async def apply_output_guardrails(self, output_data: Any, input_data: Any = None) -> Any:
        """
        Apply output guardrails to the output data.

        Returns:
            The output after guardrails processing
        """
        pass

    @abstractmethod
    def create_fallback_response(self, input_data: Any) -> Any:
        """Create a fallback response when guardrails block execution."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if the provider has been initialized."""
        return self._initialized

    def _mark_initialized(self) -> None:
        """Mark the provider as initialized."""
        self._initialized = True


class GuardrailsProviderFactory:
    """Factory for creating guardrails providers."""

    _providers = {}

    @classmethod
    def register_provider(cls, config_type: type, provider_class: type):
        """Register a provider for a specific config type."""
        cls._providers[config_type] = provider_class

    @classmethod
    def create_provider(cls,
                        config: GuardrailsBaseConfig,
                        llm_config: LLMBaseConfig | None = None) -> GuardrailsProvider:
        """Create a provider instance based on config type."""
        config_type = type(config)
        if config_type not in cls._providers:
            raise ValueError(f"No provider registered for config type: {config_type.__name__}")

        provider_class = cls._providers[config_type]
        return provider_class(config, llm_config)

    @classmethod
    def get_registered_providers(cls) -> dict:
        """Get all registered providers."""
        return cls._providers.copy()

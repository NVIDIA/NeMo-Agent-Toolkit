# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any

from aiq.data_models.guardrails import GuardrailsBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.guardrails.interface import GuardrailsProvider
from aiq.guardrails.interface import GuardrailsProviderFactory

logger = logging.getLogger(__name__)


class GuardrailsManager:
    """Manager for different types of guardrails implementations using provider delegation."""

    def __init__(self, config: GuardrailsBaseConfig, llm_config: LLMBaseConfig | None = None):
        self.config = config
        self.llm_config = llm_config
        self._provider: GuardrailsProvider | None = None
        self._initialized = False

    async def initialize(self):
        """Initialize the guardrails provider based on configuration."""
        try:
            # Create provider using factory
            self._provider = GuardrailsProviderFactory.create_provider(self.config, self.llm_config)
            await self._provider.initialize()
            logger.info("Successfully initialized guardrails provider: %s", type(self._provider).__name__)
        except Exception as e:
            logger.error("Failed to initialize guardrails provider: %s", e)
            if getattr(self.config, 'fallback_on_error', False):
                logger.warning("Continuing without guardrails due to initialization error")
                self._provider = None
            else:
                raise
        finally:
            self._initialized = True

    async def apply_input_guardrails(self, input_data: Any) -> tuple[Any, bool]:
        """
        Apply input guardrails to the input data.

        Returns:
            tuple: (processed_input, should_continue)
                - processed_input: The input after guardrails processing
                - should_continue: Whether to continue with function execution
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if self._provider is None:
            # Fallback behavior - pass through when provider failed to initialize
            return input_data, True

        return await self._provider.apply_input_guardrails(input_data)

    async def apply_output_guardrails(self, output_data: Any, input_data: Any = None) -> Any:
        """
        Apply output guardrails to the output data.

        Returns:
            The output after guardrails processing
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if self._provider is None:
            # Fallback behavior - pass through when provider failed to initialize
            return output_data

        return await self._provider.apply_output_guardrails(output_data, input_data)

    def create_fallback_response(self, input_data: Any) -> Any:
        """Create a fallback response when guardrails block execution."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        if self._provider is None:
            # Fallback behavior - use manager's own fallback logic
            fallback_message = getattr(self.config, 'fallback_response', "I cannot provide a response to that request.")
            # Create response in the same format as expected output
            if hasattr(input_data, 'messages'):
                # AIQChatRequest input -> AIQChatResponse output
                from aiq.data_models.api_server import AIQChatResponse
                return AIQChatResponse.from_string(fallback_message)
            else:
                # String input -> String output
                return fallback_message

        return self._provider.create_fallback_response(input_data)

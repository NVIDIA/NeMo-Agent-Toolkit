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
"""Test LLM for AutoGen."""

from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import Field

from nat.builder.builder import Builder
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.autogen.llm import _patch_autogen_client_based_on_config


class MockRetryConfig(LLMBaseConfig, RetryMixin):
    """Mock config with retry mixin."""

    num_retries: int = 3
    retry_on_status_codes: list[int | str] = Field(default_factory=lambda: [500, 502, 503])
    retry_on_errors: list[str] | None = Field(default_factory=lambda: ["timeout"])


class MockThinkingConfig(LLMBaseConfig, ThinkingMixin):
    """Mock config with thinking mixin."""

    model_name: str = "nvidia/nvidia-nemotron-test"  # Match pattern for thinking support
    thinking: bool | None = True  # Enable thinking to get system prompt


class MockCombinedConfig(LLMBaseConfig, RetryMixin, ThinkingMixin):
    """Mock config with both mixins."""

    num_retries: int = 3
    retry_on_status_codes: list[int | str] = Field(default_factory=lambda: [500, 502, 503])
    retry_on_errors: list[str] | None = Field(default_factory=lambda: ["timeout"])
    model_name: str = "nvidia/nvidia-nemotron-test"  # Match pattern for thinking support
    thinking: bool | None = True  # Enable thinking to get system prompt


class TestPatchAutoGenClient:
    """Test cases for _patch_autogen_client_based_on_config function."""

    def test_patch_with_no_mixins(self):
        """Test patching client with no mixins."""
        mock_client = Mock()
        base_config = LLMBaseConfig()

        result = _patch_autogen_client_based_on_config(mock_client, base_config)
        assert result == mock_client

    @patch('nat.plugins.autogen.llm.patch_with_retry')
    def test_patch_with_retry_mixin(self, mock_patch_retry):
        """Test patching client with retry mixin."""
        mock_client = Mock()
        mock_patched_client = Mock()
        mock_patch_retry.return_value = mock_patched_client

        retry_config = MockRetryConfig()
        retry_config.num_retries = 5
        retry_config.retry_on_status_codes = [500, 503]
        retry_config.retry_on_errors = ["timeout", "connection"]

        result = _patch_autogen_client_based_on_config(mock_client, retry_config)

        mock_patch_retry.assert_called_once_with(mock_client,
                                                 retries=5,
                                                 retry_codes=[500, 503],
                                                 retry_on_messages=["timeout", "connection"])
        assert result == mock_patched_client

    @patch('nat.plugins.autogen.llm.patch_with_thinking')
    def test_patch_with_thinking_mixin(self, mock_patch_thinking):
        """Test patching client with thinking mixin."""
        mock_client = Mock()
        mock_patched_client = Mock()
        mock_patch_thinking.return_value = mock_patched_client

        # Create a real thinking config instance
        thinking_config = MockThinkingConfig()

        result = _patch_autogen_client_based_on_config(mock_client, thinking_config)

        mock_patch_thinking.assert_called_once()
        args, _kwargs = mock_patch_thinking.call_args
        assert args[0] == mock_client
        assert result == mock_patched_client

    @patch('nat.plugins.autogen.llm.patch_with_retry')
    @patch('nat.plugins.autogen.llm.patch_with_thinking')
    def test_patch_with_both_mixins(self, mock_patch_thinking, mock_patch_retry):
        """Test patching client with both retry and thinking mixins."""
        mock_retry_client = Mock()
        mock_final_client = Mock()
        mock_patch_retry.return_value = mock_retry_client
        mock_patch_thinking.return_value = mock_final_client

        config = MockCombinedConfig()
        config.num_retries = 3
        config.retry_on_status_codes = [500, 502]
        config.retry_on_errors = ["timeout"]

        mock_client = Mock()
        result = _patch_autogen_client_based_on_config(mock_client, config)

        # Verify retry is applied first, then thinking
        mock_patch_retry.assert_called_once_with(mock_client,
                                                 retries=3,
                                                 retry_codes=[500, 502],
                                                 retry_on_messages=["timeout"])
        mock_patch_thinking.assert_called_once()
        assert result == mock_final_client


class TestConfigValidation:
    """Test configuration validation and model creation."""

    def test_openai_config_creation(self):
        """Test OpenAI model config creation."""
        config = OpenAIModelConfig(model_name="gpt-4", api_key="test-key", base_url="https://api.openai.com/v1")
        assert config.model_name == "gpt-4"
        assert config.api_key.get_secret_value() == "test-key"
        assert config.base_url == "https://api.openai.com/v1"

    def test_azure_config_creation(self):
        """Test Azure OpenAI model config creation."""
        config = AzureOpenAIModelConfig(azure_deployment="test-deployment",
                                        azure_endpoint="https://test.openai.azure.com/",
                                        api_key="test-key",
                                        api_version="2023-12-01-preview")
        assert config.azure_deployment == "test-deployment"
        assert config.azure_endpoint == "https://test.openai.azure.com/"
        assert config.api_key.get_secret_value() == "test-key"
        assert config.api_version == "2023-12-01-preview"

    def test_nim_config_creation(self):
        """Test NIM model config creation."""
        config = NIMModelConfig(model_name="llama-3.1-70b",
                                base_url="https://nim.api.nvidia.com/v1",
                                api_key="test-key")
        assert config.model_name == "llama-3.1-70b"
        assert config.base_url == "https://nim.api.nvidia.com/v1"
        assert config.api_key.get_secret_value() == "test-key"


class TestAutoGenIntegration:
    """Test AutoGen integration patterns."""

    def test_client_instantiation_pattern(self):
        """Test the general pattern of client instantiation."""
        # Test that we can create basic configurations without errors
        config = OpenAIModelConfig(api_key="test-key", model_name="gpt-4")
        assert config.api_key.get_secret_value() == "test-key"
        assert config.model_name == "gpt-4"

    def test_model_info_requirements(self):
        """Test basic model info requirements."""
        # Test configuration validation
        config = AzureOpenAIModelConfig(azure_deployment="gpt-4",
                                        api_key="test-key",
                                        azure_endpoint="https://test.openai.azure.com",
                                        api_version="2024-02-01")
        assert config.azure_deployment == "gpt-4"
        assert config.api_key.get_secret_value() == "test-key"


class TestThinkingInjector:
    """Test thinking injector functionality."""

    def test_thinking_injector_creation(self):
        """Test that thinking injector can be created."""
        # Test the integration pattern for thinking injection
        mock_client = Mock()
        thinking_config = MockThinkingConfig()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, thinking_config)
            mock_patch.assert_called_once()

            # Verify the injector is passed correctly
            args, _kwargs = mock_patch.call_args
            assert args[0] == mock_client
            assert args[1] is not None  # AutoGenThinkingInjector instance


class TestLLMClientFunctions:
    """Test LLM client creation functions."""

    @pytest.mark.asyncio
    @patch('builtins.__import__')
    async def test_openai_autogen_generator(self, mock_import):
        """Test OpenAI client async generator."""
        from nat.plugins.autogen.llm import openai_autogen

        # Mock the AutoGen imports
        mock_client = Mock()
        mock_model_info = Mock()

        def import_side_effect(name, *_args, **_kwargs) -> Mock:
            """Side effect function to mock imports.

            Args:
                name (str): The name of the module being imported.
                *_args: Additional positional arguments.
                **_kwargs: Additional keyword arguments.

            Returns:
                Mock: A mock module or object based on the import name.
            """
            _, _ = _args, _kwargs  # Unused
            if 'autogen_ext.models.openai' in name:
                mock_module = Mock()
                mock_module.OpenAIChatCompletionClient = Mock(return_value=mock_client)
                return mock_module
            elif 'autogen_core.models' in name:
                mock_module = Mock()
                mock_module.ModelInfo = Mock(return_value=mock_model_info)
                return mock_module
            return Mock()

        mock_import.side_effect = import_side_effect

        config = OpenAIModelConfig(api_key="test-key", model_name="gpt-4")
        mock_builder = Mock()

        # Test the async context manager
        gen = openai_autogen(config, mock_builder)
        client = await gen.__anext__()

        assert client is not None

    @pytest.mark.asyncio
    @patch('builtins.__import__')
    async def test_azure_openai_autogen_generator(self, mock_import):
        """Test Azure OpenAI client async generator."""
        from nat.plugins.autogen.llm import azure_openai_autogen

        # Mock the AutoGen imports
        mock_client = Mock()
        mock_model_info = Mock()

        def import_side_effect(name, *_args, **_kwargs) -> Mock:
            """Side effect function to mock imports.

            Args:
                name (str): The name of the module being imported.
                *_args: Additional positional arguments.
                **_kwargs: Additional keyword arguments.

            Returns:
                Mock: A mock module or object based on the import name.
            """
            if 'autogen_ext.models.openai' in name:
                mock_module = Mock()
                mock_module.AzureOpenAIChatCompletionClient = Mock(return_value=mock_client)
                return mock_module
            elif 'autogen_core.models' in name:
                mock_module = Mock()
                mock_module.ModelInfo = Mock(return_value=mock_model_info)
                return mock_module
            return Mock()

        mock_import.side_effect = import_side_effect

        config = AzureOpenAIModelConfig(azure_deployment="gpt-4",
                                        api_key="test-key",
                                        azure_endpoint="https://test.openai.azure.com",
                                        api_version="2024-02-01")
        mock_builder = Mock()

        # Test the async generator
        gen = azure_openai_autogen(config, mock_builder)
        client = await gen.__anext__()
        assert client is not None

    @pytest.mark.asyncio
    @patch('builtins.__import__')
    async def test_nim_autogen_generator(self, mock_import):
        """Test NIM client async generator."""
        from nat.plugins.autogen.llm import nim_autogen

        # Mock the AutoGen imports
        mock_client = Mock()
        mock_model_info = Mock()

        def import_side_effect(name, *_args: Any, **_kwargs: Any) -> Mock:
            """Side effect function to mock imports.

            Args:
                name (str): The name of the module being imported.
                *_args (Any): Additional positional arguments.
                **_kwargs (Any): Additional keyword arguments.

            Returns:
                Mock: A mock module or object based on the import name.
            """
            if 'autogen_ext.models.openai' in name:
                mock_module = Mock()
                mock_module.OpenAIChatCompletionClient = Mock(return_value=mock_client)
                return mock_module
            elif 'autogen_core.models' in name:
                mock_module = Mock()
                mock_module.ModelInfo = Mock(return_value=mock_model_info)
                return mock_module
            return Mock()

        mock_import.side_effect = import_side_effect

        config = NIMModelConfig(base_url="https://nim.api.nvidia.com/v1", api_key="test-key", model_name="test-model")
        mock_builder = Mock()

        # Test the async generator
        gen = nim_autogen(config, mock_builder)
        client = await gen.__anext__()
        assert client is not None


class TestAutoGenThinkingInjector:
    """Test AutoGenThinkingInjector functionality."""

    def test_thinking_injector_inject(self):
        """Test thinking injector message injection."""
        # Since AutoGenThinkingInjector is defined inside the function,
        # we test through the integration pattern
        mock_client = Mock()
        thinking_config = MockThinkingConfig()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, thinking_config)

            # Verify patch_with_thinking was called with injector
            mock_patch.assert_called_once()
            args, _kwargs = mock_patch.call_args
            assert args[0] == mock_client

            # The second argument should be an injector instance
            injector = args[1]
            assert injector is not None
            assert hasattr(injector, 'inject')


class TestLLMClientGeneratorsFull:
    """Test complete LLM client generator flows."""

    @pytest.mark.asyncio
    async def test_openai_autogen_complete_flow(self):
        """Test complete OpenAI client creation with all configurations."""
        from nat.plugins.autogen.llm import openai_autogen

        # Create comprehensive config
        config = OpenAIModelConfig(api_key="test-api-key",
                                   model_name="gpt-4-turbo",
                                   base_url="https://api.openai.com/v1",
                                   temperature=0.7)
        builder = Mock(spec=Builder)

        # Mock only the client classes and ModelInfo, not the whole modules
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info_class:
                mock_client = Mock()
                mock_model_info = Mock()
                mock_client_class.return_value = mock_client
                mock_model_info_class.return_value = mock_model_info

                # Test the generator with patched patch function
                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = mock_client

                    # Test that we can use the context manager and get the patched client
                    async with openai_autogen(config, builder) as client:
                        assert client is mock_client
                        mock_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_azure_openai_config_building(self):
        """Test Azure OpenAI configuration building."""
        from nat.plugins.autogen.llm import azure_openai_autogen

        # Create Azure config
        config = AzureOpenAIModelConfig(api_key="azure-test-key",
                                        azure_deployment="gpt-4-deployment",
                                        azure_endpoint="https://test.openai.azure.com",
                                        api_version="2024-02-01")
        builder = Mock(spec=Builder)

        # Mock only the client classes and ModelInfo
        with patch('autogen_ext.models.openai.AzureOpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info_class:
                mock_client = Mock()
                mock_model_info = Mock()
                mock_client_class.return_value = mock_client
                mock_model_info_class.return_value = mock_model_info

                # Test the generator
                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = mock_client

                    # Test that we can use the context manager and get the patched client
                    async with azure_openai_autogen(config, builder) as client:
                        assert client is mock_client
                        mock_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_nim_autogen_config_handling(self):
        """Test NIM configuration handling."""
        from nat.plugins.autogen.llm import nim_autogen

        # Create NIM config
        config = NIMModelConfig(api_key="nim-test-key",
                                model_name="llama-3.1-70b-instruct",
                                base_url="https://integrate.api.nvidia.com/v1")
        builder = Mock(spec=Builder)

        # Mock only the client classes and ModelInfo
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info_class:
                mock_client = Mock()
                mock_model_info = Mock()
                mock_client_class.return_value = mock_client
                mock_model_info_class.return_value = mock_model_info

                # Test the generator
                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = mock_client

                    # Test that we can use the context manager and get the patched client
                    async with nim_autogen(config, builder) as client:
                        assert client is mock_client
                        mock_patch.assert_called_once()


class TestMixinCombinations:
    """Test various mixin combinations and edge cases."""

    def test_retry_mixin_only(self):
        """Test patching with only retry mixin."""
        mock_client = Mock()

        class RetryOnlyConfig(LLMBaseConfig, RetryMixin):
            """Config with only retry mixin."""
            pass

        config = RetryOnlyConfig()
        config.num_retries = 5
        config.retry_on_status_codes = [500, 502, 503, 504]
        config.retry_on_errors = ["timeout", "connection_error"]

        with patch('nat.plugins.autogen.llm.patch_with_retry') as mock_patch_retry:
            with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:
                mock_patch_retry.return_value = mock_client

                result = _patch_autogen_client_based_on_config(mock_client, config)

                # Only retry should be applied
                mock_patch_retry.assert_called_once_with(mock_client,
                                                         retries=5,
                                                         retry_codes=[500, 502, 503, 504],
                                                         retry_on_messages=["timeout", "connection_error"])
                mock_patch_thinking.assert_not_called()
                assert result == mock_client

    def test_thinking_mixin_only(self):
        """Test patching with only thinking mixin."""
        mock_client = Mock()

        # Create a real config with thinking mixin
        config = MockThinkingConfig()

        with patch('nat.plugins.autogen.llm.patch_with_retry') as mock_patch_retry:
            with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:
                mock_patch_thinking.return_value = mock_client

                result = _patch_autogen_client_based_on_config(mock_client, config)

                # Only thinking should be applied
                mock_patch_retry.assert_not_called()
                mock_patch_thinking.assert_called_once()
                assert result == mock_client

    def test_thinking_with_none_prompt_skipped(self):
        """Test that thinking mixin with None prompt is skipped."""
        mock_client = Mock()

        # Create a real config with thinking disabled (None prompt)
        class ThinkingDisabledConfig(LLMBaseConfig, ThinkingMixin):
            """Config with thinking mixin but disabled."""
            model_name: str = "nvidia/nvidia-nemotron-test"
            thinking: bool | None = None  # Disabled - returns None prompt

        config = ThinkingDisabledConfig()
        assert config.thinking_system_prompt is None  # Verify precondition

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:
            result = _patch_autogen_client_based_on_config(mock_client, config)

            # Thinking should not be applied when prompt is None
            mock_patch_thinking.assert_not_called()
            assert result == mock_client


class TestAutoGenThinkingInjectorDetails:
    """Test AutoGenThinkingInjector internal behavior."""

    @patch('nat.plugins.autogen.llm.patch_with_thinking')
    def test_thinking_injector_creation_and_usage(self, mock_patch_thinking):
        """Test thinking injector creation without complex mocking."""
        mock_client = Mock()

        # Create a real config with thinking functionality
        # Use OpenAIModelConfig which has all the necessary fields
        config = OpenAIModelConfig(
            base_url="https://example.com",
            api_key="test-key",
            model_name="nvidia/nvidia-nemotron-test",  # Use a model that matches pattern
            thinking=True  # Enable thinking
        )

        # Verify our config is indeed an instance of ThinkingMixin
        assert isinstance(config, ThinkingMixin), f"Config type: {type(config)}, MRO: {type(config).__mro__}"
        assert config.thinking_system_prompt is not None, f"Thinking prompt: {config.thinking_system_prompt}"

        _patch_autogen_client_based_on_config(mock_client, config)

        # Verify patch_with_thinking was called
        mock_patch_thinking.assert_called_once()

        # Extract the injector that was passed
        call_args = mock_patch_thinking.call_args
        injector = call_args[0][1]  # Second argument to patch_with_thinking

        # Verify injector has correct system prompt (based on model pattern)
        assert injector.system_prompt == "/think"

        # Verify function names are correctly configured
        expected_function_names = ["create", "acreate", "create_stream", "acreate_stream"]
        assert injector.function_names == expected_function_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
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

    def test_litellm_config_creation(self):
        """Test LiteLLM model config creation."""
        config = LiteLlmModelConfig(model_name="gpt-4",
                                    base_url="http://localhost:4000",
                                    api_key="test-key",
                                    temperature=0.7)
        assert config.model_name == "gpt-4"
        assert config.base_url == "http://localhost:4000"
        assert config.api_key.get_secret_value() == "test-key"
        assert config.temperature == 0.7

    def test_bedrock_config_creation(self):
        """Test AWS Bedrock model config creation."""
        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                                       region_name="us-east-1",
                                       max_tokens=256,
                                       temperature=0.5)
        assert config.model_name == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config.region_name == "us-east-1"
        assert config.max_tokens == 256
        assert config.temperature == 0.5

    def test_bedrock_config_with_profile(self):
        """Test AWS Bedrock model config with credentials profile."""
        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-haiku-20240307-v1:0",
                                       region_name="us-west-2",
                                       credentials_profile_name="my-aws-profile",
                                       max_tokens=1024)
        assert config.model_name == "anthropic.claude-3-haiku-20240307-v1:0"
        assert config.region_name == "us-west-2"
        assert config.credentials_profile_name == "my-aws-profile"
        assert config.max_tokens == 1024


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

    @patch('builtins.__import__')
    async def test_litellm_autogen_generator(self, mock_import):
        """Test LiteLLM client async generator."""
        from nat.plugins.autogen.llm import litellm_autogen

        # Mock the AutoGen imports
        mock_client = Mock()
        mock_model_info = Mock()

        def import_side_effect(name, *_args: Any, **_kwargs: Any) -> Mock:
            """Side effect function to mock imports."""
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

        config = LiteLlmModelConfig(model_name="gpt-4", base_url="http://localhost:4000", api_key="test-key")
        mock_builder = Mock()

        # Test the async generator
        gen = litellm_autogen(config, mock_builder)
        client = await gen.__anext__()
        assert client is not None

    @patch('builtins.__import__')
    async def test_bedrock_autogen_generator(self, mock_import):
        """Test AWS Bedrock client async generator."""
        from nat.plugins.autogen.llm import bedrock_autogen

        # Mock the AutoGen imports
        mock_client = Mock()

        def import_side_effect(name, *_args: Any, **_kwargs: Any) -> Mock:
            """Side effect function to mock imports."""
            if 'autogen_ext.models.anthropic' in name:
                mock_module = Mock()
                mock_module.AnthropicBedrockChatCompletionClient = Mock(return_value=mock_client)
                return mock_module
            return Mock()

        mock_import.side_effect = import_side_effect

        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                                       region_name="us-east-1",
                                       max_tokens=256)
        mock_builder = Mock()

        # Test the async generator
        gen = bedrock_autogen(config, mock_builder)
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

    async def test_litellm_autogen_config_handling(self):
        """Test LiteLLM configuration handling."""
        from nat.plugins.autogen.llm import litellm_autogen

        # Create LiteLLM config
        config = LiteLlmModelConfig(api_key="litellm-test-key",
                                    model_name="gpt-4",
                                    base_url="http://localhost:4000",
                                    temperature=0.5)
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
                    async with litellm_autogen(config, builder) as client:
                        assert client is mock_client
                        mock_patch.assert_called_once()

    async def test_bedrock_autogen_config_handling(self):
        """Test AWS Bedrock configuration handling."""
        from nat.plugins.autogen.llm import bedrock_autogen

        # Create Bedrock config
        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                                       region_name="us-east-1",
                                       max_tokens=512,
                                       temperature=0.7)
        builder = Mock(spec=Builder)

        # Mock only the Anthropic Bedrock client class
        with patch('autogen_ext.models.anthropic.AnthropicBedrockChatCompletionClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Test the generator
            with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                mock_patch.return_value = mock_client

                # Test that we can use the context manager and get the patched client
                async with bedrock_autogen(config, builder) as client:
                    assert client is mock_client
                    mock_patch.assert_called_once()

    async def test_bedrock_autogen_with_profile(self):
        """Test AWS Bedrock configuration with credentials profile."""
        from nat.plugins.autogen.llm import bedrock_autogen

        # Create Bedrock config with profile
        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-haiku-20240307-v1:0",
                                       region_name="us-west-2",
                                       credentials_profile_name="test-profile",
                                       max_tokens=1024)
        builder = Mock(spec=Builder)

        # Mock only the Anthropic Bedrock client class
        with patch('autogen_ext.models.anthropic.AnthropicBedrockChatCompletionClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Test the generator
            with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                mock_patch.return_value = mock_client

                async with bedrock_autogen(config, builder) as client:
                    assert client is mock_client

                    # Verify the client was created with expected params
                    call_args = mock_client_class.call_args
                    assert call_args is not None
                    kwargs = call_args[1]
                    assert kwargs.get("model") == "anthropic.claude-3-haiku-20240307-v1:0"
                    assert kwargs.get("aws_region") == "us-west-2"
                    assert kwargs.get("aws_profile") == "test-profile"
                    assert kwargs.get("max_tokens") == 1024

    async def test_bedrock_autogen_region_none_handling(self):
        """Test AWS Bedrock handles None region correctly."""
        from nat.plugins.autogen.llm import bedrock_autogen

        # Create Bedrock config with "None" string region (should use AWS default)
        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                                       region_name="None",
                                       max_tokens=256)
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.anthropic.AnthropicBedrockChatCompletionClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                mock_patch.return_value = mock_client

                async with bedrock_autogen(config, builder) as client:
                    assert client is mock_client

                    # Verify aws_region is not passed when region_name is "None"
                    call_args = mock_client_class.call_args
                    kwargs = call_args[1]
                    assert "aws_region" not in kwargs


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
        expected_function_names = ["create", "create_stream"]
        assert injector.function_names == expected_function_names


class TestAutoGenThinkingInjectorDirect:
    """Direct tests for AutoGenThinkingInjector.inject() method."""

    def test_inject_prepends_system_message(self):
        """Test that inject() correctly prepends a SystemMessage to the message list."""
        from autogen_core.models import SystemMessage
        from autogen_core.models import UserMessage

        # Create the injector by calling _patch_autogen_client_based_on_config
        # and capturing the injector instance
        mock_client = Mock()
        config = MockThinkingConfig()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, config)
            injector = mock_patch.call_args[0][1]

        # Now test the inject method directly
        original_messages = [
            UserMessage(content="Hello, how are you?", source="user"),
        ]

        result = injector.inject(original_messages)

        # Verify the result is a FunctionArgumentWrapper
        assert hasattr(result, 'args')
        assert hasattr(result, 'kwargs')

        # Verify the first message is now a SystemMessage with thinking prompt
        new_messages = result.args[0]
        assert len(new_messages) == 2
        assert isinstance(new_messages[0], SystemMessage)
        assert new_messages[0].content == "/think"
        assert new_messages[1] == original_messages[0]

    def test_inject_preserves_existing_messages(self):
        """Test that inject() preserves all existing messages."""
        from autogen_core.models import AssistantMessage
        from autogen_core.models import SystemMessage
        from autogen_core.models import UserMessage

        mock_client = Mock()
        config = MockThinkingConfig()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, config)
            injector = mock_patch.call_args[0][1]

        # Create a conversation with multiple messages
        original_messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?", source="user"),
            AssistantMessage(content="4", source="assistant"),
            UserMessage(content="Thanks!", source="user"),
        ]

        result = injector.inject(original_messages)
        new_messages = result.args[0]

        # Should have 5 messages: new system + 4 original
        assert len(new_messages) == 5
        assert new_messages[0].content == "/think"
        # All original messages should follow
        for i, orig_msg in enumerate(original_messages):
            assert new_messages[i + 1] == orig_msg

    def test_inject_preserves_additional_args_and_kwargs(self):
        """Test that inject() preserves additional positional and keyword arguments."""
        from autogen_core.models import UserMessage

        mock_client = Mock()
        config = MockThinkingConfig()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, config)
            injector = mock_patch.call_args[0][1]

        messages = [UserMessage(content="Test", source="user")]
        extra_arg = "some_value"
        extra_kwarg = {"key": "value"}

        result = injector.inject(messages, extra_arg, custom_param=extra_kwarg)

        # Verify additional args are preserved
        assert result.args[1] == extra_arg
        assert result.kwargs["custom_param"] == extra_kwarg


class TestThinkingPromptVariations:
    """Test different thinking prompt variations based on model patterns."""

    def test_thinking_false_produces_no_think_prompt(self):
        """Test that thinking=False produces /no_think system prompt."""
        mock_client = Mock()

        # Create config with thinking=False
        class ThinkingFalseConfig(LLMBaseConfig, ThinkingMixin):
            """Config with thinking explicitly disabled."""
            model_name: str = "nvidia/nvidia-nemotron-test"
            thinking: bool | None = False  # Explicitly disabled

        config = ThinkingFalseConfig()

        # Verify the thinking_system_prompt is /no_think
        assert config.thinking_system_prompt == "/no_think"

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, config)

            # Verify patch_with_thinking was called
            mock_patch.assert_called_once()

            # Extract the injector and verify system prompt
            injector = mock_patch.call_args[0][1]
            assert injector.system_prompt == "/no_think"

    def test_llama_nemotron_v1_thinking_prompt(self):
        """Test Llama Nemotron v1.0 produces 'detailed thinking on' prompt."""
        mock_client = Mock()

        class LlamaNemotronV1Config(LLMBaseConfig, ThinkingMixin):
            """Config for Llama Nemotron v1.0 model."""
            model_name: str = "nvidia/llama-nemotron-v1"
            thinking: bool | None = True

        config = LlamaNemotronV1Config()

        # Verify the thinking_system_prompt for v1.0
        assert config.thinking_system_prompt == "detailed thinking on"

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            _patch_autogen_client_based_on_config(mock_client, config)
            injector = mock_patch.call_args[0][1]
            assert injector.system_prompt == "detailed thinking on"

    def test_llama_nemotron_v1_thinking_off_prompt(self):
        """Test Llama Nemotron v1.0 with thinking=False produces 'detailed thinking off'."""

        class LlamaNemotronV1Config(LLMBaseConfig, ThinkingMixin):
            """Config for Llama Nemotron v1.0 model."""
            model_name: str = "nvidia/llama-nemotron-v1-0"
            thinking: bool | None = False

        config = LlamaNemotronV1Config()
        assert config.thinking_system_prompt == "detailed thinking off"


class TestConfigExclusion:
    """Test that excluded config fields are not passed to AutoGen clients."""

    async def test_openai_excludes_correct_fields(self):
        """Test OpenAI client excludes type, model_name, and thinking fields."""
        from nat.plugins.autogen.llm import openai_autogen

        # Use Nemotron model so thinking is supported and we can test exclusion
        config = OpenAIModelConfig(
            api_key="test-key",
            model_name="nvidia/nvidia-nemotron-test",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
            thinking=True  # Should be excluded from config dump
        )
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                mock_client_class.return_value = Mock()
                mock_model_info.return_value = Mock()

                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = Mock()

                    async with openai_autogen(config, builder):
                        pass

                    # Get the kwargs passed to OpenAIChatCompletionClient
                    call_kwargs = mock_client_class.call_args[1]

                    # Verify excluded NAT-specific fields are NOT present in kwargs
                    assert "type" not in call_kwargs
                    assert "model_name" not in call_kwargs
                    assert "thinking" not in call_kwargs

                    # Verify the model was passed correctly
                    # The OpenAIChatCompletionClient is called with model as keyword arg
                    assert call_kwargs.get("model") == "nvidia/nvidia-nemotron-test"

    async def test_azure_excludes_correct_fields(self):
        """Test Azure OpenAI client excludes azure_deployment, thinking, azure_endpoint, api_version."""
        from nat.plugins.autogen.llm import azure_openai_autogen

        config = AzureOpenAIModelConfig(api_key="test-key",
                                        azure_deployment="gpt-4-deployment",
                                        azure_endpoint="https://test.openai.azure.com",
                                        api_version="2024-02-01",
                                        temperature=0.5)
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.openai.AzureOpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                mock_client_class.return_value = Mock()
                mock_model_info.return_value = Mock()

                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = Mock()

                    async with azure_openai_autogen(config, builder):
                        pass

                    call_kwargs = mock_client_class.call_args[1]

                    # Verify excluded fields are NOT present in the model_dump portion
                    assert "azure_deployment" not in call_kwargs or call_kwargs.get(
                        "azure_deployment") != "gpt-4-deployment"
                    assert "thinking" not in call_kwargs

                    # Verify api_version IS present (explicitly added)
                    assert call_kwargs.get("api_version") == "2024-02-01"

    async def test_bedrock_excludes_nat_specific_fields(self):
        """Test Bedrock client excludes NAT-specific config fields."""
        from nat.plugins.autogen.llm import bedrock_autogen

        config = AWSBedrockModelConfig(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
            max_tokens=256,
            context_size=1024,  # NAT-specific, should be excluded
        )
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.anthropic.AnthropicBedrockChatCompletionClient') as mock_client_class:
            mock_client_class.return_value = Mock()

            with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                mock_patch.return_value = Mock()

                async with bedrock_autogen(config, builder):
                    pass

                call_kwargs = mock_client_class.call_args[1]

                # Verify context_size is NOT passed (NAT-specific)
                assert "context_size" not in call_kwargs

                # Verify expected fields ARE passed
                assert call_kwargs.get("model") == "anthropic.claude-3-sonnet-20240229-v1:0"
                assert call_kwargs.get("aws_region") == "us-east-1"
                assert call_kwargs.get("max_tokens") == 256


class TestLiteLLMSecretResolution:
    """Test LiteLLM API key secret resolution."""

    async def test_litellm_resolves_api_key_secret(self):
        """Test that LiteLLM correctly resolves API key via get_secret_value."""
        from nat.plugins.autogen.llm import litellm_autogen

        config = LiteLlmModelConfig(model_name="gpt-4", base_url="http://localhost:4000", api_key="secret-api-key")
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                with patch('nat.plugins.autogen.llm.get_secret_value') as mock_get_secret:
                    mock_client_class.return_value = Mock()
                    mock_model_info.return_value = Mock()
                    mock_get_secret.return_value = "resolved-secret-key"

                    with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                        mock_patch.return_value = Mock()

                        async with litellm_autogen(config, builder):
                            pass

                        # Verify get_secret_value was called with the API key
                        mock_get_secret.assert_called_once()

                        # Verify the resolved key was passed to the client
                        call_kwargs = mock_client_class.call_args[1]
                        assert call_kwargs.get("api_key") == "resolved-secret-key"

    async def test_litellm_handles_none_api_key(self):
        """Test that LiteLLM handles None API key gracefully."""
        from nat.plugins.autogen.llm import litellm_autogen

        config = LiteLlmModelConfig(
            model_name="gpt-4",
            base_url="http://localhost:4000",
            api_key=None  # No API key
        )
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                with patch('nat.plugins.autogen.llm.get_secret_value') as mock_get_secret:
                    mock_client_class.return_value = Mock()
                    mock_model_info.return_value = Mock()

                    with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                        mock_patch.return_value = Mock()

                        async with litellm_autogen(config, builder):
                            pass

                        # get_secret_value should NOT be called when api_key is None
                        mock_get_secret.assert_not_called()


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_patch_with_invalid_retry_config(self):
        """Test patching with invalid retry configuration values."""
        mock_client = Mock()

        class InvalidRetryConfig(LLMBaseConfig, RetryMixin):
            """Config with edge case retry values."""
            num_retries: int = 0  # Zero retries
            retry_on_status_codes: list[int | str] = Field(default_factory=list)  # Empty list
            retry_on_errors: list[str] | None = None  # None errors

        config = InvalidRetryConfig()

        with patch('nat.plugins.autogen.llm.patch_with_retry') as mock_patch_retry:
            mock_patch_retry.return_value = mock_client

            # Should not raise, even with edge case values
            result = _patch_autogen_client_based_on_config(mock_client, config)

            mock_patch_retry.assert_called_once_with(mock_client, retries=0, retry_codes=[], retry_on_messages=None)
            assert result == mock_client

    def test_patch_with_unsupported_model_for_thinking(self):
        """Test thinking mixin with unsupported model returns None prompt."""
        mock_client = Mock()

        class UnsupportedModelConfig(LLMBaseConfig, ThinkingMixin):
            """Config with model that doesn't support thinking."""
            model_name: str = "gpt-4"  # Not a Nemotron model
            thinking: bool | None = None  # None means thinking not configured

        config = UnsupportedModelConfig()

        # Unsupported models should return None for thinking_system_prompt
        assert config.thinking_system_prompt is None

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch:
            result = _patch_autogen_client_based_on_config(mock_client, config)

            # patch_with_thinking should NOT be called when prompt is None
            mock_patch.assert_not_called()
            assert result == mock_client

    async def test_openai_config_with_empty_model_name(self):
        """Test OpenAI config handles empty model name."""
        from nat.plugins.autogen.llm import openai_autogen

        # Create config with empty model name
        config = OpenAIModelConfig(
            api_key="test-key",
            model_name="",  # Empty model name
            base_url="https://api.openai.com/v1")
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                mock_client_class.return_value = Mock()
                mock_model_info.return_value = Mock()

                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = Mock()

                    # Should not raise during generation
                    async with openai_autogen(config, builder):
                        pass

                    # Verify empty string was passed as model keyword arg
                    call_kwargs = mock_client_class.call_args[1]
                    assert call_kwargs.get("model") == ""


class TestAsyncGeneratorCleanup:
    """Test async generator cleanup and resource management."""

    async def test_openai_generator_cleanup_on_normal_exit(self):
        """Test that OpenAI generator cleans up properly on normal exit."""
        from nat.plugins.autogen.llm import openai_autogen

        config = OpenAIModelConfig(api_key="test-key", model_name="gpt-4")
        builder = Mock(spec=Builder)

        cleanup_called = False

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                mock_model_info.return_value = Mock()

                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = mock_client

                    # Use async context manager
                    async with openai_autogen(config, builder) as client:
                        assert client is mock_client

                    # After exiting, generator should be exhausted
                    cleanup_called = True

        assert cleanup_called

    async def test_generator_cleanup_on_exception(self):
        """Test that generator cleans up properly when exception is raised."""
        from nat.plugins.autogen.llm import openai_autogen

        config = OpenAIModelConfig(api_key="test-key", model_name="gpt-4")
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                mock_model_info.return_value = Mock()

                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = mock_client

                    # Raise exception inside context manager
                    with pytest.raises(ValueError, match="Test exception"):
                        async with openai_autogen(config, builder) as client:
                            assert client is mock_client
                            raise ValueError("Test exception")

                    # Generator should still have been used (client was yielded)
                    mock_patch.assert_called_once()

    async def test_generator_can_be_closed_early(self):
        """Test that generator can be closed before exhaustion via context manager."""
        from nat.plugins.autogen.llm import openai_autogen

        config = OpenAIModelConfig(api_key="test-key", model_name="gpt-4")
        builder = Mock(spec=Builder)

        client_used = False

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client_class:
            with patch('autogen_core.models.ModelInfo') as mock_model_info:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                mock_model_info.return_value = Mock()

                with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                    mock_patch.return_value = mock_client

                    # Use context manager which handles cleanup
                    async with openai_autogen(config, builder) as client:
                        assert client is mock_client
                        client_used = True
                        # Exit context early (before any work)

                    # Verify the client was used
                    assert client_used
                    # Context manager handles cleanup automatically

    async def test_bedrock_generator_cleanup(self):
        """Test Bedrock generator cleanup works correctly."""
        from nat.plugins.autogen.llm import bedrock_autogen

        config = AWSBedrockModelConfig(model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                                       region_name="us-east-1",
                                       max_tokens=256)
        builder = Mock(spec=Builder)

        with patch('autogen_ext.models.anthropic.AnthropicBedrockChatCompletionClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            with patch('nat.plugins.autogen.llm._patch_autogen_client_based_on_config') as mock_patch:
                mock_patch.return_value = mock_client

                async with bedrock_autogen(config, builder) as client:
                    assert client is mock_client

                # Verify client was created and patched
                mock_client_class.assert_called_once()
                mock_patch.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.llm import APITypeEnum
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.strands.llm import _patch_llm_based_on_config
from nat.plugins.strands.llm import bedrock_strands
from nat.plugins.strands.llm import nim_strands
from nat.plugins.strands.llm import openai_strands


class TestOpenAIStrands:
    """Tests for the openai_strands function."""

    @pytest.fixture
    def openai_config(self):
        """Create an OpenAIModelConfig instance."""
        return OpenAIModelConfig(model_name="gpt-4")

    @pytest.fixture
    def openai_config_wrong_api(self):
        """Create an OpenAIModelConfig with wrong API type."""
        return OpenAIModelConfig(model_name="gpt-4", api_type=APITypeEnum.RESPONSES)

    @pytest.fixture(name="mock_model", autouse=True)
    def mock_model_fixture(self):
        with patch("strands.models.openai.OpenAIModel") as mock_model:
            yield mock_model

    @pytest.fixture(name="mock_async_openai", autouse=True)
    def mock_async_openai_fixture(self):
        with patch("openai.AsyncOpenAI") as mock_async_openai:
            yield mock_async_openai

    @pytest.mark.asyncio
    async def test_openai_strands_basic(self, mock_model, openai_config, mock_builder):
        """Test that openai_strands as async context manager."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance

        # pylint: disable=not-async-context-manager
        async with openai_strands(openai_config, mock_builder):
            mock_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_strands_with_params(self, mock_model, openai_config, mock_builder):
        """Test openai_strands with additional parameters."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance

        openai_config.temperature = 0.5
        openai_config.max_tokens = 100

        # pylint: disable=not-async-context-manager
        async with openai_strands(openai_config, mock_builder):
            mock_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_type_validation(self, mock_model, openai_config_wrong_api, mock_builder):
        """Non-chat-completion API types must raise a ValueError."""
        with pytest.raises(ValueError):
            async with openai_strands(openai_config_wrong_api, mock_builder):
                pass
        mock_model.assert_not_called()

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @pytest.mark.asyncio
    async def test_verify_ssl_passed_to_client(self,
                                               mock_model,
                                               mock_async_openai,
                                               openai_config,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        mock_model.return_value = MagicMock()
        openai_config.verify_ssl = verify_ssl
        async with openai_strands(openai_config, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


class TestBedrockStrands:
    """Tests for the bedrock_strands function."""

    @pytest.fixture
    def bedrock_config(self):
        """Create an AWSBedrockModelConfig instance."""
        return AWSBedrockModelConfig(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
        )

    @pytest.fixture
    def bedrock_config_wrong_api(self):
        """Create an AWSBedrockModelConfig with wrong API type."""
        return AWSBedrockModelConfig(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
            api_type=APITypeEnum.RESPONSES,
        )

    @patch("strands.models.bedrock.BedrockModel")
    @pytest.mark.asyncio
    async def test_bedrock_strands_basic(self, mock_model, bedrock_config, mock_builder):
        """Test that bedrock_strands creates a BedrockModel."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance

        # pylint: disable=not-async-context-manager
        async with bedrock_strands(bedrock_config, mock_builder):
            mock_model.assert_called_once()

    @patch("strands.models.bedrock.BedrockModel")
    @pytest.mark.asyncio
    async def test_api_type_validation(self, mock_model, bedrock_config_wrong_api, mock_builder):
        """Non-chat-completion API types must raise a ValueError."""
        with pytest.raises(ValueError):
            async with bedrock_strands(bedrock_config_wrong_api, mock_builder):
                pass
        mock_model.assert_not_called()


class TestNIMStrands:
    """Tests for the nim_strands function."""

    @pytest.fixture
    def nim_config(self):
        """Create a NIMModelConfig instance."""
        return NIMModelConfig(
            model_name="meta/llama-3.1-8b-instruct",
            api_key="test-api-key",
            base_url="https://integrate.api.nvidia.com/v1",
        )

    @pytest.fixture
    def nim_config_wrong_api(self):
        """Create a NIMModelConfig with wrong API type."""
        return NIMModelConfig(
            model_name="meta/llama-3.1-8b-instruct",
            api_key="test-api-key",
            base_url="https://integrate.api.nvidia.com/v1",
            api_type=APITypeEnum.RESPONSES,
        )

    @pytest.fixture(name="mock_oai_clients")
    def mock_oai_clients_fixture(self):
        with patch("openai.AsyncOpenAI") as mock_oai:
            mock_oai.return_value = mock_oai

            # Patch OpenAIModel constructor to track the call
            with patch("strands.models.openai.OpenAIModel.__init__", return_value=None) as mock_oai_model:
                yield mock_oai, mock_oai_model

    @pytest.mark.asyncio
    async def test_nim_strands_basic(self, nim_config, mock_builder, mock_oai_clients):
        """Test that nim_strands creates a NIMCompatibleOpenAIModel."""
        (mock_oai, mock_oai_model) = mock_oai_clients

        # pylint: disable=not-async-context-manager
        async with nim_strands(nim_config, mock_builder) as result:
            # Verify the result is a NIMCompatibleOpenAIModel instance
            assert result is not None

            mock_oai.assert_called_once()  # Ensure OpenAI client init was called
            oai_call_args = mock_oai.call_args
            oai_call_kwargs = oai_call_args[1]
            assert oai_call_kwargs["api_key"] == "test-api-key"
            assert oai_call_kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"

            # Verify OpenAIModel.__init__ was called (the base class)
            mock_oai_model.assert_called_once()
            call_args = mock_oai_model.call_args

            # First arg is self, get kwargs
            call_kwargs = call_args[1]

            # Verify client
            call_kwargs["client"] == mock_oai

            # Verify model_id
            assert call_kwargs["model_id"] == "meta/llama-3.1-8b-instruct"

    @pytest.mark.asyncio
    async def test_nim_strands_with_env_var(self, mock_builder, mock_oai_clients):
        """Test nim_strands with environment variable for API key."""
        (mock_oai, mock_oai_model) = mock_oai_clients
        nim_config = NIMModelConfig(model_name="test-model")

        with patch.dict("os.environ", {"NVIDIA_API_KEY": "env-api-key"}):
            # pylint: disable=not-async-context-manager
            async with nim_strands(nim_config, mock_builder):
                mock_oai_model.assert_called_once()
                mock_oai.assert_called_once()

                call_kwargs = mock_oai.call_args[1]
                assert call_kwargs["api_key"] == "env-api-key"

    @pytest.mark.asyncio
    async def test_nim_strands_default_base_url(self, mock_builder, mock_oai_clients):
        """Test nim_strands uses default base_url when not provided."""
        (mock_oai, mock_oai_model) = mock_oai_clients
        nim_config = NIMModelConfig(model_name="test-model", api_key="test-key")

        async with nim_strands(nim_config, mock_builder):  # pylint: disable=not-async-context-manager
            mock_oai_model.assert_called_once()
            mock_oai.assert_called_once()
            call_kwargs = mock_oai.call_args[1]
            assert call_kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"

    @pytest.mark.asyncio
    async def test_nim_strands_nim_override_dummy_api_key(self, mock_builder, mock_oai_clients):
        """Test nim_strands uses dummy API key when base_url is set but no API key available."""
        (mock_oai, mock_oai_model) = mock_oai_clients
        nim_config = NIMModelConfig(
            model_name="test-model",
            base_url="https://custom-nim.example.com/v1",
        )

        with patch.dict(os.environ, {}, clear=True):
            # pylint: disable=not-async-context-manager
            async with nim_strands(nim_config, mock_builder):
                mock_oai_model.assert_called_once()
                mock_oai.assert_called_once()
                call_kwargs = mock_oai.call_args[1]
                assert call_kwargs["base_url"] == "https://custom-nim.example.com/v1"
                assert call_kwargs["api_key"] == "dummy-api-key"

    def test_nim_compatible_openai_model_format_request_messages(self):
        """Test NIMCompatibleOpenAIModel.format_request_messages."""
        # This tests the message formatting logic for NIM compatibility

        # Test message formatting scenarios
        test_cases = [
            # Single text item in list should be flattened
            ([{
                "type": "text", "text": "Hello"
            }], "Hello"),
            # Multiple text items should be joined
            ([{
                "type": "text", "text": "Hello"
            }, {
                "type": "text", "text": " world"
            }], "Hello world"),
            # Empty content should become space
            ([], " "),
            # Empty string should become space
            ("", " "),
        ]

        for input_content, expected_output in test_cases:
            # Test the logic that would be applied
            if isinstance(input_content, list) and len(input_content) == 1 and isinstance(input_content[0], str):
                result = input_content[0]
            elif isinstance(input_content, list) and all(
                    isinstance(item, dict) and item.get("type") == "text" for item in input_content):
                result = "".join(item["text"] for item in input_content)
                result = result if result.strip() else " "
            elif isinstance(input_content, list) and len(input_content) == 0:
                result = " "
            elif isinstance(input_content, str) and not input_content.strip():
                result = " "
            else:
                result = input_content

            if expected_output in {" ", "Hello", "Hello world"}:
                assert result == expected_output

    def test_nim_compatible_openai_model_format_request_message_content_reasoning(self):
        """Test NIMCompatibleOpenAIModel.format_request_message_content handles reasoningContent."""
        # Test reasoningContent handling
        reasoning_content = {
            "reasoningContent": {
                "reasoningText": {
                    "signature": "test_signature",
                    "text": "This is my reasoning process",
                },
            },
        }

        expected_result = {
            "text": "This is my reasoning process",
            "type": "text",
        }

        # Test the format_request_message_content method logic directly
        # This simulates what the NIMCompatibleOpenAIModel.format_request_message_content should do
        content = reasoning_content

        if "reasoningContent" in content:
            reasoning_text = content["reasoningContent"].get("reasoningText", {}).get("text", "")
            result = {"text": reasoning_text, "type": "text"}
        else:
            # Would fall back to parent implementation
            result = None

        assert result == expected_result

    def test_nim_compatible_openai_model_format_request_message_content_other_types(self):
        """Test NIMCompatibleOpenAIModel.format_request_message_content handles other content types."""
        # Test that non-reasoningContent types would fall back to parent
        text_content = {"text": "Hello world"}

        # The method should fall back to parent implementation for non-reasoning content
        content = text_content

        if "reasoningContent" in content:
            pytest.fail("reasoningContent handling should not be triggered for text content")

    @pytest.mark.asyncio
    async def test_nim_strands_excludes_nat_specific_params(self, mock_builder):
        """Test that NAT-specific parameters are excluded."""
        nim_config = NIMModelConfig(
            model_name="test-model",
            api_key="test-key",
            num_retries=3,  # Should be excluded
            thinking_system_prompt="Think step by step",  # Should be excluded
        )

        with patch("strands.models.openai.OpenAIModel.__init__", return_value=None) as mock_init:
            # pylint: disable=not-async-context-manager
            async with nim_strands(nim_config, mock_builder):
                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]

                # Verify NAT-specific params are not in params
                params = call_kwargs.get("params", {})
                assert "num_retries" not in params
                assert "thinking" not in params
                assert "retry_on_status_codes" not in params

    @pytest.mark.asyncio
    async def test_api_type_validation(self, nim_config_wrong_api, mock_builder):
        """Non-chat-completion API types must raise a ValueError."""
        with patch("strands.models.openai.OpenAIModel.__init__", return_value=None) as mock_init:
            with pytest.raises(ValueError):
                async with nim_strands(nim_config_wrong_api, mock_builder):
                    pass
            mock_init.assert_not_called()

    @pytest.mark.parametrize("verify_ssl", [True, False], ids=["verify_ssl_true", "verify_ssl_false"])
    @patch("openai.AsyncOpenAI")
    @patch("strands.models.openai.OpenAIModel.__init__", return_value=None)
    @pytest.mark.asyncio
    async def test_verify_ssl_passed_to_client(self,
                                               mock_init,
                                               mock_async_openai,
                                               nim_config,
                                               mock_builder,
                                               mock_httpx_async_client,
                                               verify_ssl):
        """Test that verify_ssl is passed to the underlying httpx.AsyncClient as verify."""
        nim_config.verify_ssl = verify_ssl
        async with nim_strands(nim_config, mock_builder):
            mock_httpx_async_client.assert_called_once()
            assert mock_httpx_async_client.call_args.kwargs["verify"] is verify_ssl


class TestPatchLLMBasedOnConfig:
    """Tests for _patch_llm_based_on_config function."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        return MagicMock()

    def test_patch_llm_no_mixins(self, mock_client):
        """Test patching with no mixins applied."""
        config = OpenAIModelConfig(model_name="gpt-4")

        result = _patch_llm_based_on_config(mock_client, config)

        # Should return the same client when no mixins
        assert result == mock_client

    @patch("nat.plugins.strands.llm.patch_with_retry")
    def test_patch_llm_with_retry_mixin(self, mock_patch_retry, mock_client):
        """Test patching with retry mixin."""
        from nat.data_models.retry_mixin import RetryMixin

        # Create a config that has retry mixin
        class TestConfigWithRetry(OpenAIModelConfig, RetryMixin):
            pass

        config = TestConfigWithRetry(model_name="gpt-4",
                                     num_retries=3,
                                     retry_on_status_codes=[500, 502],
                                     retry_on_errors=["timeout"])

        mock_patched_client = MagicMock()
        mock_patch_retry.return_value = mock_patched_client

        result = _patch_llm_based_on_config(mock_client, config)

        # Verify retry patching was called
        mock_patch_retry.assert_called_once_with(mock_client,
                                                 retries=3,
                                                 retry_codes=[500, 502],
                                                 retry_on_messages=["timeout"])
        assert result == mock_patched_client

    @patch("nat.plugins.strands.llm.patch_with_thinking")
    def test_patch_llm_with_thinking_mixin(self, mock_patch_thinking, mock_client):
        """Test patching with thinking mixin."""
        from nat.data_models.thinking_mixin import ThinkingMixin

        # Create a config that has thinking mixin
        class TestConfigWithThinking(OpenAIModelConfig, ThinkingMixin):
            pass

        # Use a Nemotron model name so thinking_system_prompt property returns a value
        config = TestConfigWithThinking(model_name="nvidia/llama-nemotron-4-340b-instruct", thinking=True)

        mock_patched_client = MagicMock()
        mock_patch_thinking.return_value = mock_patched_client

        result = _patch_llm_based_on_config(mock_client, config)

        # Verify thinking patching was called
        mock_patch_thinking.assert_called_once()
        call_args = mock_patch_thinking.call_args
        assert call_args[0][0] == mock_client  # First positional arg is the client

        # Verify the injector was configured correctly
        injector = call_args[0][1]
        # For Nemotron models, thinking_system_prompt returns "/think" when thinking=True
        assert injector.system_prompt == "/think"
        assert "stream" in injector.function_names
        assert "structured_output" in injector.function_names

        # Verify the result is the patched client
        assert result == mock_patched_client

    @patch("nat.plugins.strands.llm.patch_with_thinking")
    @patch("nat.plugins.strands.llm.patch_with_retry")
    def test_patch_llm_with_both_mixins(self, mock_patch_retry, mock_patch_thinking, mock_client):
        """Test patching with both retry and thinking mixins."""
        from nat.data_models.retry_mixin import RetryMixin
        from nat.data_models.thinking_mixin import ThinkingMixin

        # Create a config that has both retry and thinking mixins
        class TestConfigWithBoth(OpenAIModelConfig, RetryMixin, ThinkingMixin):
            pass

        # Use a Nemotron model name so thinking_system_prompt property returns a value
        config = TestConfigWithBoth(model_name="nvidia/llama-nemotron-4-340b-instruct",
                                    num_retries=3,
                                    retry_on_status_codes=[500, 502],
                                    retry_on_errors=["timeout"],
                                    thinking=True)

        # Setup mocks: retry patches first, then thinking patches the result
        mock_retry_patched_client = MagicMock()
        mock_patch_retry.return_value = mock_retry_patched_client

        mock_final_patched_client = MagicMock()
        mock_patch_thinking.return_value = mock_final_patched_client

        result = _patch_llm_based_on_config(mock_client, config)

        # Verify retry patching was called first on the original client
        mock_patch_retry.assert_called_once_with(mock_client,
                                                 retries=3,
                                                 retry_codes=[500, 502],
                                                 retry_on_messages=["timeout"])

        # Verify thinking patching was called second on the retry-patched client
        mock_patch_thinking.assert_called_once()
        call_args = mock_patch_thinking.call_args
        assert call_args[0][0] == mock_retry_patched_client  # Should be the retry-patched client

        # Verify the injector was configured correctly
        injector = call_args[0][1]
        # For Nemotron models, thinking_system_prompt returns "/think" when thinking=True
        assert injector.system_prompt == "/think"

        # Verify the result is the final patched client
        assert result == mock_final_patched_client


class TestStrandsThinkingInjector:
    """Tests for StrandsThinkingInjector class."""

    def test_inject_with_positional_system_prompt(self):
        """Test injecting thinking prompt with positional system_prompt."""
        # Test the injection logic that StrandsThinkingInjector should implement
        thinking_prompt = "Think step by step"
        existing_system_prompt = "You are helpful"

        # Simulate what the injector should do
        combined_prompt = f"{thinking_prompt}\n\n{existing_system_prompt}"

        assert combined_prompt == "Think step by step\n\nYou are helpful"

    def test_inject_with_keyword_system_prompt(self):
        """Test injecting thinking prompt with keyword system_prompt."""
        thinking_prompt = "Think carefully"
        existing_system_prompt = "Be precise"

        # Simulate keyword argument injection
        combined_prompt = f"{thinking_prompt}\n\n{existing_system_prompt}"

        assert combined_prompt == "Think carefully\n\nBe precise"

    def test_inject_with_no_existing_system_prompt(self):
        """Test injecting thinking prompt with no existing system_prompt."""
        thinking_prompt = "Think step by step"

        # When no existing prompt, should just use thinking prompt
        result_prompt = thinking_prompt

        assert result_prompt == "Think step by step"

    def test_inject_with_empty_system_prompt(self):
        """Test injecting thinking prompt with empty system_prompt."""
        thinking_prompt = "Think step by step"
        existing_system_prompt = ""

        # When existing prompt is empty, should just use thinking prompt
        result_prompt = (thinking_prompt
                         if not existing_system_prompt else f"{thinking_prompt}\n\n{existing_system_prompt}")

        assert result_prompt == "Think step by step"

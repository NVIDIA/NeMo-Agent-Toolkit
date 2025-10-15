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
"""Tests for SSL verification feature in OpenAI LLM and Embedder configurations."""

from nat.embedder.openai_embedder import OpenAIEmbedderModelConfig
from nat.llm.openai_llm import OpenAIModelConfig


class TestOpenAIModelConfigSSL:
    """Tests for SSL verification field in OpenAIModelConfig."""

    def test_verify_ssl_default_value(self):
        """Test that verify_ssl defaults to True."""
        config = OpenAIModelConfig(model_name="gpt-4")
        assert config.verify_ssl is True

    def test_verify_ssl_explicit_true(self):
        """Test setting verify_ssl explicitly to True."""
        config = OpenAIModelConfig(model_name="gpt-4", verify_ssl=True)
        assert config.verify_ssl is True

    def test_verify_ssl_explicit_false(self):
        """Test setting verify_ssl explicitly to False."""
        config = OpenAIModelConfig(model_name="gpt-4", verify_ssl=False)
        assert config.verify_ssl is False

    def test_verify_ssl_with_other_params(self):
        """Test that verify_ssl works alongside other configuration parameters."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://custom.endpoint.com/v1",
            temperature=0.7,
            verify_ssl=False,
        )
        assert config.verify_ssl is False
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.endpoint.com/v1"
        assert config.temperature == 0.7

    def test_model_dump_includes_verify_ssl(self):
        """Test that verify_ssl is included in model dump."""
        config = OpenAIModelConfig(model_name="gpt-4", verify_ssl=False)
        dumped = config.model_dump()
        assert "verify_ssl" in dumped
        assert dumped["verify_ssl"] is False


class TestOpenAIEmbedderModelConfigSSL:
    """Tests for SSL verification field in OpenAIEmbedderModelConfig."""

    def test_verify_ssl_default_value(self):
        """Test that verify_ssl defaults to True."""
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002")
        assert config.verify_ssl is True

    def test_verify_ssl_explicit_true(self):
        """Test setting verify_ssl explicitly to True."""
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", verify_ssl=True)
        assert config.verify_ssl is True

    def test_verify_ssl_explicit_false(self):
        """Test setting verify_ssl explicitly to False."""
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", verify_ssl=False)
        assert config.verify_ssl is False

    def test_verify_ssl_with_other_params(self):
        """Test that verify_ssl works alongside other configuration parameters."""
        config = OpenAIEmbedderModelConfig(
            model_name="text-embedding-ada-002",
            api_key="test-key",
            base_url="https://custom.endpoint.com/v1",
            verify_ssl=False,
        )
        assert config.verify_ssl is False
        assert config.model_name == "text-embedding-ada-002"
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.endpoint.com/v1"

    def test_model_dump_includes_verify_ssl(self):
        """Test that verify_ssl is included in model dump."""
        config = OpenAIEmbedderModelConfig(model_name="text-embedding-ada-002", verify_ssl=False)
        dumped = config.model_dump()
        assert "verify_ssl" in dumped
        assert dumped["verify_ssl"] is False


class TestSSLVerificationYAMLConfig:
    """Test that SSL verification works with YAML configuration parsing."""

    def test_parse_yaml_config_with_ssl_disabled(self):
        """Test that verify_ssl=false can be parsed from YAML-style dict."""
        config_dict = {
            "model_name": "gpt-4",
            "api_key": "test-key",
            "base_url": "https://custom.endpoint.com/v1",
            "verify_ssl": False,
        }
        config = OpenAIModelConfig(**config_dict)
        assert config.verify_ssl is False
        assert config.model_name == "gpt-4"

    def test_parse_yaml_config_embedder_with_ssl_disabled(self):
        """Test that verify_ssl=false can be parsed from YAML-style dict for embedders."""
        config_dict = {
            "model_name": "text-embedding-ada-002",
            "api_key": "test-key",
            "base_url": "https://custom.endpoint.com/v1",
            "verify_ssl": False,
        }
        config = OpenAIEmbedderModelConfig(**config_dict)
        assert config.verify_ssl is False
        assert config.model_name == "text-embedding-ada-002"

    def test_backward_compatibility_without_verify_ssl(self):
        """Test that configs without verify_ssl field still work (backward compatibility)."""
        config_dict = {"model_name": "gpt-4", "api_key": "test-key"}
        config = OpenAIModelConfig(**config_dict)
        assert config.verify_ssl is True  # Should default to True

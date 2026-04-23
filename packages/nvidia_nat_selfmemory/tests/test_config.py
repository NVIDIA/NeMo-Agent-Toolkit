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

from nat.plugins.selfmemory.memory import SelfMemoryProviderConfig


class TestSelfMemoryProviderConfig:

    def test_default_values(self):
        """Test config has correct defaults."""
        config = SelfMemoryProviderConfig()

        assert config.vector_store_provider == "qdrant"
        assert config.vector_store_config == {}
        assert config.embedding_provider == "openai"
        assert config.embedding_config == {}
        assert config.llm_provider is None
        assert config.llm_config == {}
        assert config.encryption_key is None

    def test_custom_values(self):
        """Test config accepts custom values."""
        config = SelfMemoryProviderConfig(
            vector_store_provider="chroma",
            vector_store_config={"persist_directory": "/tmp/chroma"},
            embedding_provider="ollama",
            embedding_config={"model": "nomic-embed-text"},
            llm_provider="openai",
            llm_config={"model": "gpt-4o-mini"},
            encryption_key="my-secret-key",
        )

        assert config.vector_store_provider == "chroma"
        assert config.vector_store_config["persist_directory"] == "/tmp/chroma"
        assert config.embedding_provider == "ollama"
        assert config.embedding_config["model"] == "nomic-embed-text"
        assert config.llm_provider == "openai"
        assert config.encryption_key == "my-secret-key"

    def test_name_attribute(self):
        """Test the config registers with name 'selfmemory'."""
        config = SelfMemoryProviderConfig()

        assert config.__class__.__name__ == "SelfMemoryProviderConfig"

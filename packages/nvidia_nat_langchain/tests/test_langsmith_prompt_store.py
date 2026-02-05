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
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langsmith.utils import LangSmithConflictError

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.models import ObjectStoreItem
from nat.plugins.langchain.langsmith_prompt_store import LangSmithPromptStore
from nat.plugins.langchain.langsmith_prompt_store import LangSmithPromptStoreConfig
from nat.plugins.langchain.langsmith_prompt_store import _validate_prompt_key

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(name="sample_prompt_manifest")
def fixture_sample_prompt_manifest() -> dict:
    """Sample prompt manifest for testing with embedded metadata."""
    return {
        "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
        "lc": 1,
        "type": "constructor",
        "kwargs": {
            "input_variables": ["question"],  # Versioned metadata embedded in the manifest
            "metadata": {
                "version": "1.0.0",
                "author": "test-user",
            },
            "messages": [{
                "id": ["langchain", "prompts", "chat", "HumanMessagePromptTemplate"],
                "lc": 1,
                "type": "constructor",
                "kwargs": {
                    "prompt": {
                        "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                        "lc": 1,
                        "type": "constructor",
                        "kwargs": {
                            "input_variables": ["question"],
                            "template": "Answer the following question: {question}",
                        },
                    },
                },
            }, ],
        },
    }


@pytest.fixture(name="mock_client")
def fixture_mock_client():
    """Create a mock AsyncClient."""
    client = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture(name="mock_prompt_info")
def fixture_mock_prompt_info():
    """Create a mock Prompt object."""
    prompt = MagicMock()
    prompt.description = "Test description"
    prompt.readme = "# Test Readme"
    prompt.tags = ["production", "approved"]
    return prompt


@pytest.fixture(name="mock_commit")
def fixture_mock_commit(sample_prompt_manifest):
    """Create a mock PromptCommit object."""
    commit = MagicMock()
    commit.manifest = sample_prompt_manifest
    return commit


# =============================================================================
# Tests for Validation Functions
# =============================================================================


class TestValidatePromptKey:
    """Tests for _validate_prompt_key function."""

    def test_valid_keys(self):
        """Test that valid prompt keys pass validation."""
        valid_keys = [
            "my-prompt",
            "test_prompt",
            "prompt1",
            "a",
            "my-test-prompt-v1",
            "prompt_with_underscores",
        ]
        for key in valid_keys:
            _validate_prompt_key(key)  # Should not raise

    def test_invalid_keys_uppercase(self):
        """Test that uppercase keys are rejected."""
        with pytest.raises(ValueError, match="Invalid prompt key"):
            _validate_prompt_key("MyPrompt")

    def test_invalid_keys_starts_with_number(self):
        """Test that keys starting with numbers are rejected."""
        with pytest.raises(ValueError, match="Invalid prompt key"):
            _validate_prompt_key("1prompt")

    def test_invalid_keys_special_chars(self):
        """Test that keys with special characters are rejected."""
        invalid_keys = ["my prompt", "prompt!", "test.prompt", "prompt@test"]
        for key in invalid_keys:
            with pytest.raises(ValueError, match="Invalid prompt key"):
                _validate_prompt_key(key)

    def test_invalid_keys_empty(self):
        """Test that empty keys are rejected."""
        with pytest.raises(ValueError, match="Invalid prompt key"):
            _validate_prompt_key("")


# =============================================================================
# Tests for LangSmithPromptStoreConfig
# =============================================================================


class TestLangSmithPromptStoreConfig:
    """Tests for LangSmithPromptStoreConfig."""

    def test_default_values(self):
        """Test config with default values."""
        config = LangSmithPromptStoreConfig()
        assert config.api_key is None
        assert config.api_url is None
        assert config.is_public is False
        assert config.default_tags is None
        assert config.timeout_ms is None

    def test_with_default_tags(self):
        """Test config with default_tags (simple strings)."""
        config = LangSmithPromptStoreConfig(default_tags=["production", "approved"])
        assert config.default_tags == ["production", "approved"]

    def test_with_timeout_ms(self):
        """Test config with timeout_ms."""
        config = LangSmithPromptStoreConfig(timeout_ms=30000)
        assert config.timeout_ms == 30000


# =============================================================================
# Tests for LangSmithPromptStore
# =============================================================================


class TestLangSmithPromptStoreInit:
    """Tests for LangSmithPromptStore initialization."""

    def test_init_with_defaults(self, monkeypatch):
        """Test initialization with default values."""
        # Clear environment variables that might be set
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        store = LangSmithPromptStore()
        assert store._api_key is None
        assert store._api_url is None
        assert store._is_public is False
        assert store._default_tags == []
        assert store._timeout_ms is None

    def test_init_with_values(self):
        """Test initialization with explicit values."""
        store = LangSmithPromptStore(
            api_key="test-key",
            api_url="https://api.test.com",
            is_public=True,
            default_tags=["env-test"],
            timeout_ms=30000,
        )
        assert store._api_key == "test-key"
        assert store._api_url == "https://api.test.com"
        assert store._is_public is True
        assert store._default_tags == ["env-test"]
        assert store._timeout_ms == 30000


class TestLangSmithPromptStoreContextManager:
    """Tests for LangSmithPromptStore context manager."""

    async def test_context_manager_connects(self):
        """Test that context manager establishes connection."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async with LangSmithPromptStore(api_key="test-key") as store:
                assert store._client is not None

            mock_client.aclose.assert_called_once()

    async def test_context_manager_closes_on_exception(self):
        """Test that context manager closes connection on exception."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError):
                async with LangSmithPromptStore(api_key="test-key"):
                    raise RuntimeError("Test error")

            mock_client.aclose.assert_called_once()

    async def test_double_enter_raises(self):
        """Test that entering context twice raises RuntimeError."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            store = LangSmithPromptStore()
            async with store:
                with pytest.raises(RuntimeError, match="already established"):
                    await store.__aenter__()


class TestLangSmithPromptStorePutObject:
    """Tests for LangSmithPromptStore.put_object."""

    async def test_put_object_success(self, mock_client):
        """Test successful put_object with versioned metadata."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                # Data is a prompt template string
                prompt_str = "Answer the following question: {question}"
                # Note: ObjectStoreItem.metadata requires dict[str, str]
                # Tags must be JSON-serialized
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={
                        "description": "Test prompt",
                        "tags": json.dumps(["production"]),
                        "version": "1.0.0",
                        "author": "test-user",
                    },
                )
                await store.put_object("test-prompt", item)

                mock_client.create_prompt.assert_called_once()
                mock_client.create_commit.assert_called_once()

                # Verify prompt-level tags
                create_call = mock_client.create_prompt.call_args
                assert "production" in create_call.kwargs.get("tags", [])

                # Verify versioned metadata in ChatPromptTemplate
                commit_call = mock_client.create_commit.call_args
                prompt_template = commit_call.kwargs.get("object") or commit_call.args[1]
                assert prompt_template.metadata == {"version": "1.0.0", "author": "test-user"}

    async def test_put_object_already_exists(self, mock_client):
        """Test put_object raises KeyAlreadyExistsError when prompt exists."""
        mock_client.create_prompt = AsyncMock(side_effect=LangSmithConflictError("Conflict"))

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                )
                with pytest.raises(KeyAlreadyExistsError):
                    await store.put_object("test-prompt", item)

    async def test_put_object_invalid_key(self, mock_client):
        """Test put_object raises ValueError for invalid key."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                )
                with pytest.raises(ValueError, match="Invalid prompt key"):
                    await store.put_object("Invalid Key!", item)


class TestLangSmithPromptStoreUpsertObject:
    """Tests for LangSmithPromptStore.upsert_object."""

    async def test_upsert_object_new(self, mock_client):
        """Test upsert_object creates new prompt when it doesn't exist."""
        mock_client.get_prompt = AsyncMock(return_value=None)
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={
                        "description": "Test prompt", "version": "1.0.0"
                    },
                )
                await store.upsert_object("test-prompt", item)

                mock_client.get_prompt.assert_called_once_with("test-prompt")
                mock_client.create_prompt.assert_called_once()
                mock_client.create_commit.assert_called_once()

    async def test_upsert_object_existing(self, mock_client, mock_prompt_info):
        """Test upsert_object updates existing prompt."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.update_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Updated answer: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={
                        "description": "Updated prompt", "version": "2.0.0"
                    },
                )
                await store.upsert_object("test-prompt", item)

                mock_client.get_prompt.assert_called_once_with("test-prompt")
                mock_client.update_prompt.assert_called_once()
                mock_client.create_commit.assert_called_once()

    async def test_upsert_object_content_unchanged(self, mock_client, mock_prompt_info):
        """Test upsert_object handles 409 when content unchanged."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.update_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock(side_effect=LangSmithConflictError("Nothing to commit"))

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Same content: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={"description": "Updated prompt"},
                )
                # Should not raise - 409 with "Nothing to commit" is handled gracefully
                await store.upsert_object("test-prompt", item)

    async def test_upsert_object_invalid_key(self, mock_client):
        """Test upsert_object raises ValueError for invalid key."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                )
                with pytest.raises(ValueError, match="Invalid prompt key"):
                    await store.upsert_object("Invalid Key!", item)


class TestLangSmithPromptStoreGetObject:
    """Tests for LangSmithPromptStore.get_object."""

    async def test_get_object_success(self, mock_client, mock_prompt_info, mock_commit):
        """Test successful get_object with versioned metadata from manifest."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.pull_prompt_commit = AsyncMock(return_value=mock_commit)

        # Mock the deserialized prompt template with versioned metadata
        mock_prompt_template = MagicMock()
        mock_prompt_template.metadata = {"version": "1.0.0", "author": "test-user"}

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client), \
             patch("nat.plugins.langchain.langsmith_prompt_store.lc_load", return_value=mock_prompt_template):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = await store.get_object("test-prompt")

                assert item.content_type == "application/json"
                # Non-versioned metadata from prompt-level info
                assert item.metadata["description"] == "Test description"
                assert item.metadata["readme"] == "# Test Readme"
                # Tags are JSON-serialized for ObjectStoreItem string compatibility
                assert json.loads(item.metadata["tags"]) == ["production", "approved"]
                # Versioned metadata from ChatPromptTemplate.metadata (via lc_load)
                assert item.metadata["version"] == "1.0.0"
                assert item.metadata["author"] == "test-user"

    async def test_get_object_not_found(self, mock_client):
        """Test get_object raises NoSuchKeyError when prompt doesn't exist."""
        mock_client.get_prompt = AsyncMock(return_value=None)

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                with pytest.raises(NoSuchKeyError):
                    await store.get_object("nonexistent-prompt")

    async def test_get_object_invalid_key(self, mock_client):
        """Test get_object raises ValueError for invalid key."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                with pytest.raises(ValueError, match="Invalid prompt key"):
                    await store.get_object("Invalid Key!")

    async def test_get_object_lc_load_failure_graceful(self, mock_client, mock_prompt_info, mock_commit):
        """Test get_object handles lc_load failure gracefully (still returns item)."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.pull_prompt_commit = AsyncMock(return_value=mock_commit)

        # Mock lc_load to raise a caught exception type (TypeError, ValueError, KeyError, AttributeError)
        lc_load_patch = patch(
            "nat.plugins.langchain.langsmith_prompt_store.lc_load",
            side_effect=TypeError("Deserialization failed"),
        )
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client), \
             lc_load_patch:
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = await store.get_object("test-prompt")

                # Should still return item with prompt-level metadata
                assert item.content_type == "application/json"
                assert item.metadata["description"] == "Test description"
                assert item.metadata["readme"] == "# Test Readme"
                # Versioned metadata should be missing (lc_load failed)
                assert "version" not in item.metadata
                assert "author" not in item.metadata


class TestLangSmithPromptStoreDeleteObject:
    """Tests for LangSmithPromptStore.delete_object."""

    async def test_delete_object_success(self, mock_client, mock_prompt_info):
        """Test successful delete_object."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.delete_prompt = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                await store.delete_object("test-prompt")

                mock_client.delete_prompt.assert_called_once_with("test-prompt")

    async def test_delete_object_not_found(self, mock_client):
        """Test delete_object raises NoSuchKeyError when prompt doesn't exist."""
        mock_client.get_prompt = AsyncMock(return_value=None)

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                with pytest.raises(NoSuchKeyError):
                    await store.delete_object("nonexistent-prompt")

    async def test_delete_object_invalid_key(self, mock_client):
        """Test delete_object raises ValueError for invalid key."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                with pytest.raises(ValueError, match="Invalid prompt key"):
                    await store.delete_object("Invalid Key!")


class TestLangSmithPromptStoreMetadata:
    """Tests for metadata extraction and handling."""

    async def test_metadata_with_default_tags(self, mock_client):
        """Test that default_tags are merged with user tags."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(
                    api_key="test-key",
                    default_tags=["default-tag", "env-test"],
            ) as store:
                prompt_str = "Answer: {question}"
                # Note: ObjectStoreItem.metadata requires dict[str, str]
                # Tags must be JSON-serialized
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={
                        "tags": json.dumps(["user-tag"]), "version": "1.0.0"
                    },
                )
                await store.put_object("test-prompt", item)

                # Check that create_prompt was called with merged tags
                call_kwargs = mock_client.create_prompt.call_args[1]
                tags = call_kwargs.get("tags") or []
                assert "default-tag" in tags
                assert "env-test" in tags
                assert "user-tag" in tags

    async def test_metadata_reserved_keys_separated(self, mock_client):
        """Test that reserved metadata keys are stored separately from versioned metadata."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                # Note: ObjectStoreItem.metadata requires dict[str, str]
                # Tags must be JSON-serialized
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={
                        "description": "Test description",
                        "readme": "# Test Readme",
                        "tags": json.dumps(["production"]),
                        "content_type": "text/plain",  # Reserved, should be excluded from versioned
                        "version": "1.0.0",  # Custom, should be in versioned metadata
                        "author": "test-user",  # Custom, should be in versioned metadata
                    },
                )
                await store.put_object("test-prompt", item)

                # Check prompt-level metadata
                create_call = mock_client.create_prompt.call_args[1]
                assert create_call.get("description") == "Test description"
                assert create_call.get("readme") == "# Test Readme"
                assert "production" in (create_call.get("tags") or [])

                # Check versioned metadata in ChatPromptTemplate
                commit_call = mock_client.create_commit.call_args
                prompt_template = commit_call.kwargs.get("object") or commit_call.args[1]
                # Only custom (non-reserved) keys should be in versioned metadata
                assert prompt_template.metadata == {"version": "1.0.0", "author": "test-user"}
                # Reserved keys should NOT be in versioned metadata
                assert "description" not in prompt_template.metadata
                assert "readme" not in prompt_template.metadata
                assert "tags" not in prompt_template.metadata
                assert "content_type" not in prompt_template.metadata

    async def test_tags_can_be_string(self, mock_client):
        """Test that tags can be provided as a single string."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={"tags": "single-tag"},  # String instead of list
                )
                await store.put_object("test-prompt", item)

                call_kwargs = mock_client.create_prompt.call_args[1]
                tags = call_kwargs.get("tags") or []
                assert "single-tag" in tags

    async def test_tags_json_round_trip(self, mock_client):
        """Test that JSON-serialized tags from get_object can be used in put_object."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                prompt_str = "Answer: {question}"
                # Simulate tags that came from a previous get_object (JSON string)
                item = ObjectStoreItem(
                    data=prompt_str.encode("utf-8"),
                    content_type="text/plain",
                    metadata={"tags": '["tag1", "tag2"]'},  # JSON string from round-trip
                )
                await store.put_object("test-prompt", item)

                call_kwargs = mock_client.create_prompt.call_args[1]
                tags = call_kwargs.get("tags") or []
                # Should be parsed as list, not treated as single tag
                assert "tag1" in tags
                assert "tag2" in tags
                assert '["tag1", "tag2"]' not in tags  # Should not be the raw JSON string

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
from pydantic import ValidationError

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.models import ObjectStoreItem
from nat.plugins.langchain.langsmith_prompt_store import LangSmithPromptStore
from nat.plugins.langchain.langsmith_prompt_store import LangSmithPromptStoreConfig
from nat.plugins.langchain.langsmith_prompt_store import PromptMetadata
from nat.plugins.langchain.langsmith_prompt_store import _validate_prompt_key
from nat.plugins.langchain.langsmith_prompt_store import _validate_tag
from nat.plugins.langchain.langsmith_prompt_store import _validate_tags

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(name="sample_prompt_manifest")
def fixture_sample_prompt_manifest() -> dict:
    """Sample prompt manifest for testing."""
    return {
        "id": ["langchain", "prompts", "chat", "ChatPromptTemplate"],
        "lc": 1,
        "type": "constructor",
        "kwargs": {
            "input_variables": ["question"],
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
    prompt.tags = ["version:1.0.0", "author:test-user"]
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


class TestValidateTag:
    """Tests for _validate_tag function."""

    def test_valid_simple_tags(self):
        """Test that valid simple tags pass validation."""
        valid_tags = ["mytag", "my-tag", "my_tag", "tag123"]
        for tag in valid_tags:
            assert _validate_tag(tag) is True

    def test_valid_key_value_tags(self):
        """Test that valid key:value tags pass validation."""
        valid_tags = ["version:1.0.0", "author:test-user", "env:prod_v1"]
        for tag in valid_tags:
            assert _validate_tag(tag) is True

    def test_invalid_simple_tags(self):
        """Test that invalid simple tags fail validation."""
        invalid_tags = ["my tag", "tag!", "tag.name"]
        for tag in invalid_tags:
            assert _validate_tag(tag) is False

    def test_invalid_key_value_tags(self):
        """Test that invalid key:value tags fail validation."""
        invalid_tags = ["bad key:value", "key:bad value", "key!:value"]
        for tag in invalid_tags:
            assert _validate_tag(tag) is False


class TestValidateTags:
    """Tests for _validate_tags function."""

    def test_valid_tags_list(self):
        """Test that a list of valid tags passes validation."""
        valid_tags = ["mytag", "version:1.0.0", "author:test"]
        _validate_tags(valid_tags)  # Should not raise

    def test_invalid_tag_in_list(self):
        """Test that an invalid tag in the list raises ValueError."""
        invalid_tags = ["valid-tag", "invalid tag"]
        with pytest.raises(ValueError, match="Invalid tag"):
            _validate_tags(invalid_tags)

    def test_empty_list(self):
        """Test that an empty list passes validation."""
        _validate_tags([])  # Should not raise


# =============================================================================
# Tests for PromptMetadata Model
# =============================================================================


class TestPromptMetadata:
    """Tests for PromptMetadata Pydantic model."""

    def test_create_with_defaults(self):
        """Test creating PromptMetadata with default values."""
        metadata = PromptMetadata()
        assert metadata.description is None
        assert metadata.readme is None
        assert metadata.custom == {}

    def test_create_with_values(self):
        """Test creating PromptMetadata with explicit values."""
        metadata = PromptMetadata(
            description="Test description",
            readme="# Readme",
            custom={
                "version": "1.0.0", "author": "test"
            },
        )
        assert metadata.description == "Test description"
        assert metadata.readme == "# Readme"
        assert metadata.custom == {"version": "1.0.0", "author": "test"}

    def test_invalid_custom_key(self):
        """Test that invalid custom keys raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid metadata key"):
            PromptMetadata(custom={"invalid key": "value"})

    def test_invalid_custom_value(self):
        """Test that invalid custom values raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid metadata value"):
            PromptMetadata(custom={"key": "invalid value"})

    def test_to_tags(self):
        """Test converting custom metadata to tags."""
        metadata = PromptMetadata(custom={"version": "1.0.0", "author": "test"})
        tags = metadata.to_tags()
        assert set(tags) == {"version:1.0.0", "author:test"}

    def test_from_tags_with_key_value(self):
        """Test creating PromptMetadata from key:value tags."""
        tags = ["version:1.0.0", "author:test-user", "simple-tag"]
        metadata = PromptMetadata.from_tags(
            tags=tags,
            description="Test",
            readme="# Readme",
        )
        assert metadata.description == "Test"
        assert metadata.readme == "# Readme"
        assert metadata.custom == {"version": "1.0.0", "author": "test-user"}

    def test_from_tags_skips_invalid(self):
        """Test that from_tags skips invalid tags when validate=True."""
        tags = ["valid:tag", "invalid key:value"]
        metadata = PromptMetadata.from_tags(tags=tags, validate=True)
        # Invalid tag should be skipped
        assert "invalid key" not in metadata.custom
        assert metadata.custom.get("valid") == "tag"

    def test_from_tags_none(self):
        """Test from_tags with None tags."""
        metadata = PromptMetadata.from_tags(tags=None)
        assert metadata.custom == {}


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

    def test_with_valid_default_tags(self):
        """Test config with valid default_tags."""
        config = LangSmithPromptStoreConfig(default_tags=["env:prod", "team:ml"])
        assert config.default_tags == ["env:prod", "team:ml"]

    def test_with_invalid_default_tags(self):
        """Test that invalid default_tags raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid tag"):
            LangSmithPromptStoreConfig(default_tags=["invalid tag"])


# =============================================================================
# Tests for LangSmithPromptStore
# =============================================================================


class TestLangSmithPromptStoreInit:
    """Tests for LangSmithPromptStore initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        store = LangSmithPromptStore()
        assert store._api_key is None
        assert store._api_url is None
        assert store._is_public is False
        assert store._default_tags == []

    def test_init_with_values(self):
        """Test initialization with explicit values."""
        store = LangSmithPromptStore(
            api_key="test-key",
            api_url="https://api.test.com",
            is_public=True,
            default_tags=["env:test"],
        )
        assert store._api_key == "test-key"
        assert store._api_url == "https://api.test.com"
        assert store._is_public is True
        assert store._default_tags == ["env:test"]

    def test_init_with_invalid_default_tags(self):
        """Test that invalid default_tags raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tag"):
            LangSmithPromptStore(default_tags=["invalid tag"])


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

    async def test_put_object_success(self, sample_prompt_manifest, mock_client):
        """Test successful put_object."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                    metadata={
                        "description": "Test prompt", "version": "1.0.0"
                    },
                )
                await store.put_object("test-prompt", item)

                mock_client.create_prompt.assert_called_once()
                mock_client.create_commit.assert_called_once()

    async def test_put_object_already_exists(self, sample_prompt_manifest, mock_client):
        """Test put_object raises KeyAlreadyExistsError when prompt exists."""
        mock_client.create_prompt = AsyncMock(side_effect=LangSmithConflictError("Conflict"))

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                )
                with pytest.raises(KeyAlreadyExistsError):
                    await store.put_object("test-prompt", item)

    async def test_put_object_invalid_key(self, sample_prompt_manifest, mock_client):
        """Test put_object raises ValueError for invalid key."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                )
                with pytest.raises(ValueError, match="Invalid prompt key"):
                    await store.put_object("Invalid Key!", item)


class TestLangSmithPromptStoreUpsertObject:
    """Tests for LangSmithPromptStore.upsert_object."""

    async def test_upsert_object_new(self, sample_prompt_manifest, mock_client):
        """Test upsert_object creates new prompt when it doesn't exist."""
        mock_client.get_prompt = AsyncMock(return_value=None)
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                    metadata={"description": "Test prompt"},
                )
                await store.upsert_object("test-prompt", item)

                mock_client.get_prompt.assert_called_once_with("test-prompt")
                mock_client.create_prompt.assert_called_once()
                mock_client.create_commit.assert_called_once()

    async def test_upsert_object_existing(self, sample_prompt_manifest, mock_client, mock_prompt_info):
        """Test upsert_object updates existing prompt."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.update_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                    metadata={"description": "Updated prompt"},
                )
                await store.upsert_object("test-prompt", item)

                mock_client.get_prompt.assert_called_once_with("test-prompt")
                mock_client.update_prompt.assert_called_once()
                mock_client.create_commit.assert_called_once()

    async def test_upsert_object_content_unchanged(self, sample_prompt_manifest, mock_client, mock_prompt_info):
        """Test upsert_object handles 409 when content unchanged."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.update_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock(side_effect=LangSmithConflictError("Nothing to commit"))

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                    metadata={"description": "Updated prompt"},
                )
                # Should not raise - 409 with "Nothing to commit" is handled gracefully
                await store.upsert_object("test-prompt", item)

    async def test_upsert_object_invalid_key(self, sample_prompt_manifest, mock_client):
        """Test upsert_object raises ValueError for invalid key."""
        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                )
                with pytest.raises(ValueError, match="Invalid prompt key"):
                    await store.upsert_object("Invalid Key!", item)


class TestLangSmithPromptStoreGetObject:
    """Tests for LangSmithPromptStore.get_object."""

    async def test_get_object_success(self, mock_client, mock_prompt_info, mock_commit):
        """Test successful get_object."""
        mock_client.get_prompt = AsyncMock(return_value=mock_prompt_info)
        mock_client.pull_prompt_commit = AsyncMock(return_value=mock_commit)

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = await store.get_object("test-prompt")

                assert item.content_type == "application/json"
                assert item.metadata["description"] == "Test description"
                assert item.metadata["readme"] == "# Test Readme"
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
    """Tests for metadata extraction and building."""

    async def test_metadata_with_default_tags(self, sample_prompt_manifest, mock_client):
        """Test that default_tags are included in metadata."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(
                    api_key="test-key",
                    default_tags=["env:test", "team:ml"],
            ) as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                    metadata={"version": "1.0.0"},
                )
                await store.put_object("test-prompt", item)

                # Check that create_prompt was called with tags including defaults
                call_kwargs = mock_client.create_prompt.call_args[1]
                tags = call_kwargs.get("tags") or []
                assert "env:test" in tags
                assert "team:ml" in tags
                assert "version:1.0.0" in tags

    async def test_metadata_reserved_keys_handled(self, sample_prompt_manifest, mock_client):
        """Test that reserved metadata keys are handled correctly."""
        mock_client.create_prompt = AsyncMock()
        mock_client.create_commit = AsyncMock()

        with patch("nat.plugins.langchain.langsmith_prompt_store.AsyncClient", return_value=mock_client):
            async with LangSmithPromptStore(api_key="test-key") as store:
                item = ObjectStoreItem(
                    data=json.dumps(sample_prompt_manifest).encode("utf-8"),
                    content_type="application/json",
                    metadata={
                        "description": "Test description",
                        "readme": "# Test Readme",
                        "content_type": "application/json",  # Should be ignored
                        "version": "1.0.0",  # Should become tag
                    },
                )
                await store.put_object("test-prompt", item)

                call_kwargs = mock_client.create_prompt.call_args[1]
                assert call_kwargs.get("description") == "Test description"
                assert call_kwargs.get("readme") == "# Test Readme"
                tags = call_kwargs.get("tags") or []
                assert "version:1.0.0" in tags
                # content_type should not be in tags
                assert not any("content_type" in tag for tag in tags)

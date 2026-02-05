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
"""
LangSmith Prompt Store implementation for NAT Object Store interface.

This module provides an ObjectStore implementation that uses LangSmith's
prompt management capabilities as the underlying storage backend.

Metadata Handling:
    The store maps ObjectStoreItem metadata to LangSmith's prompt fields:

    Reserved keys (stored as native LangSmith fields):
        - "description": Brief description of the prompt
        - "readme": Longer-form documentation/readme
        - "content_type": Content type (stored in returned metadata only)

    Custom metadata (stored as LangSmith tags in 'key:value' format):
        - Any other key-value pairs in metadata become tags
        - Keys must be alphanumeric with underscores or hyphens
        - Values must be alphanumeric with underscores, hyphens, or dots

    Example:
        metadata = {
            "description": "My prompt",      # -> LangSmith description field
            "readme": "# Docs...",           # -> LangSmith readme field
            "version": "1.0.0",              # -> tag: "version:1.0.0"
            "author": "mpenn",               # -> tag: "author:mpenn"
        }
"""

import asyncio
import json
import logging
import os
import re
from typing import Any
from typing import ClassVar

from langchain_core.prompts import ChatPromptTemplate
from langsmith import AsyncClient
from langsmith.utils import LangSmithConflictError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.common import OptionalSecretStr
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

# Reserved metadata keys that are handled specially (not converted to tags)
RESERVED_METADATA_KEYS = frozenset[str]({"description", "readme", "content_type"})

# Validation patterns for tag key:value format
TAG_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
TAG_VALUE_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

# Validation pattern for LangSmith prompt keys (handles)
# Must be lowercase alphanumeric, hyphen, or underscore, starting with a-z
PROMPT_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _validate_prompt_key(key: str) -> None:
    """
    Validate that a prompt key conforms to LangSmith requirements.

    Args:
        key: The prompt key/handle to validate.

    Raises:
        ValueError: If the key is invalid.
    """
    if not PROMPT_KEY_PATTERN.match(key):
        raise ValueError(f"Invalid prompt key '{key}': must be lowercase alphanumeric with hyphens "
                         "or underscores, starting with a letter (a-z). "
                         "Example: 'my-prompt', 'test_prompt_v1'")


def _validate_tag(tag: str) -> bool:
    """
    Validate that a tag conforms to the expected format.

    Tags can be either:
    - Simple tags: alphanumeric with underscores/hyphens
    - Key-value tags: "key:value" format

    Args:
        tag: The tag string to validate.

    Returns:
        True if valid, False otherwise.
    """
    if ":" in tag:
        key, value = tag.split(":", 1)
        return bool(TAG_KEY_PATTERN.match(key) and TAG_VALUE_PATTERN.match(value))
    # Simple tags should match key pattern
    return bool(TAG_KEY_PATTERN.match(tag))


def _validate_tags(tags: list[str], source: str = "tags") -> None:
    """
    Validate a list of tags.

    Args:
        tags: List of tags to validate.
        source: Description of where tags came from (for error messages).

    Raises:
        ValueError: If any tag is invalid.
    """
    for tag in tags:
        if not _validate_tag(tag):
            raise ValueError(f"Invalid tag '{tag}' in {source}: must be alphanumeric with underscores/hyphens, "
                             "or 'key:value' format where key is alphanumeric with _/- and value is "
                             "alphanumeric with _/-/.")


class PromptMetadata(BaseModel):
    """
    Validated metadata for LangSmith prompts.

    Reserved fields (description, readme) are stored directly in LangSmith.
    Custom key-value pairs are converted to tags in the format 'key:value'.

    Attributes:
        description: Brief description of the prompt.
        readme: Longer-form documentation/readme for the prompt.
        custom: Dictionary of custom key-value pairs to store as tags.
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, description="Brief description of the prompt.")
    readme: str | None = Field(default=None, description="Longer-form documentation for the prompt.")
    custom: dict[str, str] = Field(
        default_factory=dict,
        description="Custom key-value pairs stored as 'key:value' tags.",
    )

    @field_validator("custom")
    @classmethod
    def validate_custom_tags(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that custom metadata keys and values conform to tag format."""
        for key, value in v.items():
            if not TAG_KEY_PATTERN.match(key):
                raise ValueError(f"Invalid metadata key '{key}': must contain only alphanumeric characters, "
                                 "underscores, or hyphens")
            if not TAG_VALUE_PATTERN.match(value):
                raise ValueError(f"Invalid metadata value '{value}' for key '{key}': must contain only "
                                 "alphanumeric characters, underscores, hyphens, or dots")
        return v

    def to_tags(self) -> list[str]:
        """Convert custom metadata to list of 'key:value' tags."""
        return [f"{k}:{v}" for k, v in self.custom.items()]

    @classmethod
    def from_tags(
        cls,
        tags: list[str] | None,
        description: str | None = None,
        readme: str | None = None,
        validate: bool = True,
    ) -> "PromptMetadata":
        """
        Create PromptMetadata from LangSmith tags and fields.

        Args:
            tags: List of tags, some of which may be 'key:value' format.
            description: Prompt description.
            readme: Prompt readme.
            validate: Whether to validate tags. Set to False to skip validation
                for tags from trusted sources (e.g., LangSmith API responses).

        Returns:
            PromptMetadata instance with parsed custom tags.

        Raises:
            ValueError: If validate=True and any tag has invalid format.
        """
        custom: dict[str, str] = {}
        if tags:
            for tag in tags:
                if ":" in tag:
                    key, value = tag.split(":", 1)
                    # Validate key:value format if requested
                    if validate:
                        if not TAG_KEY_PATTERN.match(key):
                            logger.warning("Skipping tag with invalid key '%s' from LangSmith", key)
                            continue
                        if not TAG_VALUE_PATTERN.match(value):
                            logger.warning("Skipping tag '%s' with invalid value from LangSmith", tag)
                            continue
                    custom[key] = value
        return cls(description=description, readme=readme, custom=custom)


class LangSmithPromptStoreConfig(ObjectStoreBaseConfig, name="langsmith_prompt_store"):
    """
    Object store that stores prompt templates in LangSmith.

    This allows storing and retrieving prompt templates using the LangSmith
    prompt management API, enabling version control and collaboration on prompts.
    """

    API_KEY_ENV: ClassVar[str] = "LANGSMITH_API_KEY"
    API_URL_ENV: ClassVar[str] = "LANGSMITH_API_URL"

    api_key: OptionalSecretStr = Field(
        default=None,
        description=f"LangSmith API key. If omitted, reads from {API_KEY_ENV} environment variable.",
    )
    api_url: str | None = Field(
        default=None,
        description=f"LangSmith API URL. If omitted, reads from {API_URL_ENV} or uses default.",
    )
    is_public: bool = Field(
        default=False,
        description="Whether prompts should be public by default.",
    )
    default_tags: list[str] | None = Field(
        default=None,
        description="Default tags to apply to all prompts. Must be valid tag format.",
    )

    @field_validator("default_tags")
    @classmethod
    def validate_default_tags(cls, v: list[str] | None) -> list[str] | None:
        """Validate that default_tags conform to tag format."""
        if v is not None:
            _validate_tags(v, source="default_tags")
        return v


class LangSmithPromptStore(ObjectStore):
    """
    ObjectStore implementation that uses LangSmith for prompt template storage.

    This store maps the ObjectStore interface to LangSmith's prompt management API:
    - key: prompt name in LangSmith (lowercase alphanumeric, hyphens, underscores)
    - data: JSON-serialized prompt template manifest
    - metadata: stored as prompt description, readme, and custom key:value tags
    - content_type: stored in returned metadata (typically "application/json")

    Metadata Mapping:
        Reserved keys are stored as native LangSmith fields:
        - "description" -> LangSmith description
        - "readme" -> LangSmith readme

        All other metadata keys become tags in "key:value" format:
        - "version": "1.0.0" -> tag "version:1.0.0"
        - "author": "mpenn" -> tag "author:mpenn"

    Usage:
        Must be used as an async context manager to manage the client connection:

        async with LangSmithPromptStore(api_key="...") as store:
            await store.put_object("my-prompt", item)
            retrieved = await store.get_object("my-prompt")

    The store uses LangSmith's AsyncClient for all operations.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        is_public: bool = False,
        default_tags: list[str] | None = None,
    ) -> None:
        """
        Initialize the LangSmith Prompt Store.

        Args:
            api_key: LangSmith API key. Falls back to LANGSMITH_API_KEY env var.
            api_url: LangSmith API URL. Falls back to LANGSMITH_API_URL env var.
            is_public: Whether prompts should be public by default.
            default_tags: Default tags to apply to all prompts. Must be valid tag format.

        Raises:
            ValueError: If default_tags contains invalid tag format.
        """
        super().__init__()

        # Validate default_tags if provided
        if default_tags:
            _validate_tags(default_tags, source="default_tags")

        self._api_key = api_key or os.environ.get(LangSmithPromptStoreConfig.API_KEY_ENV)
        self._api_url = api_url or os.environ.get(LangSmithPromptStoreConfig.API_URL_ENV)
        self._is_public = is_public
        self._default_tags = default_tags or []
        self._client: AsyncClient | None = None
        # Lock to prevent race conditions in upsert_object
        self._upsert_lock = asyncio.Lock()

    async def __aenter__(self) -> "LangSmithPromptStore":
        """Initialize the async client connection."""
        if self._client is not None:
            raise RuntimeError("Connection already established")

        client_kwargs: dict[str, Any] = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._api_url:
            client_kwargs["api_url"] = self._api_url

        self._client = AsyncClient(**client_kwargs)
        logger.info("LangSmith prompt store connection established")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Close the async client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("LangSmith prompt store connection closed")

    def _ensure_connected(self) -> AsyncClient:
        """Ensure the client is connected and return it."""
        if self._client is None:
            raise RuntimeError("Connection not established. Use 'async with' context manager.")
        return self._client

    def _extract_metadata(self, item: ObjectStoreItem) -> tuple[str | None, str | None, list[str]]:
        """
        Extract and validate metadata from ObjectStoreItem.

        Parses the metadata dict, separating reserved fields (description, readme)
        from custom key-value pairs. Custom pairs are validated and converted to
        'key:value' tags.

        Args:
            item: The ObjectStoreItem containing metadata.

        Returns:
            Tuple of (description, readme, tags).

        Raises:
            ValueError: If custom metadata keys or values have invalid format.
        """
        raw_metadata = item.metadata or {}

        # Extract reserved fields
        description = raw_metadata.get("description")
        readme = raw_metadata.get("readme")

        # Extract custom fields (everything that's not reserved)
        custom = {k: v for k, v in raw_metadata.items() if k not in RESERVED_METADATA_KEYS}

        # Validate using Pydantic model
        prompt_metadata = PromptMetadata(
            description=description,
            readme=readme,
            custom=custom,
        )

        # Convert custom metadata to tags and merge with defaults
        tags = prompt_metadata.to_tags()
        tags = list(set(tags + self._default_tags))

        return prompt_metadata.description, prompt_metadata.readme, tags

    def _build_metadata(
        self,
        description: str | None,
        readme: str | None,
        tags: list[str] | None,
        content_type: str | None,
    ) -> dict[str, str]:
        """
        Build metadata dict from LangSmith prompt info.

        Parses 'key:value' tags back into custom metadata fields.

        Args:
            description: Prompt description.
            readme: Prompt readme (longer-form documentation).
            tags: Prompt tags (may include 'key:value' formatted custom metadata).
            content_type: Content type of the data.

        Returns:
            Metadata dictionary with reserved fields and custom key-value pairs.
        """
        # Parse tags back into PromptMetadata
        prompt_metadata = PromptMetadata.from_tags(
            tags=tags,
            description=description,
            readme=readme,
        )

        # Build the metadata dict
        metadata: dict[str, str] = {}
        if prompt_metadata.description:
            metadata["description"] = prompt_metadata.description
        if prompt_metadata.readme:
            metadata["readme"] = prompt_metadata.readme
        if content_type:
            metadata["content_type"] = content_type

        # Add custom metadata as individual keys
        metadata.update(prompt_metadata.custom)

        return metadata

    @staticmethod
    def _build_chat_prompt_template(prompt: str):
        """
        Build a ChatPromptTemplate from a prompt string.
        """
        return ChatPromptTemplate.from_messages(("human", prompt))

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save a prompt template in LangSmith with the given key (prompt name).

        If the key already exists, raise KeyAlreadyExistsError.

        Uses create_prompt + create_commit instead of push_prompt to avoid
        redundant existence checks (which cause 404 logs for new prompts).

        Args:
            key: The prompt name to save the item under.
            item: The ObjectStoreItem containing the prompt data.

        Raises:
            ValueError: If the key format is invalid.
            KeyAlreadyExistsError: If a prompt with this name already exists.
        """
        _validate_prompt_key(key)
        client = self._ensure_connected()

        description, readme, tags = self._extract_metadata(item)

        # Decode the prompt manifest from bytes
        prompt_manifest = str(item.data.decode("utf-8"))
        prompt_manifest = LangSmithPromptStore._build_chat_prompt_template(prompt=prompt_manifest)

        # Try to create the prompt directly - will fail with 409 if it exists
        try:
            await client.create_prompt(
                key,
                description=description,
                readme=readme,
                tags=tags or None,
                is_public=self._is_public,
            )
        except LangSmithConflictError:
            raise KeyAlreadyExistsError(
                key=key,
                additional_message=f"LangSmith prompt '{key}' already exists.",
            )

        # Add the prompt content as the first commit
        await client.create_commit(key, object=prompt_manifest)
        logger.info("Created LangSmith prompt: %s", key)

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save or update a prompt template in LangSmith.

        If the key already exists, update metadata and create a new version.
        Otherwise, create a new prompt with initial content.

        Uses direct create_prompt/update_prompt + create_commit calls instead of
        push_prompt to reduce redundant API calls (stateless optimization).

        Uses a lock to serialize concurrent upserts and prevent race conditions.

        Args:
            key: The prompt name to save the item under.
            item: The ObjectStoreItem containing the prompt data.

        Raises:
            ValueError: If the key format is invalid.
        """
        _validate_prompt_key(key)
        client = self._ensure_connected()

        # Use lock to prevent race conditions between concurrent upserts
        async with self._upsert_lock:
            description, readme, tags = self._extract_metadata(item)

            # Decode the prompt manifest from bytes
            prompt_manifest = str(item.data.decode("utf-8"))
            prompt_manifest = LangSmithPromptStore._build_chat_prompt_template(prompt=prompt_manifest)

            # Check if prompt exists
            existing = await client.get_prompt(key)

            if existing is not None:
                # Update metadata for existing prompt
                if description or readme or tags:
                    await client.update_prompt(
                        key,
                        description=description,
                        readme=readme,
                        tags=tags or None,
                    )
            else:
                # Create new prompt (without content - that comes next)
                await client.create_prompt(
                    key,
                    description=description,
                    readme=readme,
                    tags=tags or None,
                    is_public=self._is_public,
                )

            # Create commit with the content
            # May return 409 if content unchanged (expected for metadata-only updates)
            try:
                await client.create_commit(key, object=prompt_manifest)
                logger.info("Upserted LangSmith prompt: %s", key)
            except LangSmithConflictError as e:
                if "Nothing to commit" in str(e):
                    logger.info("LangSmith prompt '%s' content unchanged, metadata updated only", key)
                else:
                    raise

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        """
        Get a prompt template from LangSmith by key (prompt name).

        Args:
            key: The prompt name to retrieve.

        Returns:
            ObjectStoreItem containing the prompt data and metadata.

        Raises:
            ValueError: If the key format is invalid.
            NoSuchKeyError: If the prompt does not exist.
        """
        _validate_prompt_key(key)
        client = self._ensure_connected()

        # Get prompt metadata
        prompt_info = await client.get_prompt(key)
        if prompt_info is None:
            raise NoSuchKeyError(
                key=key,
                additional_message=f"LangSmith prompt '{key}' not found.",
            )

        # Pull the prompt commit to get the manifest
        commit = await client.pull_prompt_commit(key)
        manifest = commit.manifest

        # Serialize manifest to bytes
        data = json.dumps(manifest).encode("utf-8")

        # Build metadata from prompt info
        # The LangSmith SDK's Prompt type may not always have 'readme' as a defined attribute
        # depending on SDK version, so we use getattr for safe access
        readme = getattr(prompt_info, "readme", None)
        metadata = self._build_metadata(
            description=prompt_info.description,
            readme=readme,
            tags=prompt_info.tags,
            content_type="application/json",
        )

        return ObjectStoreItem(
            data=data,
            content_type="application/json",
            metadata=metadata,
        )

    @override
    async def delete_object(self, key: str) -> None:
        """
        Delete a prompt template from LangSmith by key (prompt name).

        Args:
            key: The prompt name to delete.

        Raises:
            ValueError: If the key format is invalid.
            NoSuchKeyError: If the prompt does not exist.
        """
        _validate_prompt_key(key)
        client = self._ensure_connected()

        # Check if prompt exists
        existing = await client.get_prompt(key)
        if existing is None:
            raise NoSuchKeyError(
                key=key,
                additional_message=f"LangSmith prompt '{key}' not found.",
            )

        await client.delete_prompt(key)
        logger.info("Deleted LangSmith prompt: %s", key)


@register_object_store(config_type=LangSmithPromptStoreConfig)
async def langsmith_prompt_store(config: LangSmithPromptStoreConfig, _builder: Builder):
    """
    Factory function for creating a LangSmithPromptStore instance.

    This function is registered as an object store provider and handles
    the lifecycle of the LangSmith connection.

    Args:
        config: Configuration for the LangSmith prompt store.
        _builder: NAT Builder instance (unused).

    Yields:
        Configured LangSmithPromptStore instance.
    """
    # Extract secret values
    api_key = config.api_key.get_secret_value() if config.api_key else None

    async with LangSmithPromptStore(
            api_key=api_key,
            api_url=config.api_url,
            is_public=config.is_public,
            default_tags=config.default_tags,
    ) as store:
        yield store

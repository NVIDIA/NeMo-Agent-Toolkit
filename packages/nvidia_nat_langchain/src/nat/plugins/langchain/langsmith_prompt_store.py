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
    The store uses a clean separation of concerns for metadata:

    1. VERSIONED METADATA (via ChatPromptTemplate.metadata):
       Custom key-value pairs are embedded directly in the prompt manifest.
       This metadata is versioned per-commit, so each version retains its
       original metadata even when newer versions are pushed.

    2. PROMPT-LEVEL FIELDS (not versioned):
       - "description": Brief description of the prompt (LangSmith field)
       - "readme": Longer-form documentation (LangSmith field)
       - "tags": Simple strings for categorization (LangSmith tags, UI-visible)

    Reserved metadata keys (stored as native LangSmith fields):
        - "description": Brief description of the prompt
        - "readme": Longer-form documentation/readme
        - "tags": List of simple tag strings for UI categorization
        - "content_type": Content type (stored in returned metadata only)

    Custom metadata (everything else):
        - Stored in ChatPromptTemplate.metadata (versioned, per-commit)
        - Retrieved from the manifest when pulling a specific version

    Example:
        metadata = {
            "description": "My prompt",      # -> LangSmith description field
            "readme": "# Docs...",           # -> LangSmith readme field
            "tags": ["production", "v1"],    # -> LangSmith prompt-level tags
            "version": "1.0.0",              # -> ChatPromptTemplate.metadata["version"]
            "author": "mpenn",               # -> ChatPromptTemplate.metadata["author"]
        }

    When pulling a prompt:
        - Versioned metadata is extracted from ChatPromptTemplate.metadata
        - This ensures you get the metadata from THAT specific version
        - Description/readme/tags come from prompt-level fields (always latest)
"""

import asyncio
import json
import logging
import os
import re
from typing import Any
from typing import ClassVar

from langchain_core.load import load as lc_load
from langchain_core.prompts import ChatPromptTemplate
from langsmith import AsyncClient
from langsmith.utils import LangSmithConflictError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

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

# Reserved metadata keys that are handled specially (not stored in ChatPromptTemplate.metadata)
RESERVED_METADATA_KEYS = frozenset[str]({"description", "readme", "tags", "content_type", "role"})

# Valid message roles for ChatPromptTemplate
VALID_ROLES = frozenset[str]({"system", "human", "assistant", "ai"})
DEFAULT_ROLE = "human"

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


class ExtractedMetadata(BaseModel):
    """Metadata extracted from ObjectStoreItem, separated by storage location."""

    model_config = ConfigDict(frozen=True)

    description: str | None = Field(default=None, description="Prompt-level description (not versioned).")
    readme: str | None = Field(default=None, description="Prompt-level readme/documentation (not versioned).")
    tags: list[str] = Field(default_factory=list, description="Prompt-level tags (not versioned, UI visible).")
    role: str = Field(default=DEFAULT_ROLE,
                      description="Message role for ChatPromptTemplate (system, human, assistant).")
    custom: dict[str, Any] = Field(default_factory=dict, description="Custom metadata for versioned storage.")


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
        description="Default tags to apply to all prompts (simple strings for categorization).",
    )
    timeout_ms: int | None = Field(
        default=None,
        description="Timeout in milliseconds for LangSmith API requests. If omitted, uses SDK default.",
    )


class LangSmithPromptStore(ObjectStore):
    """
    ObjectStore implementation that uses LangSmith for prompt template storage.

    This store maps the ObjectStore interface to LangSmith's prompt management API:
    - key: prompt name in LangSmith (lowercase alphanumeric, hyphens, underscores)
    - data: prompt template string (wrapped in ChatPromptTemplate)
    - metadata: split between prompt-level fields and versioned ChatPromptTemplate.metadata

    Metadata Mapping:
        Reserved keys are stored as native LangSmith fields (not versioned):
        - "description" -> LangSmith description
        - "readme" -> LangSmith readme
        - "tags" -> LangSmith prompt-level tags (simple strings)

        All other metadata keys are stored in ChatPromptTemplate.metadata (versioned):
        - "version": "1.0.0" -> ChatPromptTemplate.metadata["version"]
        - "author": "mpenn" -> ChatPromptTemplate.metadata["author"]

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
        timeout_ms: int | None = None,
    ) -> None:
        """
        Initialize the LangSmith Prompt Store.

        Args:
            api_key: LangSmith API key. Falls back to LANGSMITH_API_KEY env var.
            api_url: LangSmith API URL. Falls back to LANGSMITH_API_URL env var.
            is_public: Whether prompts should be public by default.
            default_tags: Default tags to apply to all prompts (simple strings).
            timeout_ms: Timeout in milliseconds for API requests. Uses SDK default if None.
        """
        super().__init__()

        self._api_key = api_key or os.environ.get(LangSmithPromptStoreConfig.API_KEY_ENV)
        self._api_url = api_url or os.environ.get(LangSmithPromptStoreConfig.API_URL_ENV)
        self._is_public = is_public
        self._default_tags = default_tags or []
        self._timeout_ms = timeout_ms
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
        if self._timeout_ms is not None:
            client_kwargs["timeout_ms"] = self._timeout_ms

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

    def _extract_metadata(self, item: ObjectStoreItem) -> ExtractedMetadata:
        """
        Extract metadata from ObjectStoreItem.

        Separates reserved fields (description, readme, tags) from custom
        key-value pairs. Custom pairs are stored in ChatPromptTemplate.metadata
        for versioned storage.

        Args:
            item: The ObjectStoreItem containing metadata.

        Returns:
            ExtractedMetadata with separated fields for prompt-level and versioned storage.
        """
        raw_metadata = item.metadata or {}

        # Extract reserved fields
        description = raw_metadata.get("description")
        readme = raw_metadata.get("readme")
        user_tags_raw = raw_metadata.get("tags", [])

        # Extract and validate role
        role = raw_metadata.get("role", DEFAULT_ROLE)
        if role not in VALID_ROLES:
            logger.warning("Invalid role '%s', falling back to '%s'", role, DEFAULT_ROLE)
            role = DEFAULT_ROLE

        # Parse tags - handle JSON string (from round-trip) or plain string
        # Note: ObjectStoreItem.metadata is dict[str, str], so tags will always be a string
        if isinstance(user_tags_raw, str):
            try:
                parsed = json.loads(user_tags_raw)
                user_tags = parsed if isinstance(parsed, list) else [user_tags_raw]
            except json.JSONDecodeError:
                # Treat as single tag if not valid JSON
                user_tags = [user_tags_raw]
        else:
            # Defensive: handle if caller passes non-string (e.g., in tests)
            user_tags = list(user_tags_raw) if user_tags_raw else []

        # Merge with default tags (set removes duplicates)
        tags = list(set(user_tags + self._default_tags))

        # Extract custom fields (everything that's not reserved) for versioned storage
        custom = {k: v for k, v in raw_metadata.items() if k not in RESERVED_METADATA_KEYS}

        return ExtractedMetadata(
            description=description,
            readme=readme,
            tags=tags,
            role=role,
            custom=custom,
        )

    @staticmethod
    def _build_chat_prompt_template(
        prompt: str,
        role: str = DEFAULT_ROLE,
        metadata: dict[str, Any] | None = None,
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate from a prompt string with optional metadata.

        The metadata is embedded directly in the ChatPromptTemplate, which means
        it will be serialized into the manifest and versioned per-commit.

        Args:
            prompt: The prompt template string.
            role: Message role (system, human, assistant). Defaults to "human".
            metadata: Optional dict of custom metadata to embed (versioned per-commit).

        Returns:
            ChatPromptTemplate with metadata attached.
        """
        template = ChatPromptTemplate.from_messages([(role, prompt)])
        if metadata:
            template.metadata = metadata
        return template

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Save a prompt template in LangSmith with the given key (prompt name).

        If the key already exists, raise KeyAlreadyExistsError.

        Uses create_prompt + create_commit instead of push_prompt to avoid
        redundant existence checks (which cause 404 logs for new prompts).

        Metadata handling:
        - description, readme, tags: Stored as prompt-level fields (not versioned)
        - All other metadata: Stored in ChatPromptTemplate.metadata (versioned)

        Args:
            key: The prompt name to save the item under.
            item: The ObjectStoreItem containing the prompt data.

        Raises:
            ValueError: If the key format is invalid.
            KeyAlreadyExistsError: If a prompt with this name already exists.
        """
        _validate_prompt_key(key)
        client = self._ensure_connected()

        meta = self._extract_metadata(item)

        # Decode the prompt string from bytes and build ChatPromptTemplate
        # Embed custom metadata in the template for versioned storage
        prompt_str = item.data.decode("utf-8")
        prompt_template = LangSmithPromptStore._build_chat_prompt_template(
            prompt=prompt_str,
            role=meta.role,
            # Empty dict {} is falsy, so no metadata attached if no custom fields
            metadata=meta.custom or None,
        )

        # Try to create the prompt directly - will fail with 409 if it exists
        try:
            await client.create_prompt(
                key,
                description=meta.description,
                readme=meta.readme,
                tags=meta.tags or None,
                is_public=self._is_public,
            )
        except LangSmithConflictError:
            raise KeyAlreadyExistsError(
                key=key,
                additional_message=f"LangSmith prompt '{key}' already exists.",
            )

        # Add the prompt content as the first commit (no commit tags needed)
        await client.create_commit(key, object=prompt_template)
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

        Metadata handling:
        - description, readme, tags: Stored as prompt-level fields (not versioned)
        - All other metadata: Stored in ChatPromptTemplate.metadata (versioned)

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
            meta = self._extract_metadata(item)

            # Decode the prompt string from bytes and build ChatPromptTemplate
            # Embed custom metadata in the template for versioned storage
            prompt_str = item.data.decode("utf-8")
            prompt_template = LangSmithPromptStore._build_chat_prompt_template(
                prompt=prompt_str,
                role=meta.role,
                # Empty dict {} is falsy, so no metadata attached if no custom fields
                metadata=meta.custom or None,
            )

            # Check if prompt exists
            existing = await client.get_prompt(key)

            if existing is not None:
                # Update metadata for existing prompt (prompt-level, for UI visibility)
                # Check for explicit values (None means not provided, empty string/list is valid)
                if meta.description is not None or meta.readme is not None or meta.tags:
                    await client.update_prompt(
                        key,
                        description=meta.description,
                        readme=meta.readme,
                        tags=meta.tags or None,
                    )
            else:
                # Create new prompt (without content - that comes next)
                await client.create_prompt(
                    key,
                    description=meta.description,
                    readme=meta.readme,
                    tags=meta.tags or None,
                    is_public=self._is_public,
                )

            # Create commit with the content (includes versioned metadata in manifest)
            # May return 409 if content unchanged (expected for metadata-only updates)
            try:
                await client.create_commit(key, object=prompt_template)
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

        Retrieves the prompt and extracts VERSIONED metadata from
        ChatPromptTemplate.metadata (stored in the manifest). This ensures
        you get the metadata from the specific version being pulled.

        Description, readme, and tags come from prompt-level fields (always latest).

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

        # Get prompt metadata (for description/readme/tags - not versioned)
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

        # Extract VERSIONED metadata from the ChatPromptTemplate
        # Use lc_load to deserialize the manifest back to a ChatPromptTemplate
        versioned_metadata: dict[str, Any] = {}
        extracted_role: str = DEFAULT_ROLE
        try:
            prompt_template = lc_load(manifest)
            versioned_metadata = getattr(prompt_template, "metadata", None) or {}

            # Extract role from the first message if available
            if hasattr(prompt_template, "messages") and prompt_template.messages:
                first_msg = prompt_template.messages[0]
                # MessageLikeRepresentation can be various types
                if hasattr(first_msg, "type"):
                    # Maps internal types to our role names
                    role_map = {"human": "human", "ai": "assistant", "system": "system", "assistant": "assistant"}
                    extracted_role = role_map.get(first_msg.type, DEFAULT_ROLE)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # Debug level since non-ChatPromptTemplate manifests are valid
            # These exceptions can occur when manifest structure doesn't match expected format
            logger.debug("Failed to deserialize prompt manifest for metadata extraction: %s", e)

        # Build metadata combining versioned (from manifest) and non-versioned (from prompt info)
        # All values must be strings for ObjectStoreItem compatibility
        metadata: dict[str, str] = {}
        if prompt_info.description:
            metadata["description"] = prompt_info.description
        if prompt_info.readme:
            metadata["readme"] = prompt_info.readme
        if prompt_info.tags:
            # Convert tags list to JSON string for storage
            metadata["tags"] = json.dumps(prompt_info.tags)
        metadata["content_type"] = "application/json"
        metadata["role"] = extracted_role

        # Add versioned custom metadata from ChatPromptTemplate.metadata
        # Convert non-string values to JSON strings for ObjectStoreItem compatibility
        for k, v in versioned_metadata.items():
            if isinstance(v, str):
                metadata[k] = v
            else:
                metadata[k] = json.dumps(v)

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
            timeout_ms=config.timeout_ms,
    ) as store:
        yield store

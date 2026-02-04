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
from datetime import datetime
from pathlib import Path
from typing import Protocol

from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem


class PromptStorage(Protocol):
    """
    Abstract interface for prompt optimizer storage operations.

    Provides domain-specific methods for saving and loading prompt
    checkpoints and final optimized prompts. Implementations can use
    any storage backend (filesystem, object store, database, etc.).
    """

    async def save_checkpoint(
        self,
        generation: int,
        prompts: dict[str, tuple[str, str]],
        fitness_score: float | None = None,
        evaluator_scores: dict[str, float] | None = None
    ) -> None:
        """
        Save generation checkpoint.

        Args:
            generation: Generation number (1-indexed)
            prompts: Map of param_name -> (prompt_text, purpose)
            fitness_score: Optional overall fitness score for this generation
            evaluator_scores: Optional map of evaluator_name -> score
        """
        ...

    async def save_final(
        self,
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """
        Save final optimized prompts.

        Args:
            prompts: Map of param_name -> (prompt_text, purpose)
        """
        ...

    async def load_checkpoint(
        self,
        generation: int
    ) -> dict[str, tuple[str, str]]:
        """
        Load specific checkpoint.

        Args:
            generation: Generation number to load

        Returns:
            Map of param_name -> (prompt_text, purpose)

        Raises:
            KeyError: If checkpoint not found
        """
        ...

    async def load_latest_checkpoint(
        self
    ) -> tuple[int, dict[str, tuple[str, str]]]:
        """
        Load most recent checkpoint.

        For future resume support. Finds the checkpoint with the
        highest generation number.

        Returns:
            Tuple of (generation, prompts)

        Raises:
            KeyError: If no checkpoints exist
        """
        ...


class LocalFilePromptStorage:
    """
    PromptStorage implementation using direct filesystem writes.

    Stores prompts as JSON files in the local filesystem. This is
    the backward-compatible implementation matching current behavior.

    Args:
        base_path: Base directory for storage
        key_prefix: Optional subdirectory name. If None, writes directly
                   to base_path. If provided, creates subdirectory.
    """

    def __init__(self, base_path: Path, key_prefix: str | None = None):
        self.base_path = Path(base_path)
        self.key_prefix = key_prefix

        # Determine output directory
        if key_prefix is not None:
            self.output_dir = self.base_path / key_prefix
        else:
            self.output_dir = self.base_path

        # Create directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def save_checkpoint(
        self,
        generation: int,
        prompts: dict[str, tuple[str, str]],
        fitness_score: float | None = None,
        evaluator_scores: dict[str, float] | None = None
    ) -> None:
        """Save generation checkpoint to optimized_prompts_gen{N}.json."""
        checkpoint_path = self.output_dir / f"optimized_prompts_gen{generation}.json"

        # Convert tuples to lists for JSON serialization
        prompts_data = {
            param_name: [prompt_text, purpose]
            for param_name, (prompt_text, purpose) in prompts.items()
        }

        # Build checkpoint structure with metadata
        checkpoint_data = {
            "metadata": {
                "generation": generation,
            },
            "prompts": prompts_data
        }

        # Add optional score information
        if fitness_score is not None:
            checkpoint_data["metadata"]["fitness_score"] = fitness_score
        if evaluator_scores is not None:
            checkpoint_data["metadata"]["evaluator_scores"] = evaluator_scores

        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

    async def save_final(
        self,
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """Save final prompts to optimized_prompts.json."""
        final_path = self.output_dir / "optimized_prompts.json"

        # Convert tuples to lists for JSON serialization
        final_data = {
            param_name: [prompt_text, purpose]
            for param_name, (prompt_text, purpose) in prompts.items()
        }

        final_path.write_text(json.dumps(final_data, indent=2))

    async def load_checkpoint(
        self,
        generation: int
    ) -> dict[str, tuple[str, str]]:
        """Load checkpoint. Raises KeyError if not found."""
        checkpoint_path = self.output_dir / f"optimized_prompts_gen{generation}.json"

        if not checkpoint_path.exists():
            raise KeyError(f"Checkpoint for generation {generation} not found")

        data = json.loads(checkpoint_path.read_text())

        # Handle both new format (with metadata) and old format (just prompts)
        if "prompts" in data and "metadata" in data:
            # New format: {"metadata": {...}, "prompts": {...}}
            prompts_data = data["prompts"]
        else:
            # Old format: {"param_name": [prompt, purpose], ...}
            prompts_data = data

        # Convert lists back to tuples
        return {
            param_name: (prompt_text, purpose)
            for param_name, [prompt_text, purpose] in prompts_data.items()
        }

    async def load_latest_checkpoint(
        self
    ) -> tuple[int, dict[str, tuple[str, str]]]:
        """Load most recent checkpoint. Raises KeyError if none exist."""
        # Find all checkpoint files
        checkpoint_files = list(self.output_dir.glob("optimized_prompts_gen*.json"))

        if not checkpoint_files:
            raise KeyError("No checkpoints found")

        # Extract generation numbers and find max
        generations = []
        for path in checkpoint_files:
            # Parse "optimized_prompts_gen123.json" -> 123
            name = path.stem  # "optimized_prompts_gen123"
            gen_str = name.replace("optimized_prompts_gen", "")
            generations.append((int(gen_str), path))

        max_gen, max_path = max(generations, key=lambda x: x[0])

        # Load that checkpoint
        prompts = await self.load_checkpoint(max_gen)

        return (max_gen, prompts)


class ObjectStorePromptStorage:
    """
    PromptStorage implementation using ObjectStore interface.

    Stores prompts as JSON in any object store backend (S3, Redis,
    local files via LocalFileObjectStore, etc.).

    Args:
        object_store: ObjectStore implementation
        key_prefix: Optional key prefix. If None, generates timestamp-based
                   prefix like "prompt_opt_20260204_123456".
    """

    def __init__(self, object_store: ObjectStore, key_prefix: str | None = None):
        self.object_store = object_store

        # Generate prefix if not provided
        if key_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.key_prefix = f"prompt_opt_{timestamp}"
        else:
            self.key_prefix = key_prefix

    def _make_key(self, filename: str) -> str:
        """Construct object store key with prefix."""
        return f"{self.key_prefix}/{filename}"

    def _prompts_to_json_bytes(self, prompts: dict[str, tuple[str, str]]) -> bytes:
        """Convert prompts dict to JSON bytes."""
        # Convert tuples to lists for JSON serialization
        data = {
            param_name: [prompt_text, purpose]
            for param_name, (prompt_text, purpose) in prompts.items()
        }
        return json.dumps(data, indent=2).encode("utf-8")

    async def save_checkpoint(
        self,
        generation: int,
        prompts: dict[str, tuple[str, str]],
        fitness_score: float | None = None,
        evaluator_scores: dict[str, float] | None = None
    ) -> None:
        """Save generation checkpoint to object store."""
        key = self._make_key(f"optimized_prompts_gen{generation}.json")
        data = self._prompts_to_json_bytes(prompts)

        # Build metadata dict
        metadata = {"generation": str(generation)}

        if fitness_score is not None:
            metadata["fitness_score"] = str(fitness_score)

        if evaluator_scores is not None:
            # Store evaluator scores as JSON string since metadata values must be strings
            metadata["evaluator_scores"] = json.dumps(evaluator_scores)

        item = ObjectStoreItem(
            data=data,
            content_type="application/json",
            metadata=metadata
        )

        await self.object_store.upsert_object(key, item)

    async def save_final(
        self,
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """Save final prompts to object store."""
        key = self._make_key("optimized_prompts.json")
        data = self._prompts_to_json_bytes(prompts)

        item = ObjectStoreItem(
            data=data,
            content_type="application/json",
            metadata={"type": "final"}
        )

        await self.object_store.upsert_object(key, item)

    async def load_checkpoint(
        self,
        generation: int
    ) -> dict[str, tuple[str, str]]:
        """Load checkpoint from object store. Raises KeyError if not found."""
        key = self._make_key(f"optimized_prompts_gen{generation}.json")

        try:
            item = await self.object_store.get_object(key)
        except Exception as e:
            raise KeyError(f"Checkpoint for generation {generation} not found") from e

        # Parse JSON and convert lists to tuples
        data = json.loads(item.data.decode("utf-8"))
        return {
            param_name: (prompt_text, purpose)
            for param_name, [prompt_text, purpose] in data.items()
        }

    async def load_latest_checkpoint(
        self
    ) -> tuple[int, dict[str, tuple[str, str]]]:
        """
        Load most recent checkpoint.

        Note: This is a stub for future implementation. Object stores
        don't provide efficient listing, so this would need additional
        metadata tracking or index.
        """
        raise NotImplementedError(
            "load_latest_checkpoint not yet implemented for ObjectStorePromptStorage. "
            "Future implementation will require metadata index or listing capability."
        )

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
from pathlib import Path
from typing import Protocol


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
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """
        Save generation checkpoint.

        Args:
            generation: Generation number (1-indexed)
            prompts: Map of param_name -> (prompt_text, purpose)
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
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """Save generation checkpoint to optimized_prompts_gen{N}.json."""
        checkpoint_path = self.output_dir / f"optimized_prompts_gen{generation}.json"

        # Convert tuples to lists for JSON serialization
        checkpoint_data = {
            param_name: [prompt_text, purpose]
            for param_name, (prompt_text, purpose) in prompts.items()
        }

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

        # Convert lists back to tuples
        return {
            param_name: (prompt_text, purpose)
            for param_name, [prompt_text, purpose] in data.items()
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

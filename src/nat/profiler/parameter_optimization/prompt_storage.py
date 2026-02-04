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

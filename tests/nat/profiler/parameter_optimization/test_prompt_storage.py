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

import pytest  # noqa: F401

from nat.profiler.parameter_optimization.prompt_storage import LocalFilePromptStorage
from nat.profiler.parameter_optimization.prompt_storage import PromptStorage


class TestPromptStorageProtocol:
    def test_protocol_exists(self):
        """Test PromptStorage protocol can be imported."""
        assert PromptStorage is not None

        # Protocol should have these methods
        assert hasattr(PromptStorage, 'save_checkpoint')
        assert hasattr(PromptStorage, 'save_final')
        assert hasattr(PromptStorage, 'load_checkpoint')
        assert hasattr(PromptStorage, 'load_latest_checkpoint')


class TestLocalFilePromptStorage:
    async def test_save_checkpoint_no_prefix(self, tmp_path: Path):
        """Test save_checkpoint writes to base_path when no prefix."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix=None)

        prompts = {
            "system": ("You are helpful", "System prompt"),
            "user": ("Answer questions", "User prompt")
        }

        await storage.save_checkpoint(generation=1, prompts=prompts)

        checkpoint_path = tmp_path / "optimized_prompts_gen1.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["system"] == ["You are helpful", "System prompt"]
        assert data["user"] == ["Answer questions", "User prompt"]

    async def test_save_checkpoint_with_prefix(self, tmp_path: Path):
        """Test save_checkpoint creates subdirectory when prefix provided."""
        storage = LocalFilePromptStorage(
            base_path=tmp_path,
            key_prefix="experiment_001"
        )

        prompts = {"test": ("prompt", "purpose")}
        await storage.save_checkpoint(generation=2, prompts=prompts)

        checkpoint_path = tmp_path / "experiment_001" / "optimized_prompts_gen2.json"
        assert checkpoint_path.exists()

    async def test_save_final(self, tmp_path: Path):
        """Test save_final writes optimized_prompts.json."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix=None)

        prompts = {"final": ("Final prompt", "Final purpose")}
        await storage.save_final(prompts=prompts)

        final_path = tmp_path / "optimized_prompts.json"
        assert final_path.exists()

        data = json.loads(final_path.read_text())
        assert data["final"] == ["Final prompt", "Final purpose"]

    async def test_load_checkpoint(self, tmp_path: Path):
        """Test load_checkpoint retrieves saved checkpoint."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix=None)

        # Save checkpoint
        prompts = {"test": ("prompt text", "prompt purpose")}
        await storage.save_checkpoint(generation=1, prompts=prompts)

        # Load checkpoint
        loaded = await storage.load_checkpoint(generation=1)

        assert loaded == {"test": ("prompt text", "prompt purpose")}

    async def test_load_checkpoint_raises_on_missing(self, tmp_path: Path):
        """Test load_checkpoint raises KeyError if checkpoint missing."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix=None)

        with pytest.raises(KeyError, match="Checkpoint for generation 999 not found"):
            await storage.load_checkpoint(generation=999)

    async def test_load_latest_checkpoint(self, tmp_path: Path):
        """Test load_latest_checkpoint finds highest generation."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix=None)

        # Save multiple checkpoints
        await storage.save_checkpoint(1, {"gen1": ("p1", "purpose1")})
        await storage.save_checkpoint(3, {"gen3": ("p3", "purpose3")})
        await storage.save_checkpoint(2, {"gen2": ("p2", "purpose2")})

        # Load latest (should be gen 3)
        gen, prompts = await storage.load_latest_checkpoint()

        assert gen == 3
        assert prompts == {"gen3": ("p3", "purpose3")}

    async def test_load_latest_checkpoint_raises_on_empty(self, tmp_path: Path):
        """Test load_latest_checkpoint raises KeyError if no checkpoints."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix=None)

        with pytest.raises(KeyError, match="No checkpoints found"):
            await storage.load_latest_checkpoint()

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
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest  # noqa: F401

from nat.object_store.models import ObjectStoreItem
from nat.profiler.parameter_optimization.prompt_storage import LocalFilePromptStorage
from nat.profiler.parameter_optimization.prompt_storage import ObjectStorePromptStorage
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

        prompts = {"system": ("You are helpful", "System prompt"), "user": ("Answer questions", "User prompt")}

        await storage.save_checkpoint(generation=1, prompts=prompts)

        checkpoint_path = tmp_path / "optimized_prompts_gen1.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["system"] == ["You are helpful", "System prompt"]
        assert data["user"] == ["Answer questions", "User prompt"]

    async def test_save_checkpoint_with_prefix(self, tmp_path: Path):
        """Test save_checkpoint creates subdirectory when prefix provided."""
        storage = LocalFilePromptStorage(base_path=tmp_path, key_prefix="experiment_001")

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


class TestObjectStorePromptStorage:

    async def test_save_checkpoint_with_prefix(self):
        """Test save_checkpoint constructs correct key with prefix."""
        mock_store = MagicMock()
        mock_store.upsert_object = AsyncMock()

        storage = ObjectStorePromptStorage(object_store=mock_store, key_prefix="exp001")

        prompts = {"test": ("prompt", "purpose")}
        await storage.save_checkpoint(generation=1, prompts=prompts)

        # Verify upsert called with correct key
        mock_store.upsert_object.assert_called_once()
        call_args = mock_store.upsert_object.call_args
        assert call_args[0][0] == "exp001/optimized_prompts_gen1.json"

        # Verify ObjectStoreItem structure
        item = call_args[0][1]
        assert isinstance(item, ObjectStoreItem)
        assert item.content_type == "application/json"

        # Verify data is JSON
        data = json.loads(item.data.decode())
        assert data["test"] == ["prompt", "purpose"]

    async def test_save_checkpoint_auto_prefix(self, monkeypatch):
        """Test save_checkpoint generates timestamp prefix if None."""
        mock_store = MagicMock()
        mock_store.upsert_object = AsyncMock()

        # Mock datetime to control prefix
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20260204_123456"
        monkeypatch.setattr("nat.profiler.parameter_optimization.prompt_storage.datetime",
                            MagicMock(now=MagicMock(return_value=mock_now)))

        storage = ObjectStorePromptStorage(object_store=mock_store, key_prefix=None)

        prompts = {"test": ("prompt", "purpose")}
        await storage.save_checkpoint(generation=1, prompts=prompts)

        # Verify key has timestamp prefix
        call_args = mock_store.upsert_object.call_args
        key = call_args[0][0]
        assert key.startswith("prompt_opt_20260204_123456/")

    async def test_save_final(self):
        """Test save_final constructs correct key."""
        mock_store = MagicMock()
        mock_store.upsert_object = AsyncMock()

        storage = ObjectStorePromptStorage(object_store=mock_store, key_prefix="exp001")

        prompts = {"final": ("final prompt", "purpose")}
        await storage.save_final(prompts=prompts)

        # Verify key
        call_args = mock_store.upsert_object.call_args
        assert call_args[0][0] == "exp001/optimized_prompts.json"

    async def test_load_checkpoint(self):
        """Test load_checkpoint retrieves and parses data correctly."""
        mock_store = MagicMock()

        # Mock get_object to return saved data
        saved_data = json.dumps({"test": ["prompt text", "purpose"]}).encode()
        mock_store.get_object = AsyncMock(
            return_value=ObjectStoreItem(data=saved_data, content_type="application/json"))

        storage = ObjectStorePromptStorage(object_store=mock_store, key_prefix="exp001")

        loaded = await storage.load_checkpoint(generation=1)

        assert loaded == {"test": ("prompt text", "purpose")}
        mock_store.get_object.assert_called_once_with("exp001/optimized_prompts_gen1.json")

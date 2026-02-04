# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
from nat.object_store.local_file import LocalFileObjectStore


class TestLocalFileObjectStore:
    async def test_create_local_file_object_store(self, tmp_path: Path):
        """Test LocalFileObjectStore can be instantiated with a base path."""
        store = LocalFileObjectStore(base_path=tmp_path)
        assert store is not None
        assert store.base_path == tmp_path

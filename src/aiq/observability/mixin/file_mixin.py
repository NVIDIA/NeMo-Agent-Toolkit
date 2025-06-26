# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging

import aiofiles

logger = logging.getLogger(__name__)


class FileExportMixin:
    """Mixin for file-based exporters.

    This mixin provides file I/O functionality for exporters that need to write
    serialized data to local files.

    Args:
        *args: Variable length argument list to pass to the superclass.
        filepath (str): The path to the output file.
        project (str): The project name for metadata.
        **kwargs: Additional keyword arguments to pass to the superclass.

    Attributes:
        _filepath (str): The path to the output file.
        _project (str): The project name for metadata.
        _lock (asyncio.Lock): Lock for thread-safe file operations.
    """

    def __init__(self, *args, filepath, project, **kwargs):
        """Initialize the Phoenix exporter with the specified endpoint and project."""
        self._filepath = filepath
        self._project = project
        self._lock = asyncio.Lock()
        super().__init__(*args, **kwargs)

    async def export_processed(self, item: str) -> None:
        """Export a string.

        Args:
            item (str): The item to export.
        """

        try:
            async with self._lock:
                async with aiofiles.open(self._filepath, mode="a") as f:
                    await f.write(item)
                    await f.write("\n")
        except Exception as e:
            logger.error("Error exporting event: %s", e, exc_info=True)

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import AsyncMock
from unittest.mock import patch

import aiofiles
import pytest

from aiq.observability.mixin.file_mixin import FileExportMixin


class TestFileExportMixin:
    """Test suite for FileExportMixin class."""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file for testing with automatic cleanup."""
        return tmp_path / "test_file.txt"

    @pytest.fixture
    def invalid_file_path(self, tmp_path):
        """Create a path to a non-existent directory for error testing."""
        return tmp_path / "nonexistent_dir" / "invalid_file.txt"

    @pytest.fixture
    def mock_superclass(self):
        """Mock superclass for testing mixin."""

        class MockSuperclass:

            def __init__(self, *args, **kwargs):
                pass

        return MockSuperclass

    @pytest.fixture
    def file_mixin_class(self, mock_superclass):
        """Create a concrete class that uses FileExportMixin."""

        class TestFileExporter(FileExportMixin, mock_superclass):
            pass

        return TestFileExporter

    def test_init_with_required_parameters(self, file_mixin_class, temp_file):
        """Test initialization with required parameters."""
        filepath = temp_file
        project = "test_project"

        exporter = file_mixin_class(filepath=filepath, project=project)

        assert exporter._filepath == filepath
        assert exporter._project == project
        assert isinstance(exporter._lock, asyncio.Lock)

    def test_init_with_additional_args_and_kwargs(self, file_mixin_class, temp_file):
        """Test initialization with additional arguments."""
        filepath = temp_file
        project = "test_project"
        extra_arg = "extra"
        extra_kwarg = "extra_value"

        exporter = file_mixin_class(extra_arg, filepath=filepath, project=project, extra_key=extra_kwarg)

        assert exporter._filepath == filepath
        assert exporter._project == project
        assert isinstance(exporter._lock, asyncio.Lock)

    async def test_export_processed_writes_single_string_to_file(self, file_mixin_class, temp_file):
        """Test that export_processed successfully writes a single string to file."""
        filepath = temp_file
        project = "test_project"
        test_data = "test data line"

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify the data was written to the file
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        assert test_data + "\n" == content

    async def test_export_processed_writes_list_of_strings_to_file(self, file_mixin_class, temp_file):
        """Test that export_processed successfully writes a list of strings to file."""
        filepath = temp_file
        project = "test_project"
        test_data = ["first line", "second line", "third line"]

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify all strings were written to the file
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        expected_content = "first line\nsecond line\nthird line\n"
        assert content == expected_content

    async def test_export_processed_handles_empty_list(self, file_mixin_class, temp_file):
        """Test that export_processed handles empty list correctly."""
        filepath = temp_file
        project = "test_project"
        test_data = []

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify no content was written for empty list
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        assert content == ""

    async def test_export_processed_handles_list_with_empty_strings(self, file_mixin_class, temp_file):
        """Test that export_processed handles list containing empty strings."""
        filepath = temp_file
        project = "test_project"
        test_data = ["", "non-empty", ""]

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify all items including empty strings were written
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        expected_content = "\nnon-empty\n\n"
        assert content == expected_content

    async def test_export_processed_appends_multiple_single_strings(self, file_mixin_class, temp_file):
        """Test that multiple calls to export_processed with single strings append data."""
        filepath = temp_file
        project = "test_project"
        test_data_1 = "first line"
        test_data_2 = "second line"

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data_1)
        await exporter.export_processed(test_data_2)

        # Verify both lines were written
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        expected_content = f"{test_data_1}\n{test_data_2}\n"
        assert content == expected_content

    async def test_export_processed_appends_mixed_string_and_list(self, file_mixin_class, temp_file):
        """Test that calls with both single strings and lists append correctly."""
        filepath = temp_file
        project = "test_project"
        single_string = "single line"
        string_list = ["list item 1", "list item 2"]

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(single_string)
        await exporter.export_processed(string_list)

        # Verify all data was written in correct order
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        expected_content = "single line\nlist item 1\nlist item 2\n"
        assert content == expected_content

    async def test_export_processed_concurrent_writes_with_lists(self, file_mixin_class, temp_file):
        """Test concurrent writes with both single strings and lists are handled safely."""
        filepath = temp_file
        project = "test_project"

        exporter = file_mixin_class(filepath=filepath, project=project)

        # Create mixed concurrent export tasks
        tasks = []
        expected_lines = []

        # Add single string tasks
        for i in range(5):
            data = f"single {i}"
            expected_lines.append(data)
            tasks.append(exporter.export_processed(data))

        # Add list tasks
        for i in range(3):
            data = [f"list {i} item 1", f"list {i} item 2"]
            expected_lines.extend(data)
            tasks.append(exporter.export_processed(data))

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # Verify all lines were written
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        lines = content.strip().split('\n')
        assert len(lines) == len(expected_lines)

        # Check that all expected data is present (order may vary due to concurrency)
        for data in expected_lines:
            assert data in lines

    async def test_export_processed_handles_file_errors_with_list(self, file_mixin_class, caplog, invalid_file_path):
        """Test error handling when file operations fail with list input."""
        project = "test_project"
        test_data = ["line 1", "line 2"]

        exporter = file_mixin_class(filepath=str(invalid_file_path), project=project)

        with caplog.at_level(logging.ERROR):
            await exporter.export_processed(test_data)

        # Verify error was logged
        assert len(caplog.records) == 1
        assert "Error exporting event" in caplog.records[0].message
        assert caplog.records[0].levelname == "ERROR"

    async def test_export_processed_handles_aiofiles_exception(self, file_mixin_class, temp_file, caplog):
        """Test error handling when aiofiles operations raise exceptions."""
        filepath = temp_file
        project = "test_project"
        test_data = "test data"

        exporter = file_mixin_class(filepath=filepath, project=project)

        # Mock aiofiles.open to raise an exception
        with patch('aiq.observability.mixin.file_mixin.aiofiles.open', side_effect=Exception("Mock file error")):
            with caplog.at_level(logging.ERROR):
                await exporter.export_processed(test_data)

        # Verify error was logged
        assert len(caplog.records) == 1
        assert "Error exporting event" in caplog.records[0].message
        assert "Mock file error" in str(caplog.records[0].exc_info[1])

    async def test_export_processed_uses_lock_for_thread_safety(self, file_mixin_class, temp_file):
        """Test that the asyncio lock is used for thread safety."""
        filepath = temp_file
        project = "test_project"
        test_data = "test data"

        exporter = file_mixin_class(filepath=filepath, project=project)

        # Mock the lock to track its usage
        mock_lock = AsyncMock()
        exporter._lock = mock_lock

        await exporter.export_processed(test_data)

        # Verify the lock context manager was used
        mock_lock.__aenter__.assert_called_once()
        mock_lock.__aexit__.assert_called_once()

    async def test_export_processed_uses_lock_for_thread_safety_with_list(self, file_mixin_class, temp_file):
        """Test that the asyncio lock is used for thread safety with list input."""
        filepath = temp_file
        project = "test_project"
        test_data = ["line 1", "line 2"]

        exporter = file_mixin_class(filepath=filepath, project=project)

        # Mock the lock to track its usage
        mock_lock = AsyncMock()
        exporter._lock = mock_lock

        await exporter.export_processed(test_data)

        # Verify the lock context manager was used
        mock_lock.__aenter__.assert_called_once()
        mock_lock.__aexit__.assert_called_once()

    async def test_export_processed_with_empty_string(self, file_mixin_class, temp_file):
        """Test export_processed with empty string."""
        filepath = temp_file
        project = "test_project"
        test_data = ""

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify empty string plus newline was written
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        assert content == "\n"

    async def test_export_processed_with_multiline_string(self, file_mixin_class, temp_file):
        """Test export_processed with multiline string."""
        filepath = temp_file
        project = "test_project"
        test_data = "line 1\nline 2\nline 3"

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify multiline string plus additional newline was written
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        expected_content = test_data + "\n"
        assert content == expected_content

    async def test_export_processed_with_list_containing_multiline_strings(self, file_mixin_class, temp_file):
        """Test export_processed with list containing multiline strings."""
        filepath = temp_file
        project = "test_project"
        test_data = ["line 1\nline 2", "line 3", "line 4\nline 5\nline 6"]

        exporter = file_mixin_class(filepath=filepath, project=project)

        await exporter.export_processed(test_data)

        # Verify all multiline strings were written with proper newlines
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()

        expected_content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\n"
        assert content == expected_content

    def test_file_mixin_inherits_properly(self, file_mixin_class, temp_file):
        """Test that FileExportMixin can be properly inherited."""
        filepath = temp_file
        project = "test_project"

        # Should be able to create instance without errors
        exporter = file_mixin_class(filepath=filepath, project=project)

        # Should have all expected attributes
        assert hasattr(exporter, '_filepath')
        assert hasattr(exporter, '_project')
        assert hasattr(exporter, '_lock')
        assert hasattr(exporter, 'export_processed')

        # Method should be callable
        assert callable(exporter.export_processed)

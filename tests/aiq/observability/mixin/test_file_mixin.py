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

# pylint: disable=redefined-outer-name

import asyncio
import re

import aiofiles
import pytest

from aiq.observability.mixin.file_mixin import FileExportMixin
from aiq.observability.mixin.file_mode import FileMode


class TestFileExportMixin:
    """Test suite for FileExportMixin class."""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file for testing with automatic cleanup."""
        return tmp_path / "test_file.txt"

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for rolling tests."""
        return tmp_path / "rolling_test_dir"

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
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        assert exporter._filepath == output_path
        assert exporter._project == project
        assert isinstance(exporter._lock, asyncio.Lock)

    def test_init_with_additional_args_and_kwargs(self, file_mixin_class, temp_file):
        """Test initialization with additional arguments."""
        output_path = temp_file
        project = "test_project"
        extra_arg = "extra"
        extra_kwarg = "extra_value"

        exporter = file_mixin_class(extra_arg, output_path=output_path, project=project, extra_key=extra_kwarg)

        assert exporter._filepath == output_path
        assert exporter._project == project
        assert isinstance(exporter._lock, asyncio.Lock)

    def test_init_with_rolling_enabled(self, file_mixin_class, temp_dir):
        """Test initialization with rolling enabled."""
        output_path = temp_dir / "app.log"
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path,
                                    project=project,
                                    enable_rolling=True,
                                    max_file_size=1024,
                                    max_files=5)

        assert exporter._enable_rolling is True
        assert exporter._max_file_size == 1024
        assert exporter._max_files == 5
        assert exporter._base_dir == temp_dir
        assert exporter._base_filename == "app"
        assert exporter._file_extension == ".log"
        assert exporter._current_file_path == temp_dir / "app.log"

    def test_init_rolling_with_directory_path(self, file_mixin_class, temp_dir):
        """Test rolling initialization when output_path is a directory."""
        project = "test_project"

        exporter = file_mixin_class(output_path=temp_dir, project=project, enable_rolling=True)

        assert exporter._base_dir == temp_dir
        assert exporter._base_filename == "test_project_export"
        assert exporter._file_extension == ".log"
        assert exporter._current_file_path == temp_dir / "test_project_export.log"

    def test_init_creates_directory_structure(self, file_mixin_class, tmp_path):
        """Test that initialization creates necessary directory structure."""
        nested_path = tmp_path / "logs" / "app" / "trace.log"

        exporter = file_mixin_class(output_path=nested_path, project="test", enable_rolling=True)

        # Directory should be created
        assert nested_path.parent.exists()
        assert exporter._base_dir == nested_path.parent

    async def test_export_processed_writes_single_string_to_file(self, file_mixin_class, temp_file):
        """Test that export_processed successfully writes a single string to file."""
        output_path = temp_file
        project = "test_project"
        test_data = "test data line"

        exporter = file_mixin_class(output_path=output_path, project=project)

        await exporter.export_processed(test_data)

        # Verify the data was written to the file
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        assert test_data + "\n" == content

    async def test_export_processed_writes_list_of_strings_to_file(self, file_mixin_class, temp_file):
        """Test that export_processed successfully writes a list of strings to file."""
        output_path = temp_file
        project = "test_project"
        test_data = ["first line", "second line", "third line"]

        exporter = file_mixin_class(output_path=output_path, project=project)

        await exporter.export_processed(test_data)

        # Verify all strings were written to the file
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        expected_content = "first line\nsecond line\nthird line\n"
        assert content == expected_content

    async def test_export_processed_handles_empty_list(self, file_mixin_class, temp_file):
        """Test that export_processed handles empty list correctly."""
        output_path = temp_file
        project = "test_project"
        test_data = []

        exporter = file_mixin_class(output_path=output_path, project=project)

        await exporter.export_processed(test_data)

        # Verify no content was written for empty list
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        assert content == ""

    async def test_export_processed_appends_on_multiple_calls(self, file_mixin_class, temp_file):
        """Test that multiple calls to export_processed append to the file."""
        output_path = temp_file
        project = "test_project"
        first_data = "first write"
        second_data = "second write"

        exporter = file_mixin_class(output_path=output_path, project=project)

        await exporter.export_processed(first_data)
        await exporter.export_processed(second_data)

        # Verify both writes were appended
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        expected_content = "first write\nsecond write\n"
        assert content == expected_content

    async def test_export_processed_concurrent_access(self, file_mixin_class, temp_file):
        """Test that concurrent access to export_processed is handled safely."""
        output_path = temp_file
        project = "test_project"
        concurrent_data = ["data1", "data2", "data3", "data4", "data5"]

        exporter = file_mixin_class(output_path=output_path, project=project)

        # Create concurrent export tasks
        tasks = [exporter.export_processed(data) for data in concurrent_data]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # Verify all strings were written
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        lines = content.strip().split('\n')
        assert len(lines) == len(concurrent_data)

        # All data should be present (order may vary due to concurrency)
        for data in concurrent_data:
            assert data in lines

    async def test_export_processed_concurrent_writes_with_lists(self, file_mixin_class, temp_file):
        """Test concurrent writes with both single strings and lists are handled safely."""
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        # Create mixed concurrent export tasks
        single_strings = ["single1", "single2", "single3"]
        list_data = [["list1a", "list1b"], ["list2a", "list2b"]]

        tasks = []
        expected_lines = []

        # Add single string tasks
        for s in single_strings:
            tasks.append(exporter.export_processed(s))
            expected_lines.append(s)

        # Add list tasks
        for lst in list_data:
            tasks.append(exporter.export_processed(lst))
            expected_lines.extend(lst)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # Verify all lines were written
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        lines = content.strip().split('\n')
        assert len(lines) == len(expected_lines)

        # All expected lines should be present
        for expected_line in expected_lines:
            assert expected_line in lines

    async def test_export_processed_with_error_handling(self, file_mixin_class, invalid_file_path):
        """Test error handling when file operations fail."""
        project = "test_project"

        # This should not raise an exception during initialization
        exporter = file_mixin_class(output_path=str(invalid_file_path), project=project)

        # This should handle the error gracefully (not raise exception)
        await exporter.export_processed("test data")

        # Verify the exporter is still in a valid state
        assert exporter._project == project

    async def test_export_processed_mixed_data_types(self, file_mixin_class, temp_file):
        """Test export_processed with different types of string data."""
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        # Test with various string types
        test_cases = [
            "simple string",
            "string with special characters: !@#$%^&*()",
            "unicode string: 你好世界",
            "",  # empty string
            "   spaces around   ",
        ]

        for test_string in test_cases:
            await exporter.export_processed(test_string)

        # Test newline strings separately since they affect line counting
        await exporter.export_processed("string with\nnewlines")

        # Verify content was written (not counting lines due to embedded newlines)
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        # Just verify all content is present in some form
        assert "simple string" in content
        assert "special characters" in content
        assert "你好世界" in content
        assert "spaces around" in content
        assert "string with" in content
        assert "newlines" in content

    async def test_export_processed_list_edge_cases(self, file_mixin_class, temp_file):
        """Test export_processed with various list edge cases."""
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        # Test with different list scenarios
        await exporter.export_processed([])  # empty list
        await exporter.export_processed(["single_item"])  # single item list
        await exporter.export_processed(["", "", ""])  # list of empty strings

        # Verify the file content
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        # Empty list should write nothing, single item should write one line + \n,
        # three empty strings should write three \n
        expected_content = "single_item\n\n\n\n"
        assert content == expected_content

    async def test_export_processed_large_data(self, file_mixin_class, temp_file):
        """Test export_processed with larger amounts of data."""
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        # Generate a large list
        large_list = [f"line_{i}" for i in range(1000)]

        await exporter.export_processed(large_list)

        # Verify all lines were written
        async with aiofiles.open(output_path, mode='r') as f:
            content = await f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 1000
        assert lines[0] == "line_0"
        assert lines[999] == "line_999"

    def test_output_path_attribute_access(self, file_mixin_class, temp_file):
        """Test that _filepath attribute is accessible and correct (internal representation of output_path)."""
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        assert hasattr(exporter, '_filepath')
        assert exporter._filepath == output_path

    def test_project_attribute_access(self, file_mixin_class, temp_file):
        """Test that _project attribute is accessible and correct."""
        output_path = temp_file
        project = "test_project"

        exporter = file_mixin_class(output_path=output_path, project=project)

        assert hasattr(exporter, '_project')
        assert exporter._project == project


class TestFileExportMixinRolling:
    """Test suite for FileExportMixin rolling functionality."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for rolling tests."""
        return tmp_path / "rolling_tests"

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

    async def test_file_rolling_when_size_exceeded(self, file_mixin_class, temp_dir):
        """Test that files are rolled when max_file_size is exceeded."""
        output_path = temp_dir / "app.log"

        exporter = file_mixin_class(
            output_path=output_path,
            project="test",
            enable_rolling=True,
            max_file_size=15,  # Very small to force rolling
            max_files=5)

        # Write content that will create a file exactly at the limit
        first_message = "Exactly 15 chars"  # 16 chars + newline = 16 bytes (> 15)

        # First write - creates a file that exceeds the limit
        await exporter.export_processed(first_message)
        assert output_path.exists()
        initial_files = list(temp_dir.glob("*.log"))
        assert len(initial_files) == 1

        # Second write - should trigger roll because file is already > 15 bytes
        second_message = "Second message"
        await exporter.export_processed(second_message)

        # Should now have 2 files: current + 1 rolled
        all_files = list(temp_dir.glob("*.log"))
        assert len(all_files) == 2

        # Check that one file has timestamp
        rolled_files = [f for f in all_files if re.search(r'\d{8}_\d{6}_\d{6}', f.name)]
        assert len(rolled_files) == 1

    async def test_file_rolling_preserves_content(self, file_mixin_class, temp_dir):
        """Test that rolled files preserve their content correctly."""
        output_path = temp_dir / "preserve.log"

        exporter = file_mixin_class(
            output_path=output_path,
            project="test",
            enable_rolling=True,
            max_file_size=15,  # Very small to trigger rolling
            max_files=3)

        first_content = "This first message"  # 18 chars + newline = 19 bytes (> 15)
        second_content = "Second message that is definitely longer"  # 40+ chars

        # Write first message (creates file > 15 bytes)
        await exporter.export_processed(first_content)

        # Write second message - should trigger roll because file is already > 15 bytes
        await exporter.export_processed(second_content)

        # Find the rolled file
        rolled_files = [f for f in temp_dir.glob("*.log") if re.search(r'\d{8}_\d{6}_\d{6}', f.name)]
        assert len(rolled_files) == 1

        # Check content of rolled file
        rolled_content = rolled_files[0].read_text()
        assert rolled_content.strip() == first_content

        # Check content of current file
        current_content = output_path.read_text()
        assert current_content.strip() == second_content

    async def test_file_cleanup_when_max_files_exceeded(self, file_mixin_class, temp_dir):
        """Test that old files are cleaned up when max_files limit is reached."""
        output_path = temp_dir / "cleanup.log"

        exporter = file_mixin_class(
            output_path=output_path,
            project="test",
            enable_rolling=True,
            max_file_size=10,  # Very small to force frequent rolling
            max_files=2  # Keep only 2 rolled files
        )

        # Write multiple messages to trigger several rolls
        messages = [f"Message {i} content" for i in range(6)]

        for message in messages:
            await exporter.export_processed(message)

        # Should have current file + max 2 rolled files = 3 total
        all_files = list(temp_dir.glob("*.log"))
        assert len(all_files) <= 3  # Current + 2 rolled files max

        # Check that we have exactly 2 rolled files (or less if not all triggered rolling)
        rolled_files = [f for f in all_files if re.search(r'\d{8}_\d{6}_\d{6}', f.name)]
        assert len(rolled_files) <= 2

    async def test_timestamp_precision_prevents_collisions(self, file_mixin_class, temp_dir):
        """Test that microsecond precision prevents timestamp collisions."""
        output_path = temp_dir / "precision.log"

        exporter = file_mixin_class(
            output_path=output_path,
            project="test",
            enable_rolling=True,
            max_file_size=5,  # Force rolling on nearly every write
            max_files=10)

        # Write messages rapidly to test timestamp precision
        messages = [f"Msg{i}" for i in range(8)]

        for message in messages:
            await exporter.export_processed(message)

        # Get all rolled files
        rolled_files = [f for f in temp_dir.glob("*.log") if re.search(r'\d{8}_\d{6}_\d{6}', f.name)]

        # Extract timestamps from filenames
        timestamps = []
        for f in rolled_files:
            match = re.search(r'(\d{8}_\d{6}_\d{6})', f.name)
            if match:
                timestamps.append(match.group(1))

        # All timestamps should be unique
        assert len(timestamps) == len(set(timestamps)), f"Duplicate timestamps found: {timestamps}"

        # Verify microsecond format (YYYYMMDD_HHMMSS_microseconds)
        for timestamp in timestamps:
            assert re.match(r'\d{8}_\d{6}_\d{6}', timestamp), f"Invalid timestamp format: {timestamp}"

    async def test_should_roll_file_logic(self, file_mixin_class, temp_dir):
        """Test the _should_roll_file logic works correctly."""
        output_path = temp_dir / "roll_test.log"

        exporter = file_mixin_class(output_path=output_path, project="test", enable_rolling=True, max_file_size=20)

        # Should not roll when file doesn't exist
        should_roll = await exporter._should_roll_file()
        assert should_roll is False

        # Write small content
        await exporter.export_processed("Small")
        should_roll = await exporter._should_roll_file()
        assert should_roll is False  # Should be under 20 bytes

        # Write content to exceed limit
        await exporter.export_processed("This is a longer message")
        should_roll = await exporter._should_roll_file()
        assert should_roll is True  # Should exceed 20 bytes

    async def test_rolling_disabled_behavior(self, file_mixin_class, tmp_path):
        """Test that rolling doesn't occur when disabled."""
        temp_file = tmp_path / "no_rolling.log"

        exporter = file_mixin_class(
            output_path=temp_file,
            project="test",
            enable_rolling=False,  # Explicitly disabled
            max_file_size=10  # Very small, but rolling disabled
        )

        # Write multiple large messages
        messages = ["Very long message that would normally trigger rolling" for _ in range(3)]

        for message in messages:
            await exporter.export_processed(message)

        # Should only have the original file
        parent_dir = temp_file.parent
        log_files = list(parent_dir.glob("*.log"))
        assert len(log_files) == 1
        assert log_files[0] == temp_file

    async def test_concurrent_rolling_safety(self, file_mixin_class, temp_dir):
        """Test that concurrent writes handle rolling safely."""
        output_path = temp_dir / "concurrent.log"

        exporter = file_mixin_class(output_path=output_path,
                                    project="test",
                                    enable_rolling=True,
                                    max_file_size=15,
                                    max_files=5)

        # Create concurrent tasks that should trigger rolling
        long_messages = [f"Long message {i} that triggers rolling" for i in range(5)]
        tasks = [exporter.export_processed(msg) for msg in long_messages]

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Verify all content was written (no data loss)
        all_files = list(temp_dir.glob("*.log"))
        all_content = []

        for file_path in all_files:
            content = file_path.read_text().strip()
            if content:
                all_content.extend(content.split('\n'))

        # All messages should be present somewhere
        for message in long_messages:
            assert message in all_content

    def test_get_current_file_path(self, file_mixin_class, temp_dir):
        """Test get_current_file_path method."""
        output_path = temp_dir / "current.log"

        exporter = file_mixin_class(output_path=output_path, project="test", enable_rolling=True)

        current_path = exporter.get_current_file_path()
        assert current_path == output_path
        assert current_path == exporter._current_file_path

    def test_get_file_info(self, file_mixin_class, temp_dir):
        """Test get_file_info method returns correct information."""
        output_path = temp_dir / "info.log"

        exporter = file_mixin_class(output_path=output_path,
                                    project="test",
                                    enable_rolling=True,
                                    max_file_size=1024,
                                    max_files=3,
                                    cleanup_on_init=True,
                                    mode=FileMode.APPEND)

        info = exporter.get_file_info()

        assert info["current_file"] == str(output_path)
        assert info["mode"] == "append"
        assert info["rolling_enabled"] is True
        assert info["cleanup_on_init"] is True
        assert info["max_file_size"] == 1024
        assert info["max_files"] == 3
        assert info["base_directory"] == str(temp_dir)

    def test_get_file_info_without_rolling(self, file_mixin_class, tmp_path):
        """Test get_file_info method when rolling is disabled."""
        temp_file = tmp_path / "info_test.log"
        exporter = file_mixin_class(output_path=temp_file, project="test", enable_rolling=False)

        info = exporter.get_file_info()

        assert info["current_file"] == str(temp_file)
        assert info["rolling_enabled"] is False
        assert "max_file_size" not in info
        assert "max_files" not in info
        assert "base_directory" not in info

    async def test_overwrite_mode_with_rolling(self, file_mixin_class, temp_dir):
        """Test overwrite mode behavior with rolling enabled."""
        output_path = temp_dir / "overwrite.log"

        exporter = file_mixin_class(output_path=output_path,
                                    project="test",
                                    enable_rolling=True,
                                    mode="overwrite",
                                    max_file_size=30)

        # First write should create file
        await exporter.export_processed("First message")
        assert output_path.exists()

        # Second write should append (overwrite only applies to first write)
        await exporter.export_processed("Second message")

        content = output_path.read_text()
        assert "First message" in content
        assert "Second message" in content

    async def test_cleanup_on_init_removes_existing_files(self, file_mixin_class, temp_dir):
        """Test that cleanup_on_init removes existing rolled files."""
        output_path = temp_dir / "cleanup_init.log"

        # Ensure the directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create some pre-existing rolled files
        (temp_dir / "cleanup_init_20240101_120000_123456.log").write_text("old1")
        (temp_dir / "cleanup_init_20240101_120001_123456.log").write_text("old2")
        (temp_dir / "cleanup_init_20240101_120002_123456.log").write_text("old3")

        # Create exporter with max_files=1 and cleanup_on_init=True
        exporter = file_mixin_class(output_path=output_path,
                                    project="test",
                                    enable_rolling=True,
                                    max_files=1,
                                    cleanup_on_init=True)

        # Verify exporter was initialized properly
        assert exporter._cleanup_on_init is True
        assert exporter._max_files == 1

        # Should have cleaned up to only 1 file (the newest)
        rolled_files = list(temp_dir.glob("cleanup_init_*.log"))
        assert len(rolled_files) <= 1

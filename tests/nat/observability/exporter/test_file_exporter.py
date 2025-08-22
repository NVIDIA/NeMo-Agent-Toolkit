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
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.invocation_node import InvocationNode
from nat.observability.exporter.file_exporter import FileExporter
from nat.observability.exporter.raw_exporter import RawExporter
from nat.observability.mixin.file_mixin import FileExportMixin
from nat.observability.processor.intermediate_step_serializer import IntermediateStepSerializer


@pytest.fixture
def mock_context_state():
    """Create a mock context state."""
    mock_state = Mock(spec=ContextState)
    return mock_state


@pytest.fixture
def sample_intermediate_step():
    """Create a sample intermediate step for testing."""
    return IntermediateStep(parent_id="root",
                            function_ancestry=InvocationNode(function_name="test_function", function_id="test-id"),
                            payload=IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                                            name="test_tool",
                                                            tags=["test"],
                                                            UUID="test-uuid-123"))


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    return str(tmp_path / "test_export.jsonl")


@pytest.fixture
def invalid_file_path(tmp_path):
    """Create an invalid file path for error testing."""
    return tmp_path / "nonexistent_dir" / "invalid_file.txt"


class TestFileExporterInitialization:
    """Test FileExporter initialization and constructor behavior."""

    def test_basic_initialization(self, mock_context_state, tmp_path):
        """Test basic initialization with required parameters."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        assert exporter._filepath == test_output_path
        assert exporter._project == "test_project"
        assert isinstance(exporter._processor, IntermediateStepSerializer)

    def test_initialization_without_context_state(self, tmp_path):
        """Test initialization without context state."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(output_path=str(test_output_path), project="test_project")

        assert exporter._filepath == test_output_path
        assert exporter._project == "test_project"
        assert isinstance(exporter._processor, IntermediateStepSerializer)

    def test_initialization_with_invalid_kwargs_fails(self, mock_context_state, tmp_path):
        """Test initialization fails with invalid kwargs."""
        test_output_path = tmp_path / "test.jsonl"
        with pytest.raises(TypeError):
            FileExporter(context_state=mock_context_state,
                         output_path=str(test_output_path),
                         project="test_project",
                         extra_param="extra_value")

    @patch('nat.observability.exporter.file_exporter.IntermediateStepSerializer')
    def test_processor_initialization(self, mock_serializer_class, mock_context_state, tmp_path):
        """Test that the processor is properly initialized and added."""
        mock_serializer_instance = Mock()
        mock_serializer_class.return_value = mock_serializer_instance
        test_output_path = tmp_path / "test.jsonl"

        with patch.object(FileExporter, 'add_processor') as mock_add_processor:
            exporter = FileExporter(context_state=mock_context_state,
                                    output_path=str(test_output_path),
                                    project="test_project")

            mock_serializer_class.assert_called_once()
            mock_add_processor.assert_called_once_with(mock_serializer_instance)
            assert exporter._processor == mock_serializer_instance


class TestFileExporterInheritance:
    """Test FileExporter inheritance and type relationships."""

    def test_inheritance_from_file_export_mixin(self, mock_context_state, tmp_path):
        """Test that FileExporter properly inherits from FileExportMixin."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        assert isinstance(exporter, FileExportMixin)
        assert hasattr(exporter, 'export_processed')
        assert hasattr(exporter, '_filepath')
        assert hasattr(exporter, '_project')
        assert hasattr(exporter, '_lock')

    def test_inheritance_from_raw_exporter(self, mock_context_state, tmp_path):
        """Test that FileExporter properly inherits from RawExporter."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        assert isinstance(exporter, RawExporter)
        assert hasattr(exporter, 'export')
        assert hasattr(exporter, 'add_processor')

    def test_method_resolution_order(self, mock_context_state, tmp_path):
        """Test that method resolution order is correct."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        # FileExportMixin should come before RawExporter in MRO
        mro = type(exporter).__mro__
        file_mixin_index = next(i for i, cls in enumerate(mro) if cls == FileExportMixin)
        raw_exporter_index = next(i for i, cls in enumerate(mro) if cls == RawExporter)

        assert file_mixin_index < raw_exporter_index


class TestFileExporterFunctionality:
    """Test FileExporter core functionality."""

    async def test_export_processed_single_string(self, mock_context_state, temp_file):
        """Test exporting a single string."""
        exporter = FileExporter(context_state=mock_context_state, output_path=temp_file, project="test_project")

        test_string = '{"test": "data"}'
        await exporter.export_processed(test_string)

        # Verify file content
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == test_string + '\n'

    async def test_export_processed_list_of_strings(self, mock_context_state, temp_file):
        """Test exporting a list of strings."""
        exporter = FileExporter(context_state=mock_context_state, output_path=temp_file, project="test_project")

        test_strings = ['{"test1": "data1"}', '{"test2": "data2"}']
        await exporter.export_processed(test_strings)

        # Verify file content
        with open(temp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == test_strings[0]
            assert lines[1].strip() == test_strings[1]

    async def test_export_processed_multiple_calls(self, mock_context_state, temp_file):
        """Test multiple calls to export_processed append to file."""
        exporter = FileExporter(context_state=mock_context_state, output_path=temp_file, project="test_project")

        await exporter.export_processed('{"line": 1}')
        await exporter.export_processed('{"line": 2}')

        # Verify file content
        with open(temp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == '{"line": 1}'
            assert lines[1].strip() == '{"line": 2}'

    @patch('aiofiles.open')
    async def test_export_processed_file_error_handling(self, mock_aiofiles_open, mock_context_state,
                                                        invalid_file_path):
        """Test error handling when file operations fail."""
        # Mock file operation to raise an exception
        mock_aiofiles_open.side_effect = IOError("File write error")

        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(invalid_file_path),
                                project="test_project")

        # Should not raise exception, but log error
        with patch('nat.observability.mixin.file_mixin.logger') as mock_logger:
            await exporter.export_processed('{"test": "data"}')
            # Verify error was logged (implementation logs errors but doesn't re-raise)
            mock_logger.error.assert_called()

    def test_export_method_inheritance(self, mock_context_state, sample_intermediate_step, tmp_path):
        """Test that export method works through inheritance."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        # Mock the task creation to avoid async complexity
        with patch.object(exporter, '_create_export_task') as mock_create_task:
            exporter.export(sample_intermediate_step)
            mock_create_task.assert_called_once()

            # Clean up any created coroutines
            args = mock_create_task.call_args[0]
            if args and hasattr(args[0], 'close'):
                args[0].close()


class TestFileExporterIntegration:
    """Test FileExporter integration with processing pipeline."""

    @patch('aiofiles.open')
    async def test_end_to_end_processing(self,
                                         mock_aiofiles_open,
                                         mock_context_state,
                                         sample_intermediate_step,
                                         tmp_path):
        """Test end-to-end processing from IntermediateStep to file output."""
        # Mock file operations
        mock_file = AsyncMock()
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file
        test_output_path = tmp_path / "test.jsonl"

        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        # Mock the serializer to return a known string
        with patch.object(exporter._processor, 'process', return_value='{"serialized": "data"}') as mock_process:
            await exporter._export_with_processing(sample_intermediate_step)

            # Verify processor was called
            mock_process.assert_called_once_with(sample_intermediate_step)

            # Verify file write was called
            mock_file.write.assert_called()
            written_calls = [call.args[0] for call in mock_file.write.call_args_list]
            assert '{"serialized": "data"}' in written_calls
            assert '\n' in written_calls

    async def test_processor_pipeline_integration(self, mock_context_state, sample_intermediate_step, tmp_path):
        """Test integration with the processing pipeline."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        # Mock the export_processed method to track calls
        with patch.object(exporter, 'export_processed') as mock_export_processed:
            # Mock the processor to return a known value
            with patch.object(exporter._processor, 'process', return_value='processed_output'):
                await exporter._export_with_processing(sample_intermediate_step)

                mock_export_processed.assert_called_once_with('processed_output')


class TestFileExporterEdgeCases:
    """Test FileExporter edge cases and error conditions."""

    def test_initialization_missing_output_path(self, mock_context_state):
        """Test initialization fails when output_path is missing."""
        with pytest.raises(TypeError):
            FileExporter(context_state=mock_context_state, project="test_project"
                         # Missing output_path
                         )

    def test_initialization_missing_project(self, mock_context_state):
        """Test initialization fails when project is missing."""
        with pytest.raises(TypeError):
            FileExporter(context_state=mock_context_state,
                         output_path="./.tmp/test.jsonl"
                         # Missing project - but this should use tmp_path too
                         )

    async def test_export_processed_empty_string(self, mock_context_state, temp_file):
        """Test exporting an empty string."""
        exporter = FileExporter(context_state=mock_context_state, output_path=temp_file, project="test_project")

        await exporter.export_processed('')

        # Verify file content
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == '\n'

    async def test_export_processed_empty_list(self, mock_context_state, temp_file):
        """Test exporting an empty list."""
        exporter = FileExporter(context_state=mock_context_state, output_path=temp_file, project="test_project")

        await exporter.export_processed([])

        # Verify file is empty (no writes for empty list)
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert content == ''

    async def test_concurrent_export_calls(self, mock_context_state, temp_file):
        """Test concurrent calls to export_processed use lock correctly."""
        exporter = FileExporter(context_state=mock_context_state, output_path=temp_file, project="test_project")

        # Create multiple concurrent tasks
        tasks = [exporter.export_processed(f'{{"concurrent": {i}}}') for i in range(5)]

        await asyncio.gather(*tasks)

        # Verify all lines were written
        with open(temp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 5
            # All lines should be valid (no corruption from concurrent writes)
            for line in lines:
                assert line.startswith('{"concurrent":') and line.endswith('}\n')

    def test_processor_type_checking(self, mock_context_state):
        """Test that the processor is of the correct type."""
        exporter = FileExporter(context_state=mock_context_state,
                                output_path="./.tmp/test.jsonl",
                                project="test_project")

        assert isinstance(exporter._processor, IntermediateStepSerializer)
        assert hasattr(exporter._processor, 'process')

    async def test_export_with_non_intermediate_step(self, mock_context_state, tmp_path):
        """Test export method behavior with non-IntermediateStep objects."""
        test_output_path = tmp_path / "test.jsonl"
        exporter = FileExporter(context_state=mock_context_state,
                                output_path=str(test_output_path),
                                project="test_project")

        # Mock task creation to verify it's not called for invalid types
        with patch.object(exporter, '_create_export_task') as mock_create_task:
            # These should not trigger export
            exporter.export("not an intermediate step")  # type: ignore[arg-type]
            exporter.export(123)  # type: ignore[arg-type]
            exporter.export(None)  # type: ignore[arg-type]
            exporter.export([])  # type: ignore[arg-type]

            mock_create_task.assert_not_called()


class TestFileExporterLogging:
    """Test FileExporter logging behavior."""

    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        from nat.observability.exporter.file_exporter import logger

        assert logger.name == 'nat.observability.exporter.file_exporter'

    @patch('nat.observability.exporter.file_exporter.logger')
    def test_no_unexpected_logging_during_normal_operation(self, mock_logger, mock_context_state, temp_file):
        """Test that normal operations don't produce unexpected log messages."""
        exporter = FileExporter(context_state=mock_context_state, output_path=str(temp_file), project="test_project")

        # Verify exporter was created successfully
        assert exporter is not None

        # Normal initialization should not produce warning/error logs
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

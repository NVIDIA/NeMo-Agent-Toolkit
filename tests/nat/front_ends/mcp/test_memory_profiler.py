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

from unittest.mock import MagicMock
from unittest.mock import patch

from nat.builder.intermediate_step_manager import IntermediateStepManager
from nat.front_ends.mcp.memory_profiler import MemoryProfiler


class TestMemoryProfilerInit:
    """Test MemoryProfiler initialization."""

    def test_init_disabled(self):
        """Test initialization with profiling disabled."""
        profiler = MemoryProfiler(enabled=False)

        assert profiler.enabled is False
        assert profiler.request_count == 0
        assert profiler.baseline_snapshot is None

    def test_init_enabled(self):
        """Test initialization with profiling enabled."""
        profiler = MemoryProfiler(enabled=True, log_interval=10, top_n=5)

        assert profiler.enabled is True
        assert profiler.log_interval == 10
        assert profiler.top_n == 5
        assert profiler.request_count == 0

    def test_init_normalizes_interval(self):
        """Test that log_interval is normalized to avoid modulo-by-zero."""
        profiler = MemoryProfiler(enabled=True, log_interval=0)

        assert profiler.log_interval == 1  # Should be normalized to 1


class TestMemoryProfilerDisabled:
    """Test MemoryProfiler behavior when disabled."""

    def test_on_request_complete_disabled(self):
        """Test that on_request_complete does nothing when disabled."""
        profiler = MemoryProfiler(enabled=False)

        profiler.on_request_complete()
        profiler.on_request_complete()

        assert profiler.request_count == 0  # Should not increment

    def test_get_stats_disabled(self):
        """Test that get_stats returns minimal info when disabled."""
        profiler = MemoryProfiler(enabled=False)

        stats = profiler.get_stats()

        assert stats == {"enabled": False}

    def test_log_memory_stats_disabled(self):
        """Test that log_memory_stats returns empty dict when disabled."""
        profiler = MemoryProfiler(enabled=False)

        stats = profiler.log_memory_stats()

        assert stats == {}

    def test_reset_baseline_disabled(self):
        """Test that reset_baseline does nothing when disabled."""
        profiler = MemoryProfiler(enabled=False)

        # Should not raise any errors
        profiler.reset_baseline()


class TestMemoryProfilerEnabled:
    """Test MemoryProfiler behavior when enabled."""

    def test_on_request_complete_increments(self):
        """Test that request count increments."""
        profiler = MemoryProfiler(enabled=True, log_interval=100)

        profiler.on_request_complete()
        assert profiler.request_count == 1

        profiler.on_request_complete()
        assert profiler.request_count == 2

    @patch('nat.front_ends.mcp.memory_profiler.logger')
    def test_on_request_complete_logs_at_interval(self, mock_logger):
        """Test that memory stats are logged at the configured interval."""
        profiler = MemoryProfiler(enabled=True, log_interval=2)

        # First request - no logging
        profiler.on_request_complete()
        assert profiler.request_count == 1

        # Second request - should log
        profiler.on_request_complete()
        assert profiler.request_count == 2
        # Check that info logging happened (tracemalloc might not be available)
        assert mock_logger.info.called

    def test_get_stats_returns_structure(self):
        """Test that get_stats returns expected structure."""
        profiler = MemoryProfiler(enabled=True)

        stats = profiler.get_stats()

        assert stats["enabled"] is True
        assert stats["request_count"] == 0
        assert "active_intermediate_managers" in stats
        assert "outstanding_steps" in stats
        assert "active_exporters" in stats
        assert "isolated_exporters" in stats
        assert "subject_instances" in stats


class TestMemoryProfilerInstanceTracking:
    """Test instance tracking functionality."""

    def test_safe_intermediate_step_manager_count(self):
        """Test counting IntermediateStepManager instances."""
        profiler = MemoryProfiler(enabled=True)

        # Clear any existing instances
        initial_count = profiler._safe_intermediate_step_manager_count()

        # Create a mock context state
        mock_context = MagicMock()
        mock_context.active_span_id_stack.get.return_value = ["root"]

        # Create an instance
        manager = IntermediateStepManager(mock_context)

        # Count should increase
        new_count = profiler._safe_intermediate_step_manager_count()
        assert new_count == initial_count + 1

        # Delete the instance
        del manager

        # Count should decrease after garbage collection
        import gc
        gc.collect()
        final_count = profiler._safe_intermediate_step_manager_count()
        assert final_count == initial_count

    def test_safe_outstanding_step_count(self):
        """Test counting outstanding steps."""
        profiler = MemoryProfiler(enabled=True)

        # Should not crash even if no managers exist
        count = profiler._safe_outstanding_step_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_safe_exporter_count(self):
        """Test counting exporters."""
        profiler = MemoryProfiler(enabled=True)

        # Should not crash
        count = profiler._safe_exporter_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_count_instances_of_type(self):
        """Test generic instance counting by type name."""
        profiler = MemoryProfiler(enabled=True)

        # Count some common type
        count = profiler._count_instances_of_type("dict")
        assert isinstance(count, int)
        assert count > 0  # There should be many dicts in memory


class TestMemoryProfilerThreadSafety:
    """Test thread-safety handling."""

    def test_safe_outstanding_step_count_handles_runtime_error(self):
        """Test that RuntimeError during iteration is handled gracefully."""
        profiler = MemoryProfiler(enabled=True)

        # Mock IntermediateStepManager at the source where it's imported
        with patch('nat.builder.intermediate_step_manager.IntermediateStepManager') as mock_class:
            mock_class._active_instances = MagicMock()
            mock_class._active_instances.__iter__.side_effect = RuntimeError("Set changed")

            # Should return 0 instead of crashing
            count = profiler._safe_outstanding_step_count()
            assert count == 0

    def test_safe_intermediate_step_manager_count_handles_runtime_error(self):
        """Test that RuntimeError during count is handled gracefully."""
        profiler = MemoryProfiler(enabled=True)

        # Mock at the source where it's imported
        with patch('nat.builder.intermediate_step_manager.IntermediateStepManager') as mock_class:
            mock_class.get_active_instance_count.side_effect = RuntimeError("Set modified")

            # Should return 0 instead of crashing
            count = profiler._safe_intermediate_step_manager_count()
            assert count == 0


class TestMemoryProfilerEdgeCases:
    """Test edge cases and error handling."""

    def test_count_instances_handles_exceptions(self):
        """Test that instance counting handles exceptions gracefully."""
        profiler = MemoryProfiler(enabled=True)

        # Should not crash even with invalid type name
        count = profiler._count_instances_of_type("NonExistentTypeThatDoesNotExist")
        assert count == 0

    def test_log_memory_stats_without_tracemalloc(self):
        """Test that stats logging works even if tracemalloc is unavailable."""
        profiler = MemoryProfiler(enabled=True)

        # Mock tracemalloc to be unavailable
        with patch.object(profiler, '_safe_traced_memory', return_value=None):
            stats = profiler.log_memory_stats()

            # Should return stats with None for memory values
            assert stats["current_memory_mb"] is None
            assert stats["peak_memory_mb"] is None
            assert "active_intermediate_managers" in stats

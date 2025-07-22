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
"""Tests for development mode manager."""

import asyncio
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq.data_models.config import AIQConfig
from aiq.runtime.config_manager import ConfigManager
from aiq.runtime.dev_mode_manager import DevModeManager
from aiq.runtime.dev_mode_manager import run_with_dev_mode
from aiq.runtime.events import ConfigChangeEvent
from aiq.runtime.events import ConfigEventType


class TestDevModeManager:
    """Test suite for DevModeManager."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
general:
  front_end:
    _type: console
llms:
  mock_llm:
    _type: mock
""")
            f.flush()
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def mock_config(self):
        """Create a mock AIQ configuration."""
        config = MagicMock(spec=AIQConfig)
        config.model_dump.return_value = {"test": "value"}
        return config

    @pytest.fixture
    def dev_manager(self, temp_config_file):
        """Create a DevModeManager instance."""
        return DevModeManager(
            config_file=temp_config_file,
            reload_delay=0.1,  # Faster for testing
            max_reload_attempts=2)

    def test_init(self, temp_config_file):
        """Test DevModeManager initialization."""
        manager = DevModeManager(config_file=temp_config_file,
                                 override=(("test.key", "value"), ),
                                 reload_delay=0.5,
                                 max_reload_attempts=3)

        assert manager.config_file == temp_config_file.resolve()
        assert manager.override == (("test.key", "value"), )
        assert manager.reload_delay == 0.5
        assert manager.max_reload_attempts == 3
        assert not manager._is_running
        assert manager.config_manager is None
        assert manager.config_watcher is None

    @patch('aiq.runtime.loader.create_config_manager')
    @patch('aiq.runtime.dev_mode_manager.ConfigWatcher')
    async def test_start_success(self, mock_watcher_class, mock_create_manager, dev_manager, mock_config):
        """Test successful start of development mode."""
        # Setup mocks
        mock_manager = MagicMock(spec=ConfigManager)
        mock_manager.current_config = mock_config
        mock_create_manager.return_value = mock_manager

        mock_watcher = MagicMock()
        mock_watcher_class.return_value = mock_watcher

        # Test start
        with patch('aiq.runtime.events.get_event_manager') as mock_event_manager:
            result = await dev_manager.start()

        # Assertions
        assert result == mock_config
        assert dev_manager._is_running
        assert dev_manager.config_manager == mock_manager
        assert dev_manager.config_watcher == mock_watcher

        # Verify setup calls
        mock_create_manager.assert_called_once_with(dev_manager.config_file)
        mock_watcher.add_file.assert_called_once_with(dev_manager.config_file)
        mock_watcher.start.assert_called_once()
        mock_event_manager.return_value.register_handler.assert_called_once()

    @patch('aiq.runtime.loader.create_config_manager')
    async def test_start_with_overrides(self, mock_create_manager, temp_config_file, mock_config):
        """Test start with configuration overrides."""
        override = (("test.key", "value"), ("other.key", "other_value"))
        manager = DevModeManager(config_file=temp_config_file, override=override)

        mock_manager = MagicMock(spec=ConfigManager)
        mock_manager.current_config = mock_config
        mock_create_manager.return_value = mock_manager

        with patch('aiq.runtime.dev_mode_manager.ConfigWatcher'), \
             patch('aiq.runtime.events.get_event_manager'):
            await manager.start()

        # Verify overrides were applied
        expected_overrides = {"test.key": "value", "other.key": "other_value"}
        mock_manager.set_overrides.assert_called_once_with(expected_overrides)

    @patch('aiq.runtime.loader.create_config_manager')
    async def test_start_failure(self, mock_create_manager, dev_manager):
        """Test start failure handling."""
        mock_create_manager.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            await dev_manager.start()

        assert not dev_manager._is_running
        assert dev_manager.config_manager is None

    async def test_stop(self, dev_manager):
        """Test stopping development mode."""
        # Mock running state
        dev_manager._is_running = True
        mock_watcher = MagicMock()
        dev_manager.config_watcher = mock_watcher
        dev_manager.config_manager = MagicMock()

        await dev_manager.stop()

        assert not dev_manager._is_running
        mock_watcher.stop.assert_called_once()

    async def test_stop_not_running(self, dev_manager):
        """Test stopping when not running."""
        assert not dev_manager._is_running

        # Should not raise an error
        await dev_manager.stop()

    def test_get_current_config_with_manager(self, dev_manager, mock_config):
        """Test getting current config when manager exists."""
        mock_manager = MagicMock(spec=ConfigManager)
        mock_manager.current_config = mock_config
        dev_manager.config_manager = mock_manager

        result = dev_manager.get_current_config()
        assert result == mock_config

    def test_get_current_config_without_manager(self, dev_manager, mock_config):
        """Test getting current config when manager doesn't exist."""
        dev_manager._last_successful_config = mock_config

        result = dev_manager.get_current_config()
        assert result == mock_config

    def test_handle_file_change_valid_event(self, dev_manager):
        """Test handling valid file change events."""
        dev_manager._is_running = True

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=dev_manager.config_file)

        with patch.object(dev_manager, '_reload_lock') as mock_lock, \
             patch('threading.Thread') as mock_thread:
            mock_lock.acquire.return_value = True

            dev_manager._handle_file_change(event)

            mock_thread.assert_called_once()

    def test_handle_file_change_invalid_event(self, dev_manager):
        """Test handling invalid file change events."""
        dev_manager._is_running = True

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_DELETED, file_path=dev_manager.config_file)

        with patch('threading.Thread') as mock_thread:
            dev_manager._handle_file_change(event)

            mock_thread.assert_not_called()

    def test_handle_file_change_not_running(self, dev_manager):
        """Test handling file change when not running."""
        dev_manager._is_running = False

        event = ConfigChangeEvent(event_type=ConfigEventType.FILE_MODIFIED, file_path=dev_manager.config_file)

        with patch('threading.Thread') as mock_thread:
            dev_manager._handle_file_change(event)

            mock_thread.assert_not_called()

    def test_perform_reload_success(self, dev_manager, mock_config):
        """Test successful configuration reload."""
        mock_manager = MagicMock(spec=ConfigManager)
        mock_manager.reload_config.return_value = mock_config
        dev_manager.config_manager = mock_manager
        dev_manager._is_running = True
        dev_manager._reload_attempts = 1

        callback = MagicMock()
        dev_manager.add_reload_callback(callback)

        dev_manager._perform_reload()

        assert dev_manager._reload_attempts == 0
        assert dev_manager._last_successful_config == mock_config
        callback.assert_called_once_with(mock_config)

    def test_perform_reload_failure(self, dev_manager):
        """Test configuration reload failure."""
        mock_manager = MagicMock(spec=ConfigManager)
        mock_manager.reload_config.side_effect = Exception("Reload failed")
        dev_manager.config_manager = mock_manager
        dev_manager._is_running = True
        dev_manager._reload_attempts = 0
        dev_manager.max_reload_attempts = 2

        error_callback = MagicMock()
        dev_manager.add_error_callback(error_callback)

        dev_manager._perform_reload()

        assert dev_manager._reload_attempts == 1
        error_callback.assert_called_once()

    def test_perform_reload_max_attempts_with_rollback(self, dev_manager, mock_config):
        """Test rollback after maximum reload attempts."""
        mock_manager = MagicMock(spec=ConfigManager)
        mock_manager.reload_config.side_effect = Exception("Reload failed")
        mock_manager.rollback_config.return_value = mock_config
        dev_manager.config_manager = mock_manager
        dev_manager._is_running = True
        dev_manager._reload_attempts = 2  # At max attempts
        dev_manager.max_reload_attempts = 2

        reload_callback = MagicMock()
        dev_manager.add_reload_callback(reload_callback)

        dev_manager._perform_reload()

        assert dev_manager._reload_attempts == 0  # Reset after rollback
        mock_manager.rollback_config.assert_called_once()
        reload_callback.assert_called_with(mock_config)

    def test_add_callbacks(self, dev_manager):
        """Test adding callbacks."""
        reload_callback = MagicMock()
        error_callback = MagicMock()

        dev_manager.add_reload_callback(reload_callback)
        dev_manager.add_error_callback(error_callback)

        assert reload_callback in dev_manager._reload_callbacks
        assert error_callback in dev_manager._error_callbacks

    async def test_context_manager(self, dev_manager, mock_config):
        """Test async context manager functionality."""
        with patch.object(dev_manager, 'start', return_value=mock_config) as mock_start, \
             patch.object(dev_manager, 'stop') as mock_stop:

            async with dev_manager as result:
                assert result == mock_config

            mock_start.assert_called_once()
            mock_stop.assert_called_once()


class TestRunWithDevMode:
    """Test suite for run_with_dev_mode function."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
general:
  front_end:
    _type: console
llms:
  mock_llm:
    _type: mock
""")
            f.flush()
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def mock_workflow_runner(self):
        """Create a mock workflow runner."""
        return AsyncMock()

    @patch('aiq.runtime.dev_mode_manager.DevModeManager')
    async def test_run_with_dev_mode_success(self, mock_manager_class, temp_config_file, mock_workflow_runner):
        """Test successful run with development mode."""
        mock_config = MagicMock(spec=AIQConfig)
        mock_manager = MagicMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_config)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        mock_manager_class.return_value = mock_manager

        # Mock workflow completes quickly
        mock_workflow_runner.return_value = None

        await run_with_dev_mode(temp_config_file, (), mock_workflow_runner)

        # Verify manager was created and workflow was started
        mock_manager_class.assert_called_once()
        mock_workflow_runner.assert_called_once_with(mock_config)

    @patch('aiq.runtime.dev_mode_manager.DevModeManager')
    async def test_run_with_dev_mode_keyboard_interrupt(self,
                                                        mock_manager_class,
                                                        temp_config_file,
                                                        mock_workflow_runner):
        """Test handling keyboard interrupt."""
        mock_config = MagicMock(spec=AIQConfig)
        mock_manager = MagicMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_config)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        mock_manager_class.return_value = mock_manager

        # Mock workflow raises KeyboardInterrupt
        mock_workflow_runner.side_effect = KeyboardInterrupt()

        # Should not raise an exception
        await run_with_dev_mode(temp_config_file, (), mock_workflow_runner)

    @patch('aiq.runtime.dev_mode_manager.DevModeManager')
    async def test_run_with_dev_mode_error(self, mock_manager_class, temp_config_file, mock_workflow_runner):
        """Test handling workflow errors."""
        mock_config = MagicMock(spec=AIQConfig)
        mock_manager = MagicMock()
        mock_manager.__aenter__ = AsyncMock(return_value=mock_config)
        mock_manager.__aexit__ = AsyncMock(return_value=None)
        mock_manager_class.return_value = mock_manager

        # Mock workflow raises an error
        mock_workflow_runner.side_effect = Exception("Workflow error")

        with pytest.raises(Exception, match="Workflow error"):
            await run_with_dev_mode(temp_config_file, (), mock_workflow_runner)


class TestDevModeManagerIntegration:
    """Integration tests for DevModeManager."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
general:
  front_end:
    _type: console
llms:
  mock_llm:
    _type: mock
""")
            f.flush()
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    def test_file_change_triggers_reload(self, temp_config_file):
        """Test that file changes trigger configuration reload."""
        reload_events = []

        def reload_callback(config):
            reload_events.append(config)

        manager = DevModeManager(config_file=temp_config_file, reload_delay=0.1)
        manager.add_reload_callback(reload_callback)

        async def test_sequence():
            # Start development mode
            await manager.start()

            try:
                # Wait a bit for initial setup
                await asyncio.sleep(0.2)

                # Modify the file
                with open(temp_config_file, 'a') as f:
                    f.write("\n# Comment added")

                # Wait for reload to process
                await asyncio.sleep(0.5)

                # Check that reload was triggered
                assert len(reload_events) > 0, "Expected at least one reload event"

            finally:
                await manager.stop()

        # Run the test with timeout
        asyncio.run(asyncio.wait_for(test_sequence(), timeout=5.0))

    def test_concurrent_file_changes(self, temp_config_file):
        """Test handling of concurrent file changes."""
        reload_count = 0
        reload_lock = threading.Lock()

        def reload_callback(config):
            nonlocal reload_count
            with reload_lock:
                reload_count += 1

        manager = DevModeManager(config_file=temp_config_file, reload_delay=0.1)
        manager.add_reload_callback(reload_callback)

        async def test_sequence():
            await manager.start()

            try:
                # Make multiple rapid changes
                for i in range(3):
                    with open(temp_config_file, 'a') as f:
                        f.write(f"\n# Comment {i}")
                    await asyncio.sleep(0.05)  # Rapid changes

                # Wait for all reloads to complete
                await asyncio.sleep(1.0)

                # Should have debounced to fewer reloads than changes
                with reload_lock:
                    assert reload_count >= 1, "Expected at least one reload"
                    assert reload_count <= 3, "Expected debouncing to reduce reload count"

            finally:
                await manager.stop()

        asyncio.run(asyncio.wait_for(test_sequence(), timeout=10.0))

    @pytest.mark.timeout(10)
    def test_graceful_shutdown(self, temp_config_file):
        """Test graceful shutdown of development mode."""
        manager = DevModeManager(config_file=temp_config_file)

        async def test_sequence():
            # Start and immediately stop
            await manager.start()
            assert manager._is_running

            await manager.stop()
            assert not manager._is_running

            # Should be able to start again
            await manager.start()
            assert manager._is_running

            await manager.stop()
            assert not manager._is_running

        asyncio.run(test_sequence())


if __name__ == "__main__":
    pytest.main([__file__])

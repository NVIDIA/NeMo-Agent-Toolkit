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
"""
Unit tests for configuration manager and reload functionality.
"""

import tempfile
import time
from pathlib import Path

import pytest

from aiq.runtime.config_manager import ConfigManager
from aiq.runtime.config_manager import ConfigReloadError
from aiq.runtime.config_manager import ConfigSnapshot
from aiq.runtime.config_manager import ConfigValidationError
from aiq.runtime.events import reset_event_manager
from aiq.runtime.loader import reload_config


class TestConfigSnapshot:
    """Test cases for ConfigSnapshot class."""

    def test_init_basic(self):
        """Test basic ConfigSnapshot initialization."""
        from aiq.data_models.config import AIQConfig

        config = AIQConfig(llms={}, tools={}, workflows={})
        snapshot = ConfigSnapshot(config)

        assert snapshot.config == config
        assert snapshot.overrides == {}
        assert snapshot.timestamp is not None

    def test_init_with_overrides(self):
        """Test ConfigSnapshot initialization with overrides."""
        from aiq.data_models.config import AIQConfig

        config = AIQConfig(llms={}, tools={}, workflows={})
        overrides = {"llms.temperature": "0.7"}
        snapshot = ConfigSnapshot(config, overrides)

        assert snapshot.config == config
        assert snapshot.overrides == overrides
        assert snapshot.timestamp is not None

    def test_repr(self):
        """Test ConfigSnapshot string representation."""
        from aiq.data_models.config import AIQConfig

        config = AIQConfig(llms={}, tools={}, workflows={})
        snapshot = ConfigSnapshot(config)

        repr_str = repr(snapshot)
        assert "ConfigSnapshot" in repr_str
        assert "timestamp=" in repr_str


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir).resolve()

        # Create test configuration file
        self.config_file = (self.temp_path / "test_config.yml").resolve()
        self.config_file.write_text("""
llms:
  test_llm:
    model: "test-model"
    temperature: 0.5

tools: {}

workflows:
  test_workflow:
    name: "Test Workflow"
""")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        reset_event_manager()

    def test_init(self):
        """Test ConfigManager initialization."""
        with ConfigManager(self.config_file) as manager:
            assert manager.config_file == self.config_file
            assert manager.current_config is not None
            assert manager.reload_count == 0
            assert len(manager.get_snapshots()) == 1  # Initial snapshot

    def test_init_with_preloaded_config(self):
        """Test ConfigManager initialization with pre-loaded config."""
        from aiq.data_models.config import AIQConfig

        config = AIQConfig(llms={}, tools={}, workflows={})

        with ConfigManager(self.config_file, config) as manager:
            assert manager.current_config == config
            assert manager.reload_count == 0

    def test_set_overrides(self):
        """Test setting configuration overrides."""
        overrides = {"llms.test_llm.temperature": "0.8", "workflows.test_workflow.name": "Modified Workflow"}

        with ConfigManager(self.config_file) as manager:
            manager.set_overrides(overrides)

            config_overrides = manager.config_overrides
            assert len(config_overrides) == 2

    def test_reload_config(self):
        """Test successful configuration reload."""
        with ConfigManager(self.config_file) as manager:
            initial_count = manager.reload_count

            # Modify config file
            modified_config = """
llms:
  test_llm:
    model: "modified-model"
    temperature: 0.7

tools: {}

workflows:
  test_workflow:
    name: "Modified Workflow"
"""
            self.config_file.write_text(modified_config)

            # Reload configuration
            new_config = manager.reload_config()

            assert manager.reload_count == initial_count + 1
            assert new_config is not None

            # Check that config was actually updated
            config_dict = new_config.model_dump()
            assert config_dict['llms']['test_llm']['model'] == "modified-model"
            assert config_dict['llms']['test_llm']['temperature'] == 0.7

    def test_reload_config_validate_only(self):
        """Test configuration validation without applying changes."""
        with ConfigManager(self.config_file) as manager:
            initial_count = manager.reload_count

            # Modify config file
            modified_config = """
llms:
  test_llm:
    model: "validated-model"
    temperature: 0.9

tools: {}

workflows:
  test_workflow:
    name: "Validated Workflow"
"""
            self.config_file.write_text(modified_config)

            # Validate only
            validated_config = manager.reload_config(validate_only=True)

            # Config should not have changed
            assert manager.current_config == initial_count
            assert manager.reload_count == initial_count

            # But validation should return the new config
            validated_dict = validated_config.model_dump()
            assert validated_dict['llms']['test_llm']['model'] == "validated-model"

    def test_reload_config_invalid(self):
        """Test configuration reload with invalid config."""
        with ConfigManager(self.config_file) as manager:
            # Write invalid configuration
            self.config_file.write_text("invalid: yaml: content: [")

            with pytest.raises(ConfigValidationError):
                manager.reload_config()

    def test_reload_config_with_overrides(self):
        """Test that overrides are preserved during reload."""
        overrides = {"llms.test_llm.temperature": "0.8"}

        with ConfigManager(self.config_file) as manager:
            manager.set_overrides(overrides)

            # Modify config file
            modified_config = """
llms:
  test_llm:
    model: "new-model"
    temperature: 0.5

tools: {}

workflows:
  test_workflow:
    name: "New Workflow"
"""
            self.config_file.write_text(modified_config)

            # Reload configuration
            manager.reload_config()

            # Overrides should still be applied
            current_overrides = manager.config_overrides
            assert "llms.test_llm.temperature" in current_overrides

    def test_rollback_config(self):
        """Test configuration rollback functionality."""
        with ConfigManager(self.config_file) as manager:
            # Get initial config
            initial_config = manager.current_config

            # Make first modification
            modified_config1 = """
llms:
  test_llm:
    model: "modified-1"
    temperature: 0.6

tools: {}

workflows:
  test_workflow:
    name: "Modified 1"
"""
            self.config_file.write_text(modified_config1)
            manager.reload_config()

            # Make second modification
            modified_config2 = """
llms:
  test_llm:
    model: "modified-2"
    temperature: 0.7

tools: {}

workflows:
  test_workflow:
    name: "Modified 2"
"""
            self.config_file.write_text(modified_config2)
            manager.reload_config()

            # Rollback one step
            rolled_back_config = manager.rollback_config(1)

            # Should be back to first modification
            config_dict = rolled_back_config.model_dump()
            assert config_dict['llms']['test_llm']['model'] == "modified-1"

    def test_rollback_config_too_many_steps(self):
        """Test rollback with too many steps."""
        with ConfigManager(self.config_file) as manager:
            with pytest.raises(ConfigReloadError, match="Cannot rollback"):
                manager.rollback_config(10)

    def test_snapshots_management(self):
        """Test configuration snapshots management."""
        with ConfigManager(self.config_file) as manager:
            # Should start with one snapshot
            snapshots = manager.get_snapshots()
            assert len(snapshots) == 1

            # Reload to create more snapshots
            for i in range(3):
                modified_config = f"""
llms:
  test_llm:
    model: "model-{i}"
    temperature: 0.5

tools: {{}}

workflows:
  test_workflow:
    name: "Workflow {i}"
"""
                self.config_file.write_text(modified_config)
                manager.reload_config()

            # Should have 4 snapshots total (initial + 3 reloads)
            snapshots = manager.get_snapshots()
            assert len(snapshots) == 4

            # Clear snapshots
            manager.clear_snapshots()
            snapshots = manager.get_snapshots()
            assert len(snapshots) == 1  # Should keep current snapshot

    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        nonexistent_file = self.temp_path / "nonexistent.yml"

        with pytest.raises((ConfigValidationError, FileNotFoundError)):
            ConfigManager(nonexistent_file)

    def test_context_manager(self):
        """Test ConfigManager as context manager."""
        with ConfigManager(self.config_file) as manager:
            assert isinstance(manager, ConfigManager)

        # Should work without errors after context exit


class TestReloadConfigFunction:
    """Test cases for the reload_config function."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir).resolve()

        # Create test configuration file
        self.config_file = (self.temp_path / "test_config.yml").resolve()
        self.config_file.write_text("""
llms:
  test_llm:
    model: "test-model"
    temperature: 0.5

tools: {}

workflows:
  test_workflow:
    name: "Test Workflow"
""")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        reset_event_manager()

    def test_reload_config_success(self):
        """Test successful configuration reload."""
        # Modify config file
        modified_config = """
llms:
  test_llm:
    model: "reloaded-model"
    temperature: 0.8

tools: {}

workflows:
  test_workflow:
    name: "Reloaded Workflow"
"""
        self.config_file.write_text(modified_config)

        # Reload configuration
        new_config = reload_config(self.config_file)

        # Verify changes
        config_dict = new_config.model_dump()
        assert config_dict['llms']['test_llm']['model'] == "reloaded-model"
        assert config_dict['llms']['test_llm']['temperature'] == 0.8

    def test_reload_config_validate_only(self):
        """Test configuration validation only."""
        # Modify config file
        modified_config = """
llms:
  test_llm:
    model: "validated-model"
    temperature: 0.9

tools: {}

workflows:
  test_workflow:
    name: "Validated Workflow"
"""
        self.config_file.write_text(modified_config)

        # Validate only
        validated_config = reload_config(self.config_file, validate_only=True)

        # Should return valid config
        config_dict = validated_config.model_dump()
        assert config_dict['llms']['test_llm']['model'] == "validated-model"

    def test_reload_config_invalid(self):
        """Test configuration reload with invalid config."""
        # Write invalid configuration
        self.config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigValidationError):
            reload_config(self.config_file)

    def test_reload_config_nonexistent_file(self):
        """Test reload with non-existent file."""
        nonexistent_file = self.temp_path / "nonexistent.yml"

        with pytest.raises((ConfigValidationError, ConfigReloadError)):
            reload_config(nonexistent_file)


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with file watching."""

    def setup_method(self):
        """Set up test fixtures."""
        reset_event_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir).resolve()

        # Create test configuration file
        self.config_file = (self.temp_path / "test_config.yml").resolve()
        self.config_file.write_text("""
llms:
  test_llm:
    model: "test-model"
    temperature: 0.5

tools: {}

workflows:
  test_workflow:
    name: "Test Workflow"
""")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        reset_event_manager()

    def test_config_manager_with_file_watcher(self):
        """Test ConfigManager integration with file watcher."""
        from aiq.runtime.config_watcher import ConfigWatcher
        from aiq.runtime.events import get_event_manager

        events_received = []

        def event_handler(event):
            events_received.append(event)

        get_event_manager().register_handler(event_handler)

        with ConfigWatcher() as watcher:
            watcher.add_file(self.config_file)

            with ConfigManager(self.config_file) as manager:
                # Modify the config file
                modified_config = """
llms:
  test_llm:
    model: "watched-model"
    temperature: 0.7

tools: {}

workflows:
  test_workflow:
    name: "Watched Workflow"
"""
                self.config_file.write_text(modified_config)

                # Wait for file system events
                time.sleep(0.3)

                # Should have received file change events
                assert len(events_received) > 0

                # Manually reload config
                new_config = manager.reload_config()

                # Verify the reload worked
                config_dict = new_config.model_dump()
                assert config_dict['llms']['test_llm']['model'] == "watched-model"

    def test_end_to_end_workflow(self):
        """Test complete end-to-end configuration reload workflow."""
        with ConfigManager(self.config_file) as manager:
            # Set some overrides
            overrides = {"llms.test_llm.temperature": "0.9"}
            manager.set_overrides(overrides)

            # Get initial state
            initial_snapshots = len(manager.get_snapshots())

            # Modify config multiple times
            for i in range(3):
                modified_config = f"""
llms:
  test_llm:
    model: "workflow-model-{i}"
    temperature: 0.5

tools: {{}}

workflows:
  test_workflow:
    name: "Workflow {i}"
"""
                self.config_file.write_text(modified_config)
                manager.reload_config()

            # Should have more snapshots
            assert len(manager.get_snapshots()) == initial_snapshots + 3

            # Rollback to previous version
            rolled_back = manager.rollback_config(2)

            # Verify rollback
            config_dict = rolled_back.model_dump()
            assert config_dict['llms']['test_llm']['model'] == "workflow-model-0"

            # Overrides should still be preserved
            assert len(manager.config_overrides) > 0

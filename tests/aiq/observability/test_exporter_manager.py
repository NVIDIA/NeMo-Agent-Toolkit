"""
Unit tests for ExporterManager.

This test suite focuses on the critical copy-on-write functionality that was implemented
to fix concurrency issues, along with comprehensive testing of all manager functionality.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from aiq.builder.context import AIQContextState
from aiq.observability.exporter.base_exporter import BaseExporter
from aiq.observability.exporter_manager import ExporterManager


class MockExporter(BaseExporter):
    """Mock exporter for testing."""

    def __init__(self, name: str = "test_exporter", context_state: AIQContextState | None = None):
        super().__init__(context_state)
        self._name = name
        self._export_called = False
        self._start_called = False
        self._stop_called = False
        self._wait_ready_called = False
        self._isolated_instance_created = False

    @property
    def name(self) -> str:
        return self._name

    def export(self, event):
        """Mock export method."""
        self._export_called = True

    @asynccontextmanager
    async def start(self):
        """Mock start method."""
        self._start_called = True
        self._ready_event.set()
        try:
            yield
        finally:
            self._stop_called = True

    async def wait_ready(self):
        """Mock wait_ready method."""
        self._wait_ready_called = True
        await self._ready_event.wait()

    def create_isolated_instance(self, context_state: AIQContextState) -> "MockExporter":
        """Create isolated instance for testing copy-on-write functionality."""
        isolated = MockExporter(f"{self._name}_isolated", context_state)
        isolated._isolated_instance_created = True
        return isolated


class MockExporterWithoutIsolation(BaseExporter):
    """Mock exporter without isolation support for testing fallback behavior."""

    def __init__(self, name: str = "no_isolation_exporter", context_state: AIQContextState | None = None):
        super().__init__(context_state)
        self._name = name
        # Remove the create_isolated_instance method using built-in delattr
        delattr(self, 'create_isolated_instance')

    @property
    def name(self) -> str:
        return self._name

    def export(self, event):
        """Mock export method."""
        pass

    @asynccontextmanager
    async def start(self):
        """Mock start method."""
        self._ready_event.set()
        yield

    async def wait_ready(self):
        """Mock wait_ready method."""
        await self._ready_event.wait()


@pytest.fixture
def mock_context_state():
    """Create a mock context state for testing."""
    context = Mock(spec=AIQContextState)
    context.conversation_id = Mock()
    context.conversation_id.get.return_value = "test-conversation-123"
    return context


@pytest.fixture
def exporter_manager():
    """Create an ExporterManager instance for testing."""
    return ExporterManager(shutdown_timeout=1)  # Short timeout for faster tests


@pytest.fixture
def mock_exporter():
    """Create a mock exporter for testing."""
    return MockExporter()


@pytest.fixture
def mock_exporter2():
    """Create a second mock exporter for testing."""
    return MockExporter("test_exporter2")


class TestExporterManagerInit:
    """Test ExporterManager initialization."""

    def test_init_default_timeout(self):
        """Test ExporterManager initialization with default timeout."""
        manager = ExporterManager()
        assert manager._shutdown_timeout == 120
        assert manager._running is False
        assert manager._tasks == {}
        assert manager._exporter_registry == {}
        assert manager._is_registry_shared is False

    def test_init_custom_timeout(self):
        """Test ExporterManager initialization with custom timeout."""
        manager = ExporterManager(shutdown_timeout=60)
        assert manager._shutdown_timeout == 60

    def test_create_with_shared_registry(self):
        """Test creating manager with shared registry."""
        shared_registry: dict[str, BaseExporter] = {"test": MockExporter()}
        manager = ExporterManager._create_with_shared_registry(60, shared_registry)

        assert manager._shutdown_timeout == 60
        assert manager._exporter_registry is shared_registry  # Same object reference
        assert manager._is_registry_shared is True
        assert manager._running is False
        assert manager._tasks == {}


class TestCopyOnWriteFunctionality:
    """Test the critical copy-on-write functionality that fixes concurrency issues."""

    def test_shared_registry_initially(self):
        """Test that shared registry works initially."""
        original_registry: dict[str, BaseExporter] = {"test": MockExporter()}
        manager = ExporterManager._create_with_shared_registry(120, original_registry)

        # Registry should be shared
        assert manager._exporter_registry is original_registry
        assert manager._is_registry_shared is True

    def test_ensure_registry_owned_copies_registry(self):
        """Test that _ensure_registry_owned creates a copy when registry is shared."""
        original_registry: dict[str, BaseExporter] = {"test": MockExporter()}
        manager = ExporterManager._create_with_shared_registry(120, original_registry)

        # Initially shared
        assert manager._exporter_registry is original_registry
        assert manager._is_registry_shared is True

        # Call _ensure_registry_owned
        manager._ensure_registry_owned()

        # Should now be owned (copied)
        assert manager._exporter_registry is not original_registry
        assert manager._exporter_registry == original_registry  # Same content
        assert manager._is_registry_shared is False

    def test_ensure_registry_owned_no_copy_when_already_owned(self):
        """Test that _ensure_registry_owned doesn't copy when already owned."""
        manager = ExporterManager()
        original_registry = manager._exporter_registry

        # Initially owned
        assert manager._is_registry_shared is False

        # Call _ensure_registry_owned
        manager._ensure_registry_owned()

        # Should remain the same object
        assert manager._exporter_registry is original_registry
        assert manager._is_registry_shared is False

    def test_add_exporter_triggers_copy_on_write(self):
        """Test that adding an exporter triggers copy-on-write when registry is shared."""
        original_registry: dict[str, BaseExporter] = {"existing": MockExporter("existing")}
        manager = ExporterManager._create_with_shared_registry(120, original_registry)
        new_exporter = MockExporter("new")

        # Initially shared
        assert manager._exporter_registry is original_registry
        assert manager._is_registry_shared is True

        # Add exporter should trigger copy-on-write
        manager.add_exporter("new", new_exporter)

        # Registry should now be owned (copied)
        assert manager._exporter_registry is not original_registry
        assert manager._is_registry_shared is False
        assert "existing" in manager._exporter_registry
        assert "new" in manager._exporter_registry
        assert manager._exporter_registry["new"] is new_exporter

        # Original registry should be unchanged
        assert "new" not in original_registry

    def test_remove_exporter_triggers_copy_on_write(self):
        """Test that removing an exporter triggers copy-on-write when registry is shared."""
        original_registry: dict[str, BaseExporter] = {"test1": MockExporter("test1"), "test2": MockExporter("test2")}
        manager = ExporterManager._create_with_shared_registry(120, original_registry)

        # Initially shared
        assert manager._exporter_registry is original_registry
        assert manager._is_registry_shared is True

        # Remove exporter should trigger copy-on-write
        manager.remove_exporter("test1")

        # Registry should now be owned (copied)
        assert manager._exporter_registry is not original_registry
        assert manager._is_registry_shared is False
        assert "test1" not in manager._exporter_registry
        assert "test2" in manager._exporter_registry

        # Original registry should be unchanged
        assert "test1" in original_registry

    def test_concurrent_modifications_isolated(self):
        """Test that concurrent modifications to different managers are isolated."""
        original_registry: dict[str, BaseExporter] = {"shared": MockExporter("shared")}

        # Create two managers sharing the same registry
        manager1 = ExporterManager._create_with_shared_registry(120, original_registry)
        manager2 = ExporterManager._create_with_shared_registry(120, original_registry)

        # Both should initially share the same registry
        assert manager1._exporter_registry is original_registry
        assert manager2._exporter_registry is original_registry

        # Modify manager1
        manager1.add_exporter("manager1_only", MockExporter("manager1_only"))

        # manager1 should have its own copy now
        assert manager1._exporter_registry is not original_registry
        assert "manager1_only" in manager1._exporter_registry
        assert "shared" in manager1._exporter_registry

        # manager2 should still share original registry
        assert manager2._exporter_registry is original_registry
        assert "manager1_only" not in manager2._exporter_registry
        assert "shared" in manager2._exporter_registry

        # Modify manager2
        manager2.add_exporter("manager2_only", MockExporter("manager2_only"))

        # manager2 should now have its own copy
        assert manager2._exporter_registry is not original_registry
        assert manager2._exporter_registry is not manager1._exporter_registry
        assert "manager2_only" in manager2._exporter_registry
        assert "shared" in manager2._exporter_registry

        # Managers should be completely isolated
        assert "manager1_only" not in manager2._exporter_registry
        assert "manager2_only" not in manager1._exporter_registry


class TestExporterManagerBasicFunctionality:
    """Test basic ExporterManager functionality."""

    def test_add_exporter(self, exporter_manager, mock_exporter):
        """Test adding an exporter."""
        exporter_manager.add_exporter("test", mock_exporter)

        assert "test" in exporter_manager._exporter_registry
        assert exporter_manager._exporter_registry["test"] is mock_exporter

    def test_add_exporter_overwrite_warning(self, exporter_manager, mock_exporter, mock_exporter2, caplog):
        """Test that adding an exporter with existing name logs a warning."""
        exporter_manager.add_exporter("test", mock_exporter)

        with caplog.at_level(logging.WARNING):
            exporter_manager.add_exporter("test", mock_exporter2)

        assert "already registered. Overwriting" in caplog.text
        assert exporter_manager._exporter_registry["test"] is mock_exporter2

    def test_remove_exporter(self, exporter_manager, mock_exporter):
        """Test removing an exporter."""
        exporter_manager.add_exporter("test", mock_exporter)
        exporter_manager.remove_exporter("test")

        assert "test" not in exporter_manager._exporter_registry

    def test_remove_nonexistent_exporter(self, exporter_manager):
        """Test removing a non-existent exporter raises ValueError."""
        with pytest.raises(ValueError, match="Cannot remove exporter 'nonexistent' because it is not registered"):
            exporter_manager.remove_exporter("nonexistent")

    def test_get_exporter(self, exporter_manager, mock_exporter):
        """Test getting an exporter."""
        exporter_manager.add_exporter("test", mock_exporter)
        retrieved = exporter_manager.get_exporter("test")

        assert retrieved is mock_exporter

    def test_get_nonexistent_exporter(self, exporter_manager):
        """Test getting a non-existent exporter raises ValueError."""
        with pytest.raises(ValueError, match="Cannot get exporter 'nonexistent' because it is not registered"):
            exporter_manager.get_exporter("nonexistent")

    async def test_get_all_exporters(self, exporter_manager, mock_exporter, mock_exporter2):
        """Test getting all exporters."""
        exporter_manager.add_exporter("test1", mock_exporter)
        exporter_manager.add_exporter("test2", mock_exporter2)

        all_exporters = await exporter_manager.get_all_exporters()

        assert len(all_exporters) == 2
        assert all_exporters["test1"] is mock_exporter
        assert all_exporters["test2"] is mock_exporter2


class TestCreateIsolatedExporters:
    """Test isolated exporter creation functionality."""

    def test_create_isolated_exporters_with_isolation_support(self, exporter_manager, mock_context_state):
        """Test creating isolated exporters when exporters support isolation."""
        mock_exporter = MockExporter("test1")
        mock_exporter2 = MockExporter("test2")

        exporter_manager.add_exporter("test1", mock_exporter)
        exporter_manager.add_exporter("test2", mock_exporter2)

        isolated = exporter_manager.create_isolated_exporters(mock_context_state)

        assert len(isolated) == 2
        assert "test1" in isolated
        assert "test2" in isolated

        # Should be different instances
        assert isolated["test1"] is not mock_exporter
        assert isolated["test2"] is not mock_exporter2

        # Should be isolated instances
        assert isolated["test1"]._isolated_instance_created is True
        assert isolated["test2"]._isolated_instance_created is True

    def test_create_isolated_exporters_without_isolation_support(self, exporter_manager, mock_context_state, caplog):
        """Test creating isolated exporters when exporters don't support isolation."""

        # Create a mock exporter without the create_isolated_instance method
        class SimpleExporter(BaseExporter):

            def __init__(self, name):
                super().__init__()
                self._name = name

            @property
            def name(self):
                return self._name

            def export(self, event):
                pass

            @asynccontextmanager
            async def start(self):
                self._ready_event.set()
                yield

            async def wait_ready(self):
                await self._ready_event.wait()

            def __getattribute__(self, name):
                if name == 'create_isolated_instance':
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute 'create_isolated_instance'")
                return super().__getattribute__(name)

        simple_exporter = SimpleExporter("no_isolation")
        exporter_manager.add_exporter("no_isolation", simple_exporter)

        with caplog.at_level(logging.WARNING):
            isolated = exporter_manager.create_isolated_exporters(mock_context_state)

        assert "doesn't support isolation" in caplog.text
        assert len(isolated) == 1
        assert isolated["no_isolation"] is simple_exporter  # Same instance

    def test_create_isolated_exporters_default_context(self, exporter_manager):
        """Test creating isolated exporters with default context state."""
        mock_exporter = MockExporter("test")
        exporter_manager.add_exporter("test", mock_exporter)

        with patch('aiq.builder.context.AIQContextState.get') as mock_get:
            mock_context = Mock(spec=AIQContextState)
            mock_get.return_value = mock_context

            isolated = exporter_manager.create_isolated_exporters()

            assert len(isolated) == 1
            mock_get.assert_called_once()


class TestExporterManagerLifecycle:
    """Test ExporterManager lifecycle management."""

    async def test_start_and_stop_context_manager(self, exporter_manager, mock_exporter):
        """Test the start/stop context manager functionality."""
        exporter_manager.add_exporter("test", mock_exporter)

        async with exporter_manager.start():
            assert exporter_manager._running is True
            assert mock_exporter._wait_ready_called is True
            assert mock_exporter._start_called is True

        # After context exit, should be stopped
        assert exporter_manager._running is False
        assert mock_exporter._stop_called is True

    async def test_start_with_isolated_context(self, exporter_manager, mock_context_state):
        """Test starting with isolated context state."""
        mock_exporter = MockExporter("test")
        exporter_manager.add_exporter("test", mock_exporter)

        async with exporter_manager.start(mock_context_state):
            assert exporter_manager._running is True
            # The isolated exporter should be started, not the original
            assert mock_exporter._start_called is False  # Original not started

        assert exporter_manager._running is False

    async def test_start_already_running_raises_error(self, exporter_manager, mock_exporter):
        """Test that starting when already running raises RuntimeError."""
        exporter_manager.add_exporter("test", mock_exporter)

        async with exporter_manager.start():
            with pytest.raises(RuntimeError, match="already running"):
                async with exporter_manager.start():
                    pass

    async def test_stop_not_running_does_nothing(self, exporter_manager):
        """Test that stopping when not running does nothing."""
        # Should not raise any error
        await exporter_manager.stop()
        assert exporter_manager._running is False

    async def test_exporter_task_exception_handling(self, exporter_manager, caplog):
        """Test that exceptions in exporter tasks are properly caught and logged."""

        # Create a mock exporter that raises an exception
        class FailingExporter(MockExporter):

            @asynccontextmanager
            async def start(self):
                self._ready_event.set()
                raise RuntimeError("Test exception")
                yield  # Never reached

        failing_exporter = FailingExporter("failing")
        exporter_manager.add_exporter("failing", failing_exporter)

        with caplog.at_level(logging.ERROR):
            # The context manager should complete successfully even with failing exporters
            async with exporter_manager.start():
                pass  # Exception should be caught and logged, not propagated

        # Verify the exception was logged
        assert "Failed to run exporter" in caplog.text
        assert "Test exception" in caplog.text

    async def test_shutdown_timeout_handling(self, caplog):
        """Test handling of shutdown timeout."""

        class SlowExporter(MockExporter):

            @asynccontextmanager
            async def start(self):
                self._ready_event.set()
                try:
                    # Simulate slow shutdown
                    await asyncio.sleep(10)  # Longer than timeout
                    yield
                except asyncio.CancelledError:
                    # Simulate a stuck exporter that doesn't respond to cancellation
                    await asyncio.sleep(10)  # This will cause timeout

        manager = ExporterManager(shutdown_timeout=0.1)  # Very short timeout
        slow_exporter = SlowExporter("slow")
        manager.add_exporter("slow", slow_exporter)

        with caplog.at_level(logging.WARNING):
            async with manager.start():
                pass  # Will timeout on exit

        assert "did not shut down in time" in caplog.text


class TestExporterManagerFactoryMethods:
    """Test ExporterManager factory methods."""

    def test_from_exporters(self, mock_exporter, mock_exporter2):
        """Test creating ExporterManager from exporters dict."""
        exporters = {"test1": mock_exporter, "test2": mock_exporter2}

        manager = ExporterManager.from_exporters(exporters, shutdown_timeout=60)

        assert manager._shutdown_timeout == 60
        assert len(manager._exporter_registry) == 2
        assert manager._exporter_registry["test1"] is mock_exporter
        assert manager._exporter_registry["test2"] is mock_exporter2

    def test_get_method_creates_shared_copy(self, exporter_manager, mock_exporter):
        """Test that get() method creates a copy with shared registry."""
        exporter_manager.add_exporter("test", mock_exporter)

        copy = exporter_manager.get()

        # Should be different instances
        assert copy is not exporter_manager

        # But should share the same registry (copy-on-write)
        assert copy._exporter_registry is exporter_manager._exporter_registry
        assert copy._is_registry_shared is True
        assert copy._shutdown_timeout == exporter_manager._shutdown_timeout


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety aspects."""

    async def test_concurrent_start_operations(self, exporter_manager, mock_exporter):
        """Test that concurrent start operations are properly locked."""
        exporter_manager.add_exporter("test", mock_exporter)

        # Try to start concurrently - second should fail
        async def start_operation():
            async with exporter_manager.start():
                await asyncio.sleep(0.1)  # Hold the context briefly

        task1 = asyncio.create_task(start_operation())
        await asyncio.sleep(0.05)  # Let first task start

        with pytest.raises(RuntimeError, match="already running"):
            async with exporter_manager.start():
                pass

        await task1  # Clean up

    async def test_concurrent_registry_modifications(self):
        """Test concurrent modifications to shared registries."""
        shared_registry: dict[str, BaseExporter] = {"shared": MockExporter("shared")}

        async def modify_manager(manager_num: int):
            manager = ExporterManager._create_with_shared_registry(120, shared_registry)
            await asyncio.sleep(0.01)  # Small delay to increase chance of race condition
            manager.add_exporter(f"exporter_{manager_num}", MockExporter(f"exporter_{manager_num}"))
            return manager

        # Create multiple managers concurrently
        tasks = [modify_manager(i) for i in range(10)]
        managers = await asyncio.gather(*tasks)

        # Each manager should have its own registry after modification
        for i, manager in enumerate(managers):
            assert manager._is_registry_shared is False
            assert f"exporter_{i}" in manager._exporter_registry
            assert "shared" in manager._exporter_registry

            # Other managers' exporters should not be present
            for j in range(10):
                if i != j:
                    assert f"exporter_{j}" not in manager._exporter_registry


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Integration tests simulating real-world usage scenarios."""

    async def test_workflow_execution_simulation(self, mock_context_state):
        """Test simulation of multiple concurrent workflow executions."""
        # Create a base manager with some exporters
        base_manager = ExporterManager()
        base_manager.add_exporter("metrics", MockExporter("metrics"))
        base_manager.add_exporter("traces", MockExporter("traces"))

        async def simulate_workflow_execution(workflow_id: int):
            # Each workflow gets its own manager copy
            workflow_manager = base_manager.get()

            # Start the workflow with isolated context
            async with workflow_manager.start(mock_context_state):
                # Simulate some work
                await asyncio.sleep(0.01)
                return workflow_id

        # Run multiple workflows concurrently
        workflow_tasks = [simulate_workflow_execution(i) for i in range(5)]
        results = await asyncio.gather(*workflow_tasks)

        assert results == [0, 1, 2, 3, 4]

    async def test_dynamic_exporter_management(self, exporter_manager):
        """Test dynamic addition and removal of exporters during lifecycle."""
        initial_exporter = MockExporter("initial")
        exporter_manager.add_exporter("initial", initial_exporter)

        async with exporter_manager.start():
            # Add exporter during runtime (won't be started automatically)
            runtime_exporter = MockExporter("runtime")
            exporter_manager.add_exporter("runtime", runtime_exporter)

            # Remove initial exporter
            exporter_manager.remove_exporter("initial")

        # Verify final state
        assert "initial" not in exporter_manager._exporter_registry
        assert "runtime" in exporter_manager._exporter_registry

    async def test_error_recovery_scenario(self, caplog):
        """Test that the manager handles exporter failures gracefully."""
        manager = ExporterManager()

        good_exporter = MockExporter("good")

        class RecoveringExporter(MockExporter):

            def __init__(self, name):
                super().__init__(name)
                self.attempt_count = 0

            @asynccontextmanager
            async def start(self):
                self.attempt_count += 1
                self._ready_event.set()
                if self.attempt_count == 1:
                    raise RuntimeError("First attempt fails")
                yield  # Second attempt succeeds

        recovering_exporter = RecoveringExporter("recovering")

        manager.add_exporter("good", good_exporter)
        manager.add_exporter("recovering", recovering_exporter)

        # Manager should handle the failure gracefully
        with caplog.at_level(logging.ERROR):
            async with manager.start():
                pass  # Should complete successfully despite one exporter failing

        # Verify the exception was logged
        assert "Failed to run exporter" in caplog.text
        assert "First attempt fails" in caplog.text

        # Good exporter should have been started and stopped
        assert good_exporter._start_called is True
        assert good_exporter._stop_called is True

        # Recovering exporter should have attempted once
        assert recovering_exporter.attempt_count == 1

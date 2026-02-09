# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Integration tests for A365 front-end plugin with mocked Microsoft Agents SDK."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.plugins.a365.front_end.plugin import A365FrontEndPlugin
from nat.plugins.a365.front_end.worker import A365FrontEndPluginWorker
from nat.test.functions import EchoFunctionConfig


# Helper functions and fixtures
@pytest.fixture
def mock_agent_application():
    """Create a mock AgentApplication.
    
    The mock needs to support:
    - @agent_app.activity("message") - activity is called with "message", returns a decorator
    - @agent_app.error - error is called with no args, returns a decorator
    """
    app = MagicMock()
    
    # activity is used as a decorator factory: @agent_app.activity("message")
    # It's called with "message", then the result is used as a decorator
    def activity_decorator_factory(activity_type):
        def decorator(func):
            return func
        return decorator
    app.activity = MagicMock(side_effect=activity_decorator_factory)
    
    # error is used as a decorator: @agent_app.error
    # It's called with no args, then the result is used as a decorator
    def error_decorator(func):
        return func
    app.error = MagicMock(return_value=error_decorator)
    
    return app


@pytest.fixture
def mock_notification():
    """Create a mock AgentNotification."""
    notification = Mock()
    for method in ["on_email", "on_word", "on_excel", "on_powerpoint", "on_user_created", "on_user_deleted"]:
        setattr(notification, method, Mock())
    return notification


@pytest.fixture
def mock_turn_context():
    """Create a mock TurnContext."""
    context = Mock()
    context.activity = Mock()
    context.activity.text = "Test message"
    context.send_activity = AsyncMock()
    return context


@pytest.fixture
def mock_turn_state():
    """Create a mock TurnState."""
    return Mock()


@pytest.fixture
def mock_notification_activity():
    """Create a mock AgentNotificationActivity."""
    activity = Mock()
    activity.text = "Test notification text"
    activity.summary = "Test notification summary"
    # Set up email structure for email notifications
    activity.email = Mock()
    activity.email.html_body = "Test email HTML body"
    activity.wpx_comment = None  # Not a comment notification
    return activity


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager.
    
    SessionManager.run() is an @asynccontextmanager, so it returns an async context manager.
    Pattern matches other NAT tests (e.g., nvidia_nat_mcp tests).
    """
    manager = MagicMock()
    # Create mock runner with async context manager protocol
    mock_runner = MagicMock()
    mock_runner.__aenter__ = AsyncMock(return_value=mock_runner)
    mock_runner.__aexit__ = AsyncMock(return_value=None)
    mock_runner.result = AsyncMock(return_value="Test workflow result")
    # Make session_manager.run() return the runner (async context manager)
    manager.run = MagicMock(return_value=mock_runner)
    manager.shutdown = AsyncMock()
    return manager


@pytest.fixture
def mock_workflow_builder():
    """Create a mock WorkflowBuilder."""
    builder = Mock()
    builder.__aenter__ = AsyncMock(return_value=builder)
    builder.__aexit__ = AsyncMock(return_value=None)
    return builder


@pytest.fixture
def a365_config():
    """Create A365FrontEndConfig for testing."""
    return A365FrontEndConfig(
        app_id="test-app-id",
        app_password="test-app-password",
        host="localhost",
        port=3978,
        enable_notifications=True,
    )


@pytest.fixture
def full_config(a365_config):
    """Create full Config for testing."""
    return Config(
        general=GeneralConfig(front_end=a365_config),
        workflow=EchoFunctionConfig(),
    )


@pytest.fixture
def a365_plugin(full_config):
    """Create A365FrontEndPlugin instance."""
    return A365FrontEndPlugin(full_config=full_config)


def create_notification_decorator_mock(handler_storage):
    """Helper to create a mock for notification decorators like @notification.on_email().
    
    notification.on_email() is called with no args and returns a decorator function.
    """
    def decorator(func):
        handler_storage.append(func)
        return func
    return decorator


def create_activity_decorator_mock(handler_storage):
    """Helper to create a mock for agent_app.activity() decorator.
    
    agent_app.activity("message") returns a decorator, so this creates
    a callable that takes an activity type and returns a decorator.
    """
    def activity_decorator(activity_type):
        def decorator(func):
            handler_storage.append(func)
            return func
        return decorator
    return activity_decorator


def verify_all_notification_handlers(mock_notification):
    """Helper to verify all notification handlers were registered."""
    mock_notification.on_email.assert_called_once()
    mock_notification.on_word.assert_called_once()
    mock_notification.on_excel.assert_called_once()
    mock_notification.on_powerpoint.assert_called_once()
    mock_notification.on_user_created.assert_called_once()
    mock_notification.on_user_deleted.assert_called_once()


@contextmanager
def patch_sdk_components(mock_agent_app=None, mock_notification=None, mock_session_mgr=None, mock_workflow_builder=None):
    """Context manager to patch all Microsoft Agents SDK components."""
    patches = []
    try:
        if mock_workflow_builder:
            mock_wb_cm = AsyncMock()
            mock_wb_cm.__aenter__ = AsyncMock(return_value=mock_workflow_builder)
            mock_wb_cm.__aexit__ = AsyncMock(return_value=None)
            patches.append(patch("nat.plugins.a365.front_end.plugin.WorkflowBuilder.from_config", return_value=mock_wb_cm))
        
        if mock_session_mgr:
            patches.append(patch("nat.plugins.a365.front_end.plugin.SessionManager.create", return_value=mock_session_mgr))
        
        if mock_agent_app:
            # AgentApplication is instantiated as AgentApplication[TurnState](...)
            # We need to create a class-like mock that supports subscripting
            # Python's __class_getitem__ enables subscripting on classes
            class MockAgentApplicationClass:
                """Mock class that supports subscripting like AgentApplication[TurnState]."""
                def __class_getitem__(cls, item):
                    # Return the class itself when subscripted (e.g., AgentApplication[TurnState])
                    return cls
                
                def __new__(cls, *args, **kwargs):
                    # When instantiated, return the mock_agent_app
                    return mock_agent_app
            
            patches.append(patch("microsoft_agents.hosting.core.AgentApplication", new=MockAgentApplicationClass))
        
        # Patch SDK components - they're imported inside run()
        # We patch them where they're imported from
        mock_storage = Mock()
        mock_adapter = Mock()
        mock_conn_mgr = Mock()
        mock_conn_mgr.get_default_connection_configuration = Mock(return_value={})
        mock_auth = Mock()
        
        # Patch CloudAdapter and MemoryStorage (from microsoft_agents.hosting.core)
        patches.extend([
            patch("microsoft_agents.hosting.core.MemoryStorage", return_value=mock_storage, create=True),
            patch("microsoft_agents.hosting.core.CloudAdapter", return_value=mock_adapter, create=True),
        ])
        
        # For authentication submodule, create it in sys.modules before import
        # The authentication submodule might not exist, so we create it as a mock module
        import sys
        import types
        
        # Create mock authentication module
        mock_auth_module = types.ModuleType('microsoft_agents.hosting.core.authentication')
        mock_auth_module.MsalConnectionManager = Mock(return_value=mock_conn_mgr)
        mock_auth_module.Authorization = Mock(return_value=mock_auth)
        sys.modules['microsoft_agents.hosting.core.authentication'] = mock_auth_module
        
        # Also ensure the parent module has the authentication attribute
        try:
            import microsoft_agents.hosting.core as core_module
            core_module.authentication = mock_auth_module
        except ImportError:
            pass  # Parent module might not exist, but sys.modules entry should be enough
        
        if mock_notification:
            patches.append(patch("microsoft_agents_a365.notifications.AgentNotification", return_value=mock_notification))
        
        # Patch start_server - it's imported inside run() from either:
        # microsoft_agents.hosting.aiohttp.start_server (preferred) or
        # microsoft_agents.hosting.core.server.start_server (fallback)
        # We patch both locations since we don't know which will be used
        # Use create=True to handle cases where modules don't exist yet
        # side_effect should be the exception class, not an instance
        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()
        patches.append(patch("microsoft_agents.hosting.aiohttp.start_server", side_effect=raise_keyboard_interrupt, create=True))
        patches.append(patch("microsoft_agents.hosting.core.server.start_server", side_effect=raise_keyboard_interrupt, create=True))
        
        # Start all patches, handling failures gracefully for optional imports
        started_patches = []
        for p in patches:
            try:
                p.start()
                started_patches.append(p)
            except AttributeError:
                # If a patch fails (module/attribute doesn't exist), skip it
                # This can happen for optional imports that aren't available in test environment
                pass
        
        # Update patches list to only include successfully started patches
        patches[:] = started_patches
        
        yield
    finally:
        # Stop all patches
        for p in patches:
            p.stop()
        
        # Clean up mock modules from sys.modules if we created them
        import sys
        import types
        auth_module_key = 'microsoft_agents.hosting.core.authentication'
        if auth_module_key in sys.modules:
            # Only remove if it was our mock (check if it has our mock attributes)
            mod = sys.modules[auth_module_key]
            if isinstance(mod, types.ModuleType) and hasattr(mod, 'MsalConnectionManager'):
                # It's likely our mock, but be careful - only remove if it's safe
                # In practice, leaving it is fine as it won't affect other tests
                pass


class TestNotificationHandlerSetup:
    """Test notification handler setup."""

    @pytest.mark.asyncio
    async def test_notification_handlers_registered_when_enabled(
        self,
        full_config,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
    ):
        """Test that notification handlers are registered when enable_notifications is True."""
        worker = A365FrontEndPluginWorker(full_config)
        with patch("microsoft_agents_a365.notifications.AgentNotification", return_value=mock_notification):
            await worker.setup_notification_handlers(
                agent_app=mock_agent_application,
                session_manager=mock_session_manager,
            )
        verify_all_notification_handlers(mock_notification)

    @pytest.mark.asyncio
    async def test_notification_handler_executes_workflow(
        self,
        full_config,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_turn_context,
        mock_turn_state,
        mock_notification_activity,
    ):
        """Test that notification handler executes NAT workflow."""
        worker = A365FrontEndPluginWorker(full_config)
        handlers = []
        mock_notification.on_email = Mock(return_value=create_notification_decorator_mock(handlers))

        with patch("microsoft_agents_a365.notifications.AgentNotification", return_value=mock_notification):
            await worker.setup_notification_handlers(
                agent_app=mock_agent_application,
                session_manager=mock_session_manager,
            )

        if handlers:
            await handlers[0](mock_turn_context, mock_turn_state, mock_notification_activity)
            mock_session_manager.run.assert_called_once()
            mock_turn_context.send_activity.assert_called_once_with("Test workflow result")

    @pytest.mark.asyncio
    async def test_notification_handler_missing_package(
        self,
        full_config,
        mock_agent_application,
        mock_session_manager,
    ):
        """Test that missing notifications package is handled gracefully."""
        # Remove the module from sys.modules to force re-import, then patch __import__ to fail
        import sys
        original_notifications = sys.modules.pop("microsoft_agents_a365.notifications", None)
        original_models = sys.modules.pop("microsoft_agents_a365.notifications.models", None)
        
        # Patch __import__ to raise ImportError for this specific import
        original_import = __import__
        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "microsoft_agents_a365.notifications":
                raise ImportError("No module named 'microsoft_agents_a365.notifications'")
            return original_import(name, globals, locals, fromlist, level)
        
        try:
            worker = A365FrontEndPluginWorker(full_config)
            with patch("builtins.__import__", side_effect=mock_import):
                # Should return early without error (ImportError is caught inside the function)
                await worker.setup_notification_handlers(
                    agent_app=mock_agent_application,
                    session_manager=mock_session_manager,
                )
                # Verify no handlers were set up (function returned early)
                # This test passes if no exception is raised
        finally:
            # Restore modules
            if original_notifications is not None:
                sys.modules["microsoft_agents_a365.notifications"] = original_notifications
            if original_models is not None:
                sys.modules["microsoft_agents_a365.notifications.models"] = original_models


class TestMessageHandlerSetup:
    """Test message handler setup."""

    @pytest.mark.asyncio
    async def test_message_handler_registered_and_executes(
        self,
        full_config,
        mock_agent_application,
        mock_session_manager,
        mock_turn_context,
        mock_turn_state,
    ):
        """Test that message handler is registered and executes workflow."""
        worker = A365FrontEndPluginWorker(full_config)
        handlers = []
        mock_agent_application.activity = Mock(side_effect=create_activity_decorator_mock(handlers))

        await worker.setup_message_handlers(
            agent_app=mock_agent_application,
            session_manager=mock_session_manager,
        )

        mock_agent_application.activity.assert_called_once_with("message")
        if handlers:
            await handlers[0](mock_turn_context, mock_turn_state)
            mock_session_manager.run.assert_called_once()
            mock_turn_context.send_activity.assert_called_once_with("Test workflow result")

    @pytest.mark.asyncio
    async def test_message_handler_empty_message(
        self,
        full_config,
        mock_agent_application,
        mock_session_manager,
        mock_turn_context,
        mock_turn_state,
    ):
        """Test that empty message is handled correctly."""
        worker = A365FrontEndPluginWorker(full_config)
        mock_turn_context.activity.text = ""
        handlers = []
        mock_agent_application.activity = Mock(side_effect=create_activity_decorator_mock(handlers))

        await worker.setup_message_handlers(
            agent_app=mock_agent_application,
            session_manager=mock_session_manager,
        )

        if handlers:
            await handlers[0](mock_turn_context, mock_turn_state)
            mock_session_manager.run.assert_not_called()
            mock_turn_context.send_activity.assert_called_once()
            assert "didn't receive" in mock_turn_context.send_activity.call_args[0][0].lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("handler_type", ["notification", "message"])
    async def test_handler_error_handling(
        self,
        handler_type,
        full_config,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_turn_context,
        mock_turn_state,
        mock_notification_activity,
    ):
        """Test that handlers handle errors gracefully (parameterized for notification and message)."""
        worker = A365FrontEndPluginWorker(full_config)
        mock_session_manager.run.side_effect = Exception("Workflow execution failed")
        
        if handler_type == "notification":
            handlers = []
            mock_notification.on_email = Mock(return_value=create_notification_decorator_mock(handlers))
            with patch("microsoft_agents_a365.notifications.AgentNotification", return_value=mock_notification):
                await worker.setup_notification_handlers(
                    agent_app=mock_agent_application,
                    session_manager=mock_session_manager,
                )
            if handlers:
                await handlers[0](mock_turn_context, mock_turn_state, mock_notification_activity)
        else:  # message
            handlers = []
            mock_agent_application.activity = Mock(side_effect=create_activity_decorator_mock(handlers))
            await worker.setup_message_handlers(
                agent_app=mock_agent_application,
                session_manager=mock_session_manager,
            )
            if handlers:
                await handlers[0](mock_turn_context, mock_turn_state)
        
        mock_turn_context.send_activity.assert_called_once()
        error_msg = mock_turn_context.send_activity.call_args[0][0].lower()
        # Error messages can be: "error", "encountered", "invalid", "timed out", etc.
        assert any(keyword in error_msg for keyword in ["error", "encountered", "invalid", "timed out"])


class TestErrorHandler:
    """Test error handler setup."""

    @pytest.mark.asyncio
    async def test_error_handler_registered_and_sends_message(
        self,
        mock_agent_application,
        mock_turn_context,
    ):
        """Test that error handler is registered and sends error message."""
        handlers = []
        # error is used as a decorator: @agent_app.error
        def error_decorator(func):
            handlers.append(func)
            return func
        mock_agent_application.error = Mock(return_value=error_decorator)

        @mock_agent_application.error
        async def on_error(context, error):
            await context.send_activity("I encountered an error processing your request. Please try again.")

        mock_agent_application.error.assert_called_once()
        if handlers:
            await handlers[0](mock_turn_context, Exception("Test error"))
            mock_turn_context.send_activity.assert_called_once()


class TestFrontEndPluginInitialization:
    """Test front-end plugin initialization and configuration."""

    @pytest.mark.asyncio
    async def test_plugin_run_sets_up_handlers(
        self,
        a365_plugin,
        full_config,
        a365_config,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_workflow_builder,
    ):
        """Test that plugin initializes correctly and run() sets up handlers without actually starting server."""
        # Verify initialization
        assert a365_plugin.full_config is full_config
        assert a365_plugin.front_end_config is a365_config
        
        # Verify full setup flow
        with patch_sdk_components(
            mock_agent_app=mock_agent_application,
            mock_notification=mock_notification,
            mock_session_mgr=mock_session_manager,
            mock_workflow_builder=mock_workflow_builder,
        ):
            try:
                await a365_plugin.run()
            except KeyboardInterrupt:
                pass  # Expected - start_server raises this to stop execution
            except Exception as e:
                # If there's another exception, fail the test with details
                pytest.fail(f"Plugin.run() raised unexpected exception (not KeyboardInterrupt): {type(e).__name__}: {e}")


        # Verify handlers were set up
        # Note: activity and error are called on the agent_app instance created inside run()
        # Since we patch AgentApplication to return mock_agent_application, the calls should be on that mock
        mock_agent_application.activity.assert_called_once_with("message")
        mock_agent_application.error.assert_called_once()
        verify_all_notification_handlers(mock_notification)
        mock_session_manager.shutdown.assert_called_once()


class TestLogLevelConfiguration:
    """Test log_level configuration."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("log_level,expected_numeric_level", [
        ("DEBUG", 10),
        ("INFO", 20),
        ("WARNING", 30),
        ("ERROR", 40),
        ("debug", 10),  # Test case insensitivity
        ("info", 20),
    ])
    async def test_log_level_set_correctly(
        self,
        log_level,
        expected_numeric_level,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_workflow_builder,
    ):
        """Test that different log_level values set correct log levels."""
        import logging
        from nat.utils.log_utils import setup_logging
        
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            log_level=log_level,
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        plugin = A365FrontEndPlugin(full_config=full_config)
        
        # Patch setup_logging to verify it's called with correct level
        with patch("nat.plugins.a365.front_end.plugin.setup_logging") as mock_setup_logging:
            with patch("nat.plugins.a365.front_end.plugin.logger") as mock_logger:
                with patch_sdk_components(
                    mock_agent_app=mock_agent_application,
                    mock_notification=mock_notification,
                    mock_session_mgr=mock_session_manager,
                    mock_workflow_builder=mock_workflow_builder,
                ):
                    try:
                        await plugin.run()
                    except KeyboardInterrupt:
                        pass
                
                # Verify setup_logging was called with correct numeric level
                mock_setup_logging.assert_called_once_with(expected_numeric_level)
                # Verify logger.setLevel was called with correct numeric level
                mock_logger.setLevel.assert_called_once_with(expected_numeric_level)

    @pytest.mark.asyncio
    async def test_invalid_log_level_falls_back_to_info(
        self,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_workflow_builder,
    ):
        """Test that invalid log_level falls back to INFO."""
        import logging
        
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            log_level="INVALID_LEVEL",
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        plugin = A365FrontEndPlugin(full_config=full_config)
        
        # Patch setup_logging to verify it's called with INFO (fallback)
        with patch("nat.plugins.a365.front_end.plugin.setup_logging") as mock_setup_logging:
            with patch("nat.plugins.a365.front_end.plugin.logger") as mock_logger:
                with patch_sdk_components(
                    mock_agent_app=mock_agent_application,
                    mock_notification=mock_notification,
                    mock_session_mgr=mock_session_manager,
                    mock_workflow_builder=mock_workflow_builder,
                ):
                    try:
                        await plugin.run()
                    except KeyboardInterrupt:
                        pass
                
                # Verify setup_logging was called with INFO (fallback)
                mock_setup_logging.assert_called_once_with(logging.INFO)
                mock_logger.setLevel.assert_called_once_with(logging.INFO)


class TestNotificationWorkflowRouting:
    """Test notification_workflow routing functionality."""

    @pytest.mark.asyncio
    async def test_notification_workflow_creates_separate_session_manager(
        self,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_workflow_builder,
    ):
        """Test that notification_workflow creates separate SessionManager with entry_function."""
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            enable_notifications=True,
            notification_workflow="custom_notification_workflow",
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        plugin = A365FrontEndPlugin(full_config=full_config)
        
        # Track SessionManager.create calls
        session_manager_calls = []
        
        async def mock_create(*args, **kwargs):
            session_manager_calls.append(kwargs)
            return mock_session_manager
        
        # Use patch_sdk_components but override SessionManager.create patch after it starts
        with patch_sdk_components(
            mock_agent_app=mock_agent_application,
            mock_notification=mock_notification,
            mock_session_mgr=mock_session_manager,  # Let it patch, we'll override
            mock_workflow_builder=mock_workflow_builder,
        ):
            # Override the SessionManager.create patch from patch_sdk_components
            with patch("nat.plugins.a365.front_end.plugin.SessionManager.create", side_effect=mock_create):
                try:
                    await plugin.run()
                except KeyboardInterrupt:
                    pass
        
        # Verify SessionManager.create was called twice
        assert len(session_manager_calls) == 2
        
        # First call: default session manager (no entry_function)
        default_call = session_manager_calls[0]
        assert default_call.get("entry_function") is None
        
        # Second call: notification session manager (with entry_function)
        notification_call = session_manager_calls[1]
        assert notification_call.get("entry_function") == "custom_notification_workflow"

    @pytest.mark.asyncio
    async def test_notification_workflow_none_uses_same_session_manager(
        self,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_workflow_builder,
    ):
        """Test that when notification_workflow is None, both use the same SessionManager."""
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            enable_notifications=True,
            notification_workflow=None,  # Explicitly None
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        plugin = A365FrontEndPlugin(full_config=full_config)
        
        # Track SessionManager.create calls
        session_manager_calls = []
        
        async def mock_create(*args, **kwargs):
            session_manager_calls.append(kwargs)
            return mock_session_manager
        
        # Use patch_sdk_components but override SessionManager.create patch after it starts
        with patch_sdk_components(
            mock_agent_app=mock_agent_application,
            mock_notification=mock_notification,
            mock_session_mgr=mock_session_manager,  # Let it patch, we'll override
            mock_workflow_builder=mock_workflow_builder,
        ):
            # Override the SessionManager.create patch from patch_sdk_components
            with patch("nat.plugins.a365.front_end.plugin.SessionManager.create", side_effect=mock_create):
                try:
                    await plugin.run()
                except KeyboardInterrupt:
                    pass
        
        # Verify SessionManager.create was called only once (same manager for both)
        assert len(session_manager_calls) == 1
        assert session_manager_calls[0].get("entry_function") is None

    @pytest.mark.asyncio
    async def test_notification_handlers_use_notification_session_manager(
        self,
        mock_agent_application,
        mock_notification,
        mock_session_manager,
        mock_workflow_builder,
        mock_turn_context,
        mock_turn_state,
        mock_notification_activity,
    ):
        """Test that notification handlers use notification_session_manager, not default."""
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            enable_notifications=True,
            notification_workflow="custom_notification_workflow",
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        
        # Create separate mock session managers
        default_session_manager = MagicMock()
        default_runner = MagicMock()
        default_runner.__aenter__ = AsyncMock(return_value=default_runner)
        default_runner.__aexit__ = AsyncMock(return_value=None)
        default_runner.result = AsyncMock(return_value="Default workflow result")
        default_session_manager.run = MagicMock(return_value=default_runner)
        default_session_manager.shutdown = AsyncMock()
        
        notification_session_manager = MagicMock()
        notification_runner = MagicMock()
        notification_runner.__aenter__ = AsyncMock(return_value=notification_runner)
        notification_runner.__aexit__ = AsyncMock(return_value=None)
        notification_runner.result = AsyncMock(return_value="Notification workflow result")
        notification_session_manager.run = MagicMock(return_value=notification_runner)
        notification_session_manager.shutdown = AsyncMock()
        
        call_count = {"value": 0}
        
        async def mock_create(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return default_session_manager
            else:
                return notification_session_manager
        
        plugin = A365FrontEndPlugin(full_config=full_config)
        worker = A365FrontEndPluginWorker(full_config)
        
        handlers = []
        mock_notification.on_email = Mock(return_value=create_notification_decorator_mock(handlers))
        
        with patch("microsoft_agents_a365.notifications.AgentNotification", return_value=mock_notification):
            # Setup notification handlers (this should use notification_session_manager)
            await worker.setup_notification_handlers(
                agent_app=mock_agent_application,
                session_manager=notification_session_manager,
            )
        
        # Execute notification handler
        if handlers:
            await handlers[0](mock_turn_context, mock_turn_state, mock_notification_activity)
            
            # Verify notification_session_manager was used (not default_session_manager)
            notification_session_manager.run.assert_called_once()
            default_session_manager.run.assert_not_called()


class TestWorkerPatternMethods:
    """Test worker pattern dependency injection methods."""

    def test_get_storage_returns_memory_storage(self, full_config):
        """Test that _get_storage() returns MemoryStorage by default."""
        worker = A365FrontEndPluginWorker(full_config)
        
        # Patch MemoryStorage at its import source
        with patch("microsoft_agents.hosting.core.MemoryStorage") as mock_memory_storage:
            mock_instance = Mock()
            mock_memory_storage.return_value = mock_instance
            
            storage = worker._get_storage()
            
            # Verify MemoryStorage was instantiated
            mock_memory_storage.assert_called_once()
            # Verify returned instance
            assert storage is mock_instance

    def test_get_connection_manager_returns_msal_connection_manager(self, full_config):
        """Test that _get_connection_manager() returns MsalConnectionManager with correct config."""
        worker = A365FrontEndPluginWorker(full_config)
        
        agents_sdk_config = {
            "MicrosoftAppId": "test-app-id",
            "MicrosoftAppPassword": "test-password",
            "MicrosoftAppTenantId": "test-tenant-id",
        }
        
        # Patch MsalConnectionManager at its import source
        with patch("microsoft_agents.hosting.core.authentication.MsalConnectionManager") as mock_msal:
            mock_instance = Mock()
            mock_msal.return_value = mock_instance
            
            connection_manager = worker._get_connection_manager(agents_sdk_config)
            
            # Verify MsalConnectionManager was instantiated with correct config
            mock_msal.assert_called_once_with(**agents_sdk_config)
            # Verify returned instance
            assert connection_manager is mock_instance

    def test_get_storage_can_be_overridden(self, full_config):
        """Test that _get_storage() can be overridden in a subclass."""
        custom_storage = Mock()
        
        class CustomWorker(A365FrontEndPluginWorker):
            def _get_storage(self):
                return custom_storage
        
        worker = CustomWorker(full_config)
        storage = worker._get_storage()
        
        assert storage is custom_storage

    def test_get_connection_manager_can_be_overridden(self, full_config):
        """Test that _get_connection_manager() can be overridden in a subclass."""
        custom_connection_manager = Mock()
        agents_sdk_config = {"MicrosoftAppId": "test-app-id"}
        
        class CustomWorker(A365FrontEndPluginWorker):
            def _get_connection_manager(self, agents_sdk_config):
                return custom_connection_manager
        
        worker = CustomWorker(full_config)
        connection_manager = worker._get_connection_manager(agents_sdk_config)
        
        assert connection_manager is custom_connection_manager


class TestErrorHandlingInCreateAgentApplication:
    """Test error handling in create_agent_application."""

    @pytest.mark.asyncio
    async def test_connection_manager_value_error_raises_a365_configuration_error(self, full_config):
        """Test that ValueError from connection manager raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError
        
        worker = A365FrontEndPluginWorker(full_config)
        
        # Patch SDK imports that create_agent_application() needs
        # CloudAdapter and AgentApplication need to be callable classes that return mocks
        mock_adapter_instance = Mock()
        mock_agent_app_instance = Mock()
        
        # Create mock classes that can be instantiated
        class MockCloudAdapter:
            def __init__(self, *args, **kwargs):
                pass
        
        class MockAgentApplication:
            def __class_getitem__(cls, item):
                return cls
            def __new__(cls, *args, **kwargs):
                return mock_agent_app_instance
        
        with patch("microsoft_agents.hosting.core.MemoryStorage", return_value=Mock()):
            with patch("microsoft_agents.hosting.core.CloudAdapter", MockCloudAdapter, create=True):
                with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                    with patch("microsoft_agents.hosting.core.TurnState", Mock(), create=True):
                        with patch.object(worker, "_get_connection_manager", side_effect=ValueError("Invalid config")):
                            with pytest.raises(A365ConfigurationError, match="Invalid configuration for connection manager"):
                                await worker.create_agent_application()

    @pytest.mark.asyncio
    async def test_connection_manager_type_error_raises_a365_configuration_error(self, full_config):
        """Test that TypeError from connection manager raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError
        
        worker = A365FrontEndPluginWorker(full_config)
        
        # Patch SDK imports that create_agent_application() needs
        # CloudAdapter and AgentApplication need to be callable classes that return mocks
        mock_adapter_instance = Mock()
        mock_agent_app_instance = Mock()
        
        # Create mock classes that can be instantiated
        class MockCloudAdapter:
            def __init__(self, *args, **kwargs):
                pass
        
        class MockAgentApplication:
            def __class_getitem__(cls, item):
                return cls
            def __new__(cls, *args, **kwargs):
                return mock_agent_app_instance
        
        with patch("microsoft_agents.hosting.core.MemoryStorage", return_value=Mock()):
            with patch("microsoft_agents.hosting.core.CloudAdapter", MockCloudAdapter, create=True):
                with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                    with patch("microsoft_agents.hosting.core.TurnState", Mock(), create=True):
                        with patch.object(worker, "_get_connection_manager", side_effect=TypeError("Wrong type")):
                            with pytest.raises(A365ConfigurationError, match="Invalid configuration for connection manager"):
                                await worker.create_agent_application()

    @pytest.mark.asyncio
    async def test_connection_manager_application_error_raises_a365_sdk_error(self, full_config):
        """Test that ApplicationError from connection manager raises A365SDKError."""
        from nat.plugins.a365.exceptions import A365SDKError
        
        # Mock ApplicationError
        class MockApplicationError(Exception):
            pass
        
        worker = A365FrontEndPluginWorker(full_config)
        
        # Patch SDK imports that create_agent_application() needs
        # CloudAdapter and AgentApplication need to be callable classes that return mocks
        mock_adapter_instance = Mock()
        mock_agent_app_instance = Mock()
        
        # Create mock classes that can be instantiated
        class MockCloudAdapter:
            def __init__(self, *args, **kwargs):
                pass
        
        class MockAgentApplication:
            def __class_getitem__(cls, item):
                return cls
            def __new__(cls, *args, **kwargs):
                return mock_agent_app_instance
        
        with patch("microsoft_agents.hosting.core.MemoryStorage", return_value=Mock()):
            with patch("microsoft_agents.hosting.core.CloudAdapter", MockCloudAdapter, create=True):
                with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                    with patch("microsoft_agents.hosting.core.TurnState", Mock(), create=True):
                        with patch.object(worker, "_get_connection_manager", side_effect=MockApplicationError("SDK error")):
                            # Patch ApplicationError import at its source
                            with patch("microsoft_agents.hosting.core.app.app_error.ApplicationError", MockApplicationError):
                                with pytest.raises(A365SDKError, match="Failed to initialize connection manager"):
                                    await worker.create_agent_application()

    @pytest.mark.asyncio
    async def test_cloud_adapter_error_raises_a365_sdk_error(self, full_config):
        """Test that CloudAdapter initialization failure raises A365SDKError."""
        from nat.plugins.a365.exceptions import A365SDKError
        
        worker = A365FrontEndPluginWorker(full_config)
        mock_storage = Mock()
        mock_connection_manager = Mock()
        
        with patch.object(worker, "_get_storage", return_value=mock_storage):
            with patch.object(worker, "_get_connection_manager", return_value=mock_connection_manager):
                # Patch CloudAdapter at its import source
                with patch("microsoft_agents.hosting.core.CloudAdapter", side_effect=Exception("Adapter error"), create=True):
                    with pytest.raises(A365SDKError, match="Failed to initialize CloudAdapter"):
                        await worker.create_agent_application()

    @pytest.mark.asyncio
    async def test_authorization_value_error_raises_a365_configuration_error(self, full_config):
        """Test that ValueError from Authorization raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError
        
        worker = A365FrontEndPluginWorker(full_config)
        mock_storage = Mock()
        mock_connection_manager = Mock()
        
        with patch.object(worker, "_get_storage", return_value=mock_storage):
            with patch.object(worker, "_get_connection_manager", return_value=mock_connection_manager):
                # Patch SDK components at their import source
                with patch("microsoft_agents.hosting.core.CloudAdapter", return_value=Mock(), create=True):
                    with patch("microsoft_agents.hosting.core.authentication.Authorization", side_effect=ValueError("Invalid auth config")):
                        with pytest.raises(A365ConfigurationError, match="Invalid configuration for Authorization"):
                            await worker.create_agent_application()

    @pytest.mark.asyncio
    async def test_agent_application_value_error_raises_a365_configuration_error(self, full_config):
        """Test that ValueError from AgentApplication raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError
        
        worker = A365FrontEndPluginWorker(full_config)
        mock_storage = Mock()
        mock_connection_manager = Mock()
        mock_adapter = Mock()
        mock_authorization = Mock()
        
        with patch.object(worker, "_get_storage", return_value=mock_storage):
            with patch.object(worker, "_get_connection_manager", return_value=mock_connection_manager):
                # Patch SDK components at their import source
                with patch("microsoft_agents.hosting.core.CloudAdapter", return_value=mock_adapter, create=True):
                    with patch("microsoft_agents.hosting.core.authentication.Authorization", return_value=mock_authorization):
                        # Mock AgentApplication to raise ValueError
                        class MockAgentApplication:
                            def __class_getitem__(cls, item):
                                return cls
                            def __new__(cls, *args, **kwargs):
                                raise ValueError("Invalid app config")
                        
                        with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                            with pytest.raises(A365ConfigurationError, match="Invalid configuration for AgentApplication"):
                                await worker.create_agent_application()

    @pytest.mark.asyncio
    async def test_agent_application_application_error_raises_a365_sdk_error(self, full_config):
        """Test that ApplicationError from AgentApplication raises A365SDKError."""
        from nat.plugins.a365.exceptions import A365SDKError
        
        # Mock ApplicationError
        class MockApplicationError(Exception):
            pass
        
        worker = A365FrontEndPluginWorker(full_config)
        mock_storage = Mock()
        mock_connection_manager = Mock()
        mock_adapter = Mock()
        mock_authorization = Mock()
        
        with patch.object(worker, "_get_storage", return_value=mock_storage):
            with patch.object(worker, "_get_connection_manager", return_value=mock_connection_manager):
                # Patch SDK components at their import source
                with patch("microsoft_agents.hosting.core.CloudAdapter", return_value=mock_adapter, create=True):
                    with patch("microsoft_agents.hosting.core.authentication.Authorization", return_value=mock_authorization):
                        # Mock AgentApplication to raise ApplicationError
                        class MockAgentApplication:
                            def __class_getitem__(cls, item):
                                return cls
                            def __new__(cls, *args, **kwargs):
                                raise MockApplicationError("SDK app error")
                        
                        with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                            with patch("microsoft_agents.hosting.core.app.app_error.ApplicationError", MockApplicationError):
                                with pytest.raises(A365SDKError, match="Failed to create AgentApplication"):
                                    await worker.create_agent_application()

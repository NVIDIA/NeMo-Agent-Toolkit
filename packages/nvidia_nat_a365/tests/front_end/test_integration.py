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

import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.a365.exceptions import A365AuthenticationError
from nat.plugins.a365.exceptions import A365SDKError
from nat.plugins.a365.exceptions import A365WorkflowExecutionError
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.plugins.a365.front_end.plugin import A365FrontEndPlugin
from nat.plugins.a365.front_end.worker import A365FrontEndPluginWorker
from nat.test.functions import EchoFunctionConfig


# Helper functions and fixtures
@pytest.fixture(name="mock_agent_application")
def mock_agent_application_fixture():
    """Create a mock AgentApplication.

    The mock needs to support:
    - ``@agent_app.activity("message")`` — activity is a decorator factory:
      called with ``"message"``, the result is then used as a decorator over the handler.
    - ``@agent_app.error`` — error is used directly as a decorator (no call):
      it must behave as ``error(func) -> func`` so the handler ends up bound.
    """
    app = MagicMock()

    # activity is used as a decorator factory: @agent_app.activity("message")
    # It's called with "message", then the result is used as a decorator
    def activity_decorator_factory(activity_type):

        def decorator(func):
            return func

        return decorator

    app.activity = MagicMock(side_effect=activity_decorator_factory)

    # error is used as a *direct* decorator: ``@agent_app.error`` (no parentheses).
    # The SDK's contract is ``error(func) -> func``, so the attribute itself must be the
    # callable that takes the handler function. Use ``side_effect`` (not ``return_value``)
    # so that ``@agent_app.error`` actually invokes ``error_decorator(on_error)`` and the
    # caller ends up with the real handler bound — not a fresh MagicMock.
    def error_decorator(func):
        return func

    app.error = MagicMock(side_effect=error_decorator)

    return app


@pytest.fixture(name="mock_notification")
def mock_notification_fixture():
    """Create a mock AgentNotification."""
    notification = Mock()
    for method in ["on_email", "on_word", "on_excel", "on_powerpoint", "on_user_created", "on_user_deleted"]:
        setattr(notification, method, Mock())
    return notification


@pytest.fixture(name="mock_turn_context")
def mock_turn_context_fixture():
    """Create a mock TurnContext."""
    context = Mock()
    context.activity = Mock()
    context.activity.text = "Test message"
    context.send_activity = AsyncMock()
    return context


@pytest.fixture(name="mock_turn_state")
def mock_turn_state_fixture():
    """Create a mock TurnState."""
    return Mock()


@pytest.fixture(name="mock_notification_activity")
def mock_notification_activity_fixture():
    """Create a mock AgentNotificationActivity."""
    activity = Mock()
    activity.text = "Test notification text"
    activity.summary = "Test notification summary"
    # Set up email structure for email notifications
    activity.email = Mock()
    activity.email.html_body = "Test email HTML body"
    activity.wpx_comment = None  # Not a comment notification
    return activity


@pytest.fixture(name="mock_session_manager")
def mock_session_manager_fixture():
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


@pytest.fixture(name="mock_workflow_builder")
def mock_workflow_builder_fixture():
    """Create a mock WorkflowBuilder."""
    builder = Mock()
    builder.__aenter__ = AsyncMock(return_value=builder)
    builder.__aexit__ = AsyncMock(return_value=None)
    return builder


@pytest.fixture(name="a365_config")
def a365_config_fixture():
    """Create A365FrontEndConfig for testing."""
    return A365FrontEndConfig(
        app_id="test-app-id",
        app_password="test-app-password",
        host="localhost",
        port=3978,
        enable_notifications=True,
    )


@pytest.fixture(name="full_config")
def full_config_fixture(a365_config):
    """Create full Config for testing."""
    return Config(
        general=GeneralConfig(front_end=a365_config),
        workflow=EchoFunctionConfig(),
    )


@pytest.fixture(name="a365_plugin")
def a365_plugin_fixture(full_config):
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
def patch_sdk_components(mock_agent_app=None,
                         mock_notification=None,
                         mock_session_mgr=None,
                         mock_workflow_builder=None):
    """Context manager to patch all Microsoft Agents SDK components."""
    patches = []
    try:
        if mock_workflow_builder:
            mock_wb_cm = AsyncMock()
            mock_wb_cm.__aenter__ = AsyncMock(return_value=mock_workflow_builder)
            mock_wb_cm.__aexit__ = AsyncMock(return_value=None)
            patches.append(
                patch("nat.plugins.a365.front_end.plugin.WorkflowBuilder.from_config", return_value=mock_wb_cm))

        if mock_session_mgr:
            patches.append(
                patch("nat.plugins.a365.front_end.plugin.SessionManager.create", return_value=mock_session_mgr))

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
        mock_storage = Mock()
        mock_adapter = Mock()
        mock_conn_mgr = Mock()
        mock_conn_mgr.get_default_connection_configuration = Mock(return_value=Mock())
        mock_auth = Mock()

        patches.extend([
            patch("microsoft_agents.hosting.core.MemoryStorage", return_value=mock_storage, create=True),
            patch("microsoft_agents.hosting.aiohttp.CloudAdapter", return_value=mock_adapter, create=True),
            patch("microsoft_agents.authentication.msal.MsalConnectionManager", return_value=mock_conn_mgr),
            patch("microsoft_agents.hosting.core.app.oauth.Authorization", return_value=mock_auth),
        ])

        # Optional patches: targets whose underlying module is NOT a hard dependency of
        # ``nvidia-nat-a365`` and may legitimately be missing in the test environment.
        # Currently this is only ``microsoft_agents_a365.notifications`` (mirrors the
        # production ``except ModuleNotFoundError`` in ``worker.setup_notification_handlers``).
        # Every other patch targets a hard dep — if it fails to start, that's a real
        # breakage (typo, SDK rename, etc.) and should surface, not be swallowed.
        optional_patches: set[int] = set()

        if mock_notification:
            optional_patch = patch(
                "microsoft_agents_a365.notifications.AgentNotification",
                return_value=mock_notification,
            )
            optional_patches.add(id(optional_patch))
            patches.append(optional_patch)

        async def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        patches.append(
            patch(
                "nat.plugins.a365.front_end.plugin._start_aiohttp_site",
                side_effect=raise_keyboard_interrupt,
            ))

        started_patches = []
        for p in patches:
            try:
                p.start()
            except ModuleNotFoundError:
                # Tolerated only for patches we marked optional above. For required
                # SDK targets, a missing module is a real configuration error.
                if id(p) not in optional_patches:
                    raise
                continue
            # NOTE: A missing *attribute* on an existing module raises AttributeError
            # and is deliberately NOT caught here — that's how we detect SDK renames
            # (e.g., a future MemoryStorage move) instead of running tests against the
            # un-mocked real implementation.
            started_patches.append(p)

        # Update patches list to only include successfully started patches
        patches[:] = started_patches

        yield
    finally:
        # Stop all patches
        for p in patches:
            p.stop()


class TestNotificationHandlerSetup:
    """Test notification handler setup."""

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

        assert len(handlers) == 1, (
            f"Expected exactly one email notification handler registered by "
            f"setup_notification_handlers, got {len(handlers)}.")
        await handlers[0](mock_turn_context, mock_turn_state, mock_notification_activity)
        mock_session_manager.run.assert_called_once()
        mock_turn_context.send_activity.assert_called_once_with("Test workflow result")

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

        # Patch __import__ to raise ModuleNotFoundError for this specific import.
        # ModuleNotFoundError is what Python actually raises when a package is missing,
        # and is what setup_notification_handlers narrowly catches for graceful degradation.
        original_import = __import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "microsoft_agents_a365.notifications":
                raise ModuleNotFoundError("No module named 'microsoft_agents_a365.notifications'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            worker = A365FrontEndPluginWorker(full_config)
            with patch("builtins.__import__", side_effect=mock_import):
                # Should return early without error (ModuleNotFoundError is caught inside the function)
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
        assert len(handlers) == 1, (f"Expected exactly one message handler registered by "
                                    f"setup_message_handlers, got {len(handlers)}.")
        await handlers[0](mock_turn_context, mock_turn_state)
        mock_session_manager.run.assert_called_once()
        mock_turn_context.send_activity.assert_called_once_with("Test workflow result")

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

        assert len(handlers) == 1, (f"Expected exactly one message handler registered by "
                                    f"setup_message_handlers, got {len(handlers)}.")
        await handlers[0](mock_turn_context, mock_turn_state)
        mock_session_manager.run.assert_not_called()
        mock_turn_context.send_activity.assert_called_once()
        assert "didn't receive" in mock_turn_context.send_activity.call_args[0][0].lower()

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
            assert len(handlers) == 1, (f"Expected exactly one email notification handler, got {len(handlers)}.")
            await handlers[0](mock_turn_context, mock_turn_state, mock_notification_activity)
        else:  # message
            handlers = []
            mock_agent_application.activity = Mock(side_effect=create_activity_decorator_mock(handlers))
            await worker.setup_message_handlers(
                agent_app=mock_agent_application,
                session_manager=mock_session_manager,
            )
            assert len(handlers) == 1, (f"Expected exactly one message handler, got {len(handlers)}.")
            await handlers[0](mock_turn_context, mock_turn_state)

        mock_turn_context.send_activity.assert_called_once()
        error_msg = mock_turn_context.send_activity.call_args[0][0].lower()
        # Error messages can be: "error", "encountered", "invalid", "timed out", etc.
        assert any(keyword in error_msg for keyword in ["error", "encountered", "invalid", "timed out"])


class TestConnectionManagerConfiguration:
    """Test Microsoft Agents connection manager setup."""

    def test_connection_manager_includes_allowed_audience_aliases(self):
        """Test that extra inbound JWT audiences are exposed as SDK connection aliases."""
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            tenant_id="test-tenant-id",
            allowed_audiences=["alternate-aud", "test-app-id"],
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        worker = A365FrontEndPluginWorker(full_config)

        mock_service_connection = Mock()
        mock_service_connection.CLIENT_ID = "test-app-id"
        mock_service_connection.AUTH_TYPE = "client_secret"

        with patch("microsoft_agents.authentication.msal.MsalConnectionManager") as mock_manager:
            worker._get_connection_manager(mock_service_connection)

        connections = mock_manager.call_args.kwargs["connections_configurations"]
        assert "SERVICE_CONNECTION" in connections
        assert connections["SERVICE_CONNECTION"] is mock_service_connection
        assert "AUDIENCE_ALIAS_1" in connections
        assert connections["AUDIENCE_ALIAS_1"].CLIENT_ID == "alternate-aud"
        assert "AUDIENCE_ALIAS_2" not in connections

    def test_connection_manager_dedups_aliases_case_insensitively(self):
        """Aliases that case-insensitively duplicate each other or the bot's app_id are dropped.

        Regression guard against silently inflating the SDK connection map (and the
        bot secret's in-memory exposure) when YAML configs contain accidental
        duplicates with mixed casing.
        """
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            tenant_id="test-tenant-id",
            allowed_audiences=[
                "alternate-aud",  # unique
                "Alternate-Aud",  # case-insensitive dup of #1 -> dropped
                "alternate-aud",  # exact dup of #1 -> dropped
                "TEST-APP-ID",  # case-insensitive dup of service CLIENT_ID -> dropped
                "another-aud",  # unique
            ],
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        worker = A365FrontEndPluginWorker(full_config)

        mock_service_connection = Mock()
        mock_service_connection.CLIENT_ID = "test-app-id"
        mock_service_connection.AUTH_TYPE = "client_secret"

        with patch("microsoft_agents.authentication.msal.MsalConnectionManager") as mock_manager:
            worker._get_connection_manager(mock_service_connection)

        connections = mock_manager.call_args.kwargs["connections_configurations"]
        # Expect exactly: SERVICE_CONNECTION, AUDIENCE_ALIAS_1 (alternate-aud),
        # AUDIENCE_ALIAS_2 (another-aud). No gaps in numbering despite the dups.
        assert set(connections.keys()) == {"SERVICE_CONNECTION", "AUDIENCE_ALIAS_1", "AUDIENCE_ALIAS_2"}
        assert connections["AUDIENCE_ALIAS_1"].CLIENT_ID == "alternate-aud"
        assert connections["AUDIENCE_ALIAS_2"].CLIENT_ID == "another-aud"

    def test_connection_manager_logs_accepted_audiences(self, caplog):
        """``_get_connection_manager`` should emit a single INFO line naming each accepted audience.

        Operators debugging 401s on Teams/Bot Framework need a way to confirm
        which audiences are actually installed. The startup log is that surface.
        """
        a365_config = A365FrontEndConfig(
            app_id="test-app-id",
            app_password="test-app-password",
            tenant_id="test-tenant-id",
            allowed_audiences=["alternate-aud"],
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        worker = A365FrontEndPluginWorker(full_config)

        mock_service_connection = Mock()
        mock_service_connection.CLIENT_ID = "test-app-id"
        mock_service_connection.AUTH_TYPE = "client_secret"

        with patch("microsoft_agents.authentication.msal.MsalConnectionManager"):
            with caplog.at_level(logging.INFO, logger="nat.plugins.a365.front_end.worker"):
                worker._get_connection_manager(mock_service_connection)

        # Single line should name both audiences with their connection names.
        matching = [r for r in caplog.records if "accepting JWT audiences" in r.getMessage()]
        assert len(matching) == 1, f"Expected exactly one accepted-audiences log, got {len(matching)}"
        message = matching[0].getMessage()
        assert "SERVICE_CONNECTION=test-app-id" in message
        assert "AUDIENCE_ALIAS_1=alternate-aud" in message

    def test_jwt_patch_accepts_alias_audience_end_to_end(self):
        """End-to-end: a JWT-validator built from the production connection manager accepts alias audiences.

        This test pins the SDK behavior the audience-alias feature actually relies on:
        ``MsalConnectionManager.__init__`` cross-populates ``AgentAuthConfiguration._connections``
        across every entry in ``connections_configurations`` (the "# JWT-patch" loop in 0.8.0),
        and ``JwtTokenValidator`` calls ``_jwt_patch_is_valid_aud`` to accept any matching audience.

        Both ``_connections`` and ``_jwt_patch_is_valid_aud`` are SDK private members. If
        Microsoft renames, removes, or behaviorally changes either in a patch version, this
        test fails and we catch the regression before production sees 401s on inbound Teams
        tokens with non-``app_id`` audiences.

        The test does NOT verify cryptographic JWT signing -- only audience-set membership,
        which is the part of validation our alias mechanism contributes to. Signature
        verification is exercised by the SDK's own test suite.
        """
        # Real SDK imports, not mocks. If these fail to resolve, the SDK contract has changed.
        from microsoft_agents.authentication.msal import MsalConnectionManager
        from microsoft_agents.hosting.core import AgentAuthConfiguration
        from microsoft_agents.hosting.core.authorization.auth_types import AuthTypes

        a365_config = A365FrontEndConfig(
            app_id="11111111-1111-1111-1111-111111111111",
            app_password="dummy-secret-not-used-by-validator",
            tenant_id="22222222-2222-2222-2222-222222222222",
            allowed_audiences=["33333333-3333-3333-3333-333333333333"],
        )
        full_config = Config(
            general=GeneralConfig(front_end=a365_config),
            workflow=EchoFunctionConfig(),
        )
        worker = A365FrontEndPluginWorker(full_config)

        # Build the real SERVICE_CONNECTION exactly as production does.
        service_connection = AgentAuthConfiguration(
            client_id=a365_config.app_id,
            client_secret="dummy-secret-not-used-by-validator",
            auth_type=AuthTypes.client_secret,
            connection_name="SERVICE_CONNECTION",
            tenant_id=a365_config.tenant_id,
        )

        # Drive through the real connection manager so its "# JWT-patch" cross-population
        # runs and mutates ``service_connection._connections`` to include the alias.
        connections_dict = worker._build_connection_configurations(service_connection)
        MsalConnectionManager(connections_configurations=connections_dict)

        # After MsalConnectionManager.__init__, the SDK private _jwt_patch_is_valid_aud
        # method on the SERVICE_CONNECTION config should accept both app_id and the alias.
        # If this attribute disappears, the SDK has changed in an incompatible way.
        assert hasattr(service_connection, "_jwt_patch_is_valid_aud"), (
            "SDK regression: AgentAuthConfiguration._jwt_patch_is_valid_aud is gone. "
            "The audience-alias feature relies on this private method; an alternative "
            "validation API must be wired up before merging.")

        # Primary CLIENT_ID accepted.
        assert service_connection._jwt_patch_is_valid_aud(a365_config.app_id), (
            "SDK regression: validator rejects the bot's own app_id. Something is "
            "very wrong with the connection manager wiring.")

        # Alias CLIENT_ID accepted -- this is the actual feature under test.
        assert service_connection._jwt_patch_is_valid_aud("33333333-3333-3333-3333-333333333333"), (
            "Audience-alias regression: _jwt_patch_is_valid_aud rejected an alias audience "
            "that was passed to MsalConnectionManager. Either the SDK's # JWT-patch loop "
            "no longer cross-populates ``_connections``, or NAT's audience-alias mechanism "
            "in worker.py has stopped routing aliases through ``connections_configurations``.")

        # Sanity: a truly-unknown audience is rejected.
        assert not service_connection._jwt_patch_is_valid_aud("99999999-9999-9999-9999-999999999999"), (
            "Validator accepted an audience that was never configured -- the alias mechanism "
            "appears to be accepting everything, which is a security regression.")


class TestErrorHandler:
    """Test error handler setup.

    These tests exercise the *production* ``A365FrontEndPluginWorker.setup_error_handlers``
    rather than reimplementing a stub handler locally. We rely on the
    ``mock_agent_application`` fixture, whose ``app.error`` is a ``MagicMock`` with
    ``side_effect=error_decorator``, so ``@agent_app.error`` passes the production
    handler straight through to us via ``call_args``.
    """

    @staticmethod
    def _register_and_capture_handler(full_config, mock_agent_application):
        """Run ``setup_error_handlers`` on a real worker and return the registered handler."""
        worker = A365FrontEndPluginWorker(full_config)
        worker.setup_error_handlers(mock_agent_application)
        mock_agent_application.error.assert_called_once()
        registered = mock_agent_application.error.call_args[0][0]
        assert callable(registered), "Expected setup_error_handlers to register a callable"
        return registered

    async def test_error_handler_is_registered_on_agent_app(
        self,
        full_config,
        mock_agent_application,
    ):
        """``setup_error_handlers`` must register exactly one handler on ``agent_app.error``."""
        handler = self._register_and_capture_handler(full_config, mock_agent_application)
        # Production registers an async function named ``on_error``; sanity-check the shape.
        import inspect
        assert inspect.iscoroutinefunction(handler)

    @pytest.mark.parametrize(
        "error, expected_substring",
        [
            # Custom exception types — explicit isinstance branches in production code.
            pytest.param(
                A365AuthenticationError("token rejected"),
                "authentication failed",
                id="auth-error",
            ),
            pytest.param(
                A365SDKError("address already in use", sdk_component="aiohttp_server"),
                "server configuration error",
                id="sdk-error-address-in-message",
            ),
            pytest.param(
                A365SDKError("unrelated SDK failure"),
                "a system error occurred",
                id="sdk-error-generic",
            ),
            pytest.param(
                A365WorkflowExecutionError("workflow crashed"),
                "encountered an error",
                id="workflow-execution-error",
            ),
            # Generic exceptions — string-match fallback branches.
            pytest.param(
                Exception("Connection timeout while contacting upstream"),
                "timed out",
                id="generic-timeout",
            ),
            pytest.param(
                Exception("Unauthorized access to resource"),
                "authentication failed",
                id="generic-unauthorized",
            ),
            pytest.param(
                Exception("Network connection refused"),
                "connection error",
                id="generic-connection",
            ),
            pytest.param(
                Exception("Something completely unexpected"),
                "encountered an error",
                id="generic-fallback",
            ),
        ],
    )
    async def test_error_handler_classifies_and_messages_user(
        self,
        full_config,
        mock_agent_application,
        mock_turn_context,
        error,
        expected_substring,
    ):
        """The registered handler must classify exceptions and send the right user-facing message."""
        handler = self._register_and_capture_handler(full_config, mock_agent_application)

        await handler(mock_turn_context, error)

        mock_turn_context.send_activity.assert_called_once()
        sent = mock_turn_context.send_activity.call_args[0][0]
        assert expected_substring in sent.lower(), (
            f"Error classification produced unexpected user message.\n"
            f"  expected substring: {expected_substring!r}\n"
            f"  actual message:     {sent!r}")


class TestFrontEndPluginInitialization:
    """Test front-end plugin initialization and configuration."""

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
                pytest.fail(
                    f"Plugin.run() raised unexpected exception (not KeyboardInterrupt): {type(e).__name__}: {e}")

        # Verify handlers were set up
        # Note: activity and error are called on the agent_app instance created inside run()
        # Since we patch AgentApplication to return mock_agent_application, the calls should be on that mock
        mock_agent_application.activity.assert_called_once_with("message")
        mock_agent_application.error.assert_called_once()
        verify_all_notification_handlers(mock_notification)
        mock_session_manager.shutdown.assert_called_once()


class TestLogLevelConfiguration:
    """Test log_level configuration."""

    @pytest.mark.parametrize(
        "log_level,expected_numeric_level",
        [
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

        A365FrontEndPlugin(full_config=full_config)
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
        assert len(handlers) == 1, (
            f"Expected exactly one email notification handler registered by "
            f"setup_notification_handlers, got {len(handlers)}.")
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
        from microsoft_agents.hosting.core import AgentAuthConfiguration
        from microsoft_agents.hosting.core.authorization.auth_types import AuthTypes

        worker = A365FrontEndPluginWorker(full_config)

        service_connection = AgentAuthConfiguration(
            client_id="test-app-id",
            client_secret="test-password",
            tenant_id="test-tenant-id",
            auth_type=AuthTypes.client_secret,
            connection_name="SERVICE_CONNECTION",
        )

        with patch("microsoft_agents.authentication.msal.MsalConnectionManager") as mock_msal:
            mock_instance = Mock()
            mock_msal.return_value = mock_instance

            connection_manager = worker._get_connection_manager(service_connection)

            mock_msal.assert_called_once_with(connections_configurations={"SERVICE_CONNECTION": service_connection})
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
        service_connection = Mock()

        class CustomWorker(A365FrontEndPluginWorker):

            def _get_connection_manager(self, service_connection):
                return custom_connection_manager

        worker = CustomWorker(full_config)
        connection_manager = worker._get_connection_manager(service_connection)

        assert connection_manager is custom_connection_manager


class TestErrorHandlingInCreateAgentApplication:
    """Test error handling in create_agent_application."""

    async def test_connection_manager_value_error_raises_a365_configuration_error(self, full_config):
        """Test that ValueError from connection manager raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError

        worker = A365FrontEndPluginWorker(full_config)

        # Patch SDK imports that create_agent_application() needs
        # CloudAdapter and AgentApplication need to be callable classes that return mocks
        Mock()
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
            with patch("microsoft_agents.hosting.aiohttp.CloudAdapter", MockCloudAdapter, create=True):
                with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                    with patch("microsoft_agents.hosting.core.TurnState", Mock(), create=True):
                        with patch.object(worker, "_get_connection_manager", side_effect=ValueError("Invalid config")):
                            with pytest.raises(A365ConfigurationError,
                                               match="Invalid configuration for connection manager"):
                                await worker.create_agent_application()

    async def test_connection_manager_type_error_raises_a365_configuration_error(self, full_config):
        """Test that TypeError from connection manager raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError

        worker = A365FrontEndPluginWorker(full_config)

        # Patch SDK imports that create_agent_application() needs
        # CloudAdapter and AgentApplication need to be callable classes that return mocks
        Mock()
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
            with patch("microsoft_agents.hosting.aiohttp.CloudAdapter", MockCloudAdapter, create=True):
                with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                    with patch("microsoft_agents.hosting.core.TurnState", Mock(), create=True):
                        with patch.object(worker, "_get_connection_manager", side_effect=TypeError("Wrong type")):
                            with pytest.raises(A365ConfigurationError,
                                               match="Invalid configuration for connection manager"):
                                await worker.create_agent_application()

    async def test_connection_manager_application_error_raises_a365_sdk_error(self, full_config):
        """Test that ApplicationError from connection manager raises A365SDKError."""
        from nat.plugins.a365.exceptions import A365SDKError

        # Mock ApplicationError
        class MockApplicationError(Exception):
            pass

        worker = A365FrontEndPluginWorker(full_config)

        # Patch SDK imports that create_agent_application() needs
        # CloudAdapter and AgentApplication need to be callable classes that return mocks
        Mock()
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
            with patch("microsoft_agents.hosting.aiohttp.CloudAdapter", MockCloudAdapter, create=True):
                with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                    with patch("microsoft_agents.hosting.core.TurnState", Mock(), create=True):
                        with patch.object(worker,
                                          "_get_connection_manager",
                                          side_effect=MockApplicationError("SDK error")):
                            # Patch ApplicationError import at its source
                            with patch("microsoft_agents.hosting.core.app.app_error.ApplicationError",
                                       MockApplicationError):
                                with pytest.raises(A365SDKError, match="Failed to initialize connection manager"):
                                    await worker.create_agent_application()

    async def test_cloud_adapter_error_raises_a365_sdk_error(self, full_config):
        """Test that CloudAdapter initialization failure raises A365SDKError."""
        from nat.plugins.a365.exceptions import A365SDKError

        worker = A365FrontEndPluginWorker(full_config)
        mock_storage = Mock()
        mock_connection_manager = Mock()

        with patch.object(worker, "_get_storage", return_value=mock_storage):
            with patch.object(worker, "_get_connection_manager", return_value=mock_connection_manager):
                # Patch CloudAdapter at its import source
                with patch("microsoft_agents.hosting.aiohttp.CloudAdapter",
                           side_effect=Exception("Adapter error"),
                           create=True):
                    with pytest.raises(A365SDKError, match="Failed to initialize CloudAdapter"):
                        await worker.create_agent_application()

    async def test_authorization_value_error_raises_a365_configuration_error(self, full_config):
        """Test that ValueError from Authorization raises A365ConfigurationError."""
        from nat.plugins.a365.exceptions import A365ConfigurationError

        worker = A365FrontEndPluginWorker(full_config)
        mock_storage = Mock()
        mock_connection_manager = Mock()

        with patch.object(worker, "_get_storage", return_value=mock_storage):
            with patch.object(worker, "_get_connection_manager", return_value=mock_connection_manager):
                # Patch SDK components at their import source
                with patch("microsoft_agents.hosting.aiohttp.CloudAdapter", return_value=Mock(), create=True):
                    with patch("microsoft_agents.hosting.core.app.oauth.Authorization",
                               side_effect=ValueError("Invalid auth config")):
                        with pytest.raises(A365ConfigurationError, match="Invalid configuration for Authorization"):
                            await worker.create_agent_application()

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
                with patch("microsoft_agents.hosting.aiohttp.CloudAdapter", return_value=mock_adapter, create=True):
                    with patch("microsoft_agents.hosting.core.app.oauth.Authorization",
                               return_value=mock_authorization):
                        # Mock AgentApplication to raise ValueError
                        class MockAgentApplication:

                            def __class_getitem__(cls, item):
                                return cls

                            def __new__(cls, *args, **kwargs):
                                raise ValueError("Invalid app config")

                        with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                            with pytest.raises(A365ConfigurationError,
                                               match="Invalid configuration for AgentApplication"):
                                await worker.create_agent_application()

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
                with patch("microsoft_agents.hosting.aiohttp.CloudAdapter", return_value=mock_adapter, create=True):
                    with patch("microsoft_agents.hosting.core.app.oauth.Authorization",
                               return_value=mock_authorization):
                        # Mock AgentApplication to raise ApplicationError
                        class MockAgentApplication:

                            def __class_getitem__(cls, item):
                                return cls

                            def __new__(cls, *args, **kwargs):
                                raise MockApplicationError("SDK app error")

                        with patch("microsoft_agents.hosting.core.AgentApplication", MockAgentApplication, create=True):
                            with patch("microsoft_agents.hosting.core.app.app_error.ApplicationError",
                                       MockApplicationError):
                                with pytest.raises(A365SDKError, match="Failed to create AgentApplication"):
                                    await worker.create_agent_application()


async def test_worker_publishes_turn_identity_during_message_handler():
    """on_message must set the A365 turn identity for the workflow body."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from unittest.mock import Mock

    from nat.plugins.a365.turn_context import A365TurnIdentity
    from nat.plugins.a365.turn_context import get_turn_identity

    captured: dict = {}

    class FakeRunner:

        async def __aenter__(self):
            captured["identity_during_run"] = get_turn_identity()
            return self

        async def __aexit__(self, *exc):
            return False

        async def result(self, to_type=str):
            return "hello"

    fake_session_manager = SimpleNamespace(run=lambda payload: FakeRunner(), )

    worker = A365FrontEndPluginWorker.__new__(A365FrontEndPluginWorker)
    worker.full_config = SimpleNamespace(general=SimpleNamespace(front_end=SimpleNamespace()))
    worker.front_end_config = worker.full_config.general.front_end

    registered: dict = {}

    class FakeApp:

        def activity(self, kind):

            def decorator(fn):
                registered[kind] = fn
                return fn

            return decorator

    await worker.setup_message_handlers(FakeApp(), fake_session_manager)

    activity = SimpleNamespace(
        text="hi",
        is_agentic_request=lambda: True,
        get_agentic_instance_id=lambda: "turn-agent",
        get_agentic_tenant_id=lambda: "turn-tenant",
        get_agentic_user=lambda: "user-1",
    )
    context = SimpleNamespace(activity=activity, send_activity=AsyncMock())

    handler = registered["message"]
    await handler(context, state=Mock())

    assert captured["identity_during_run"] == A365TurnIdentity(
        agent_app_id="turn-agent",
        tenant_id="turn-tenant",
        on_behalf_user_id="user-1",
    )
    # And it must be reset after the handler returns.
    assert get_turn_identity() is None

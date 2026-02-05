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

"""A365 notifications handler for Teams, Email, and Office 365 integration."""

import logging
from typing import Any, Optional

from microsoft_agents.hosting.core import TurnContext, TurnState
from microsoft_agents_a365.notifications import AgentNotification
from microsoft_agents_a365.notifications.models import AgentNotificationActivity

logger = logging.getLogger(__name__)


class A365NotificationHandler:
    """Handler for Agent 365 notifications from Teams, Email, and Office apps.

    This class wraps the A365 SDK's AgentNotification to provide a NAT-friendly
    interface for handling notifications from various A365 channels.

    Note: This requires integration with Microsoft's Agents SDK hosting framework.
    For NAT workflows, you may need to bridge between NAT's execution model and
    the Agents SDK's TurnContext/TurnState model.

    Example:
        ```python
        from microsoft_agents.hosting.core import Application
        from nat.plugins.a365.notifications import A365NotificationHandler

        app = Application()
        handler = A365NotificationHandler(app)

        @handler.on_email()
        async def handle_email(context: TurnContext, state: TurnState, activity: AgentNotificationActivity):
            # Process email notification
            print(f"Received email: {activity.text}")
        ```
    """

    def __init__(self, app: Any, logger: Optional[logging.Logger] = None):
        """Initialize the A365 notification handler.

        Args:
            app: Microsoft Agents SDK Application instance.
            logger: Optional logger instance. If not provided, uses module logger.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._app = app
        self._notification = AgentNotification(app)

        self._logger.info("Initialized A365 notification handler")

    @property
    def notification(self) -> AgentNotification:
        """Get the underlying AgentNotification instance.

        Returns:
            The AgentNotification instance for advanced usage.
        """
        return self._notification

    def on_email(self, **kwargs: Any):
        """Decorator for handling email notifications.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for email notification handlers.

        Example:
            ```python
            @handler.on_email()
            async def handle_email(context, state, activity):
                # Handle email notification
                pass
            ```
        """
        return self._notification.on_email(**kwargs)

    def on_word(self, **kwargs: Any):
        """Decorator for handling Word document notifications.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for Word notification handlers.
        """
        return self._notification.on_word(**kwargs)

    def on_excel(self, **kwargs: Any):
        """Decorator for handling Excel document notifications.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for Excel notification handlers.
        """
        return self._notification.on_excel(**kwargs)

    def on_powerpoint(self, **kwargs: Any):
        """Decorator for handling PowerPoint document notifications.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for PowerPoint notification handlers.
        """
        return self._notification.on_powerpoint(**kwargs)

    def on_lifecycle(self, **kwargs: Any):
        """Decorator for handling agent lifecycle notifications.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for lifecycle notification handlers.
        """
        return self._notification.on_lifecycle(**kwargs)

    def on_user_created(self, **kwargs: Any):
        """Decorator for handling user created lifecycle events.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for user created handlers.
        """
        return self._notification.on_user_created(**kwargs)

    def on_user_deleted(self, **kwargs: Any):
        """Decorator for handling user deleted lifecycle events.

        Args:
            **kwargs: Additional arguments to pass to the route handler.

        Returns:
            Decorator function for user deleted handlers.
        """
        return self._notification.on_user_deleted(**kwargs)

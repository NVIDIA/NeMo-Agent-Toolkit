# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for Slack monitor tools.
Exports all Slack monitor functions to be available in AIQ toolkit.
"""

from .slack_monitor import slack_monitor_client

# Export all functions
__all__ = ["slack_monitor_client"] 
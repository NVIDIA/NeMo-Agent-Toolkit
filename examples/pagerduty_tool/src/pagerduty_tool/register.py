# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for PagerDuty tools.
Exports all PagerDuty functions to be available in AIQ toolkit.
"""

from .pagerduty import pagerduty_client

# Export all functions
__all__ = ["pagerduty_client"]
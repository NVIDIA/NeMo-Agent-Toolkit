# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for Confluence tools.
Exports all Confluence functions to be available in AIQ toolkit.
"""

from .confluence import confluence_client

# Export all functions
__all__ = ["confluence_client"] 
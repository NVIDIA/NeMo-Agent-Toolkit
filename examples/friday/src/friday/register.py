# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for Friday AI Assistant tools.
Exports all Confluence and Loki functions to be available in AIQ toolkit.
"""

from .confluence_tools import confluence_client
from .loki_tools import loki_log_analyzer

# Export all functions
__all__ = ["confluence_client", "loki_log_analyzer"] 
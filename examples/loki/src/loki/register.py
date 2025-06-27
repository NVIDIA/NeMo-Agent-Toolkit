# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for Loki integration tools.
Exports all Loki functions to be available in AIQ toolkit.
"""

from .loki_tools import loki_log_analyzer

# Export all functions
__all__ = ["loki_log_analyzer"] 
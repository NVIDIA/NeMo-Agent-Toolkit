# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for GitLab tools.
Exports all GitLab functions to be available in AIQ toolkit.
"""

from .gitlab_tools import gitlab_getfile

# Export all functions
__all__ = ["gitlab_getfile"] 
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OptionalImportError(Exception):
    """Raised when an optional import fails."""
    pass


def optional_import(module_name: str, package: str | None = None) -> Any:
    """
    Attempt to import a module that might not be installed.

    Args:
        module_name: The name of the module to import
        package: Optional package name for relative imports

    Returns:
        The imported module if successful

    Raises:
        OptionalImportError: If the module cannot be imported
    """
    try:
        return importlib.import_module(module_name, package)
    except ImportError as e:
        raise OptionalImportError(f"Optional dependency '{module_name}' is not installed. "
                                  f"Please install it with 'uv pip install aiq[telemetry]'") from e


# Convenience functions for commonly used optional imports
def get_opentelemetry():
    """Get the opentelemetry module if available."""
    return optional_import("opentelemetry")


def get_opentelemetry_sdk():
    """Get the opentelemetry.sdk module if available."""
    return optional_import("opentelemetry.sdk")


def get_phoenix():
    """Get the phoenix module if available."""
    return optional_import("phoenix")

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

"""Custom exceptions for A365 plugin (shared across all modules)."""


class A365Error(Exception):
    """Base exception for A365 plugin errors."""
    pass


class A365AuthenticationError(A365Error):
    """Authentication-related errors.
    
    Used for authentication failures across A365 modules:
    - Front-end: Bot Framework authentication failures
    - Tooling: A365 Gateway and MCP server authentication failures
    - Telemetry: Token resolver authentication failures
    """
    
    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class A365ConfigurationError(A365Error):
    """Configuration-related errors.
    
    Used for configuration validation failures across A365 modules:
    - Front-end: Invalid front-end configuration (missing fields, wrong types)
    - Tooling: Invalid tooling configuration (reconnect settings, auth config)
    - Telemetry: Invalid telemetry configuration (token resolver path)
    """
    
    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class A365WorkflowExecutionError(A365Error):
    """Errors during workflow execution.
    
    Used when NAT workflows fail during execution in A365 handlers.
    """
    
    def __init__(self, message: str, workflow_type: str = "workflow", original_error: Exception | None = None):
        super().__init__(message)
        self.workflow_type = workflow_type
        self.original_error = original_error


class A365SDKError(A365Error):
    """Errors related to Microsoft Agents SDK components.
    
    Used for SDK-related errors across A365 modules:
    - Front-end: Microsoft Agents SDK (AgentApplication, CloudAdapter, etc.)
    - Telemetry: Agent365Exporter SDK errors
    - Tooling: McpToolServerConfigurationService SDK errors
    """
    
    def __init__(self, message: str, sdk_component: str | None = None, original_error: Exception | None = None):
        super().__init__(message)
        self.sdk_component = sdk_component
        self.original_error = original_error

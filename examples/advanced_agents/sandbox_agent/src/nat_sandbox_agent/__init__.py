# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Sandbox Agent - A general-purpose agent with sandboxed execution capabilities.

This agent can execute code, browse the web, manipulate files, and generate
documents, all within an isolated sandbox environment (Docker or Daytona).

Features:
    - Shell command execution in isolated containers
    - Python code execution for data processing
    - Web browsing and content extraction
    - Web search using DuckDuckGo
    - File operations (read, write, list, delete)
    - Document generation (PDF, Word, PowerPoint, HTML, Markdown)

Usage:
    # CLI mode
    nat run configs/config.yaml

    # Evaluation mode
    nat eval configs/config_gaia.yaml
"""

__version__ = "0.1.0"

# Import registration to enable NAT component discovery
from nat_sandbox_agent import register  # noqa: F401

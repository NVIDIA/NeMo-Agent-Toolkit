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

"""Base executor class for sandbox tools."""

import logging

from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.tools.common import DEFAULT_MAX_OUTPUT_CHARS
from nat_sandbox_agent.tools.common import truncate_output

logger = logging.getLogger(__name__)


class SandboxToolExecutor:
    """Base execution layer for sandbox-based tools.

    This class provides the core sandbox execution capabilities that
    individual tool modules build upon. All tools share the same sandbox
    instance, ensuring consistent file system state across operations.
    """

    def __init__(
        self,
        sandbox: BaseSandbox,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        default_timeout: float = 120,
    ):
        """Initialize the tool executor.

        Args:
            sandbox: The sandbox instance to use for execution.
            max_output_chars: Maximum characters to return in output.
            default_timeout: Default timeout for operations in seconds.
        """
        self.sandbox = sandbox
        self.max_output_chars = max_output_chars
        self.default_timeout = default_timeout

    def truncate(self, text: str) -> str:
        """Truncate output to maximum length."""
        return truncate_output(text, self.max_output_chars)

    async def list_generated_files(self) -> list[str]:
        """List files in the output directory using shell command."""
        try:
            result = await self.sandbox.run_command(
                "ls -1 /workspace/output",
                timeout=self.default_timeout,
            )
            if result.success:
                files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
                return [f"/workspace/output/{f}" for f in files]
            # Log non-success results for debugging
            logger.error(
                f"Failed to list generated files: exit_code={result.exit_code}, "
                f"stderr={result.stderr}"
            )
            return []
        except Exception:
            logger.exception("Exception while listing generated files")
            return []

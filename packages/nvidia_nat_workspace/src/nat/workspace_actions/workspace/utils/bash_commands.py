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
"""Bash command classification and exit-code interpretation utilities."""

from __future__ import annotations

import re


READ_ONLY_COMMANDS: set[str] = {
    # File inspection
    "ls",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "grep",
    "awk",
    "sed",
    "find",
    "which",
    "whereis",
    "type",
    "file",
    "stat",
    "wc",
    "diff",
    "cmp",
    # System info
    "pwd",
    "echo",
    "printf",
    "date",
    "cal",
    "uptime",
    "whoami",
    "id",
    "groups",
    "env",
    "printenv",
    "set",
    "export",
    "hostname",
    "uname",
    "df",
    "du",
    "free",
    "ps",
    "top",
    "htop",
    "pgrep",
    "lsof",
    "netstat",
    "ss",
    "ip",
    "ifconfig",
    # VCS operations (read-only)
    "git status",
    "git log",
    "git diff",
    "git show",
    "git branch",
    "git remote",
    "git tag",
    "git describe",
    "git rev-parse",
    "git ls-files",
    "git blame",
    # Package managers (query only)
    "npm list",
    "npm ls",
    "npm view",
    "npm info",
    "npm outdated",
    "npm audit",
    "yarn list",
    "yarn info",
    "yarn why",
    "pnpm list",
    "pnpm why",
    "pip list",
    "pip show",
    "pip freeze",
    "pip check",
    "cargo tree",
    "cargo metadata",
    "cargo check",
    "go list",
    "go mod graph",
    "go version",
    # Container/orchestration
    "docker ps",
    "docker images",
    "docker logs",
    "docker inspect",
    "kubectl get",
    "kubectl describe",
    "kubectl logs",
    # Modern tools
    "tree",
    "exa",
    "bat",
    "rg",
    "fd",
    "ag",
    "ack",
    "jq",
    "yq",
    "curl --head",
    "curl -I",
    "wget --spider",
    "test",
    "[",
    "[[",
}


LONG_RUNNING_COMMANDS: list[str] = [
    "npm run dev",
    "npm run start",
    "npm run serve",
    "npm run watch",
    "yarn dev",
    "yarn start",
    "yarn serve",
    "yarn watch",
    "pnpm dev",
    "pnpm start",
    "pnpm serve",
    "pnpm watch",
    "python -m http.server",
    "python3 -m http.server",
    "node server",
    "nodemon",
    "ts-node-dev",
    "cargo run",
    "cargo watch",
    "go run",
    "docker-compose up",
    "docker compose up",
    "kubectl port-forward",
    "kubectl proxy",
    "tail -f",
    "watch",
    "webpack --watch",
    "vite",
    "next dev",
    "nuxt dev",
]


_IMAGE_DATA_URI_RE = re.compile(r"^data:image/[a-z0-9.+_-]+;base64,", re.IGNORECASE)


def is_read_only_command(command: str) -> bool:
    """Return whether a command is considered read-only."""
    trimmed = command.strip().lower()
    if not trimmed:
        return False

    first_word = trimmed.split()[0]

    # Check single-word commands first.
    if first_word in READ_ONLY_COMMANDS:
        return True

    # Then check multi-word prefixes (e.g., "git status", "npm list").
    for read_only in READ_ONLY_COMMANDS:
        if " " in read_only and trimmed.startswith(read_only):
            return True

    # For piped commands, all parts must be read-only.
    if "|" in command:
        parts = [part.strip() for part in command.split("|")]
        return all(is_read_only_command(part) for part in parts)

    return False


def is_long_running_command(command: str) -> bool:
    """Return whether a command is likely long-running."""
    trimmed = command.strip().lower()
    return any(trimmed.startswith(cmd) for cmd in LONG_RUNNING_COMMANDS)


def interpret_exit_code(code: int | None, sig: str | None) -> str | None:
    """Map exit codes/signals to human-readable explanations."""
    if sig:
        signal_map = {
            "SIGTERM": "Process was terminated",
            "SIGKILL": "Process was forcefully killed",
            "SIGINT": "Process was interrupted (Ctrl+C)",
            "SIGSEGV": "Process crashed (segmentation fault)",
            "SIGPIPE": "Broken pipe (output destination closed)",
        }
        return signal_map.get(sig, f"Process received signal: {sig}")

    if code is None:
        return None

    match code:
        case 0 | 1 | 2:
            return None
        case 126:
            return "Command found but not executable"
        case 127:
            return "Command not found"
        case 128:
            return "Invalid exit argument"
        case 130:
            return "Process terminated by Ctrl+C (SIGINT)"
        case 137:
            return "Process killed (SIGKILL, possibly OOM)"
        case 139:
            return "Segmentation fault (SIGSEGV)"
        case 143:
            return "Process terminated (SIGTERM)"
        case _:
            if 128 < code < 192:
                return f"Process terminated by signal {code - 128}"
            return None


def is_image_output(output: str) -> bool:
    """Return whether output starts with a base64 image data URI."""
    return bool(_IMAGE_DATA_URI_RE.match(output.strip()))

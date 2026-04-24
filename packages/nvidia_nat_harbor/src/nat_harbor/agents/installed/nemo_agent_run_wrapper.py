#!/usr/bin/env python3
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

"""Runtime helpers for NemoAgent wrapper behavior."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import sys
import traceback


def to_bool(value: str | None) -> bool:
    """Parse env-var style boolean strings."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def maybe_enable_debugpy() -> None:
    """Optionally enable debugpy attach based on NVIDIA_NAT_DEBUGPY_* env vars."""
    port_str = os.environ.get("NVIDIA_NAT_DEBUGPY_PORT")
    if not port_str:
        return

    try:
        port = int(port_str)
    except ValueError:
        print(
            f"[nemo-agent-wrapper] Ignoring invalid NVIDIA_NAT_DEBUGPY_PORT={port_str!r}",
            file=sys.stderr,
        )
        return

    host = os.environ.get("NVIDIA_NAT_DEBUGPY_HOST", "127.0.0.1")
    wait_for_client = to_bool(os.environ.get("NVIDIA_NAT_DEBUGPY_WAIT_FOR_CLIENT"))

    try:
        debugpy = importlib.import_module("debugpy")
    except ImportError:
        print(
            "[nemo-agent-wrapper] debugpy not installed; cannot enable debugger attach.",
            file=sys.stderr,
        )
        return

    try:
        debugpy.listen((host, port))
        print(
            f"[nemo-agent-wrapper] debugpy listening on {host}:{port} "
            f"(wait_for_client={wait_for_client})",
            file=sys.stderr,
        )
        if wait_for_client:
            debugpy.wait_for_client()
            print("[nemo-agent-wrapper] debugpy client attached.", file=sys.stderr)
    except Exception:
        print("[nemo-agent-wrapper] Failed to initialize debugpy:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def normalize_result_text(raw_text: str) -> str:
    """Normalize command-style stdout into raw JSON payload when applicable."""
    text = raw_text.strip()
    if not text:
        return text

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    try:
        parts = shlex.split(text)
    except ValueError:
        return text

    if (
        parts
        and parts[0] == "echo"
        and ">" in parts
        and "/app/result.json" in parts
        and len(parts) >= 2
    ):
        payload = parts[1]
        try:
            json.loads(payload)
            return payload
        except json.JSONDecodeError:
            return text

    return text


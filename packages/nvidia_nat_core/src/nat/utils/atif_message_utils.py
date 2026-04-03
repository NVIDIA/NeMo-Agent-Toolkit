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
"""Shared helpers for extracting text from ATIF messages and trajectories."""

from __future__ import annotations

from collections.abc import Sequence

from nat.atif import ATIFContentPart
from nat.atif import ATIFTrajectory


def content_part_to_text(part: ATIFContentPart) -> str:
    """Convert a single ATIF content part to text."""
    if part.type == "text":
        return part.text or ""
    if part.type == "image":
        return part.source.path if part.source else ""
    return ""


def message_to_text(message: str | Sequence[ATIFContentPart] | None) -> str:
    """Convert ATIF message content to plain text."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    return "\n".join([content_part_to_text(part) for part in message if content_part_to_text(part)])


def trajectory_to_user_input(trajectory: ATIFTrajectory) -> str:
    """Return the first non-empty user message from an ATIF trajectory."""
    for step in trajectory.steps:
        if step.source == "user":
            text = message_to_text(step.message)
            if text:
                return text
    return ""

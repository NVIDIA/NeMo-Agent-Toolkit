# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Compiler-related exception types."""

from __future__ import annotations


class GraphValidationError(ValueError):
    """Raised when a framework adapter produces an invalid Graph."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        msg = "Framework adapter produced an invalid Graph:\n" + "\n".join(f"  - {i}" for i in issues)
        super().__init__(msg)

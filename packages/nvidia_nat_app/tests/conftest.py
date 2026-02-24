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
"""Shared test fixtures and helpers for nvidia_nat_app."""

from __future__ import annotations

import pytest

from nat_app.graph.access import AccessSet
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.analysis import NodeAnalysis


@pytest.fixture(autouse=True)
def _suppress_experimental_warning():
    """Suppress the package-level experimental warning during tests."""
    import warnings

    from nat_app import ExperimentalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        yield


class MinimalAdapter(AbstractFrameworkAdapter):
    """Minimal concrete adapter for tests that need an AbstractFrameworkAdapter."""

    def extract(self, source):
        return source

    def build(self, original, result):
        return result


def make_node(
    name: str,
    reads: set[str] | None = None,
    writes: set[str] | None = None,
    confidence: str = "full",
    special_calls: set[str] | None = None,
) -> NodeAnalysis:
    """Build a NodeAnalysis with AccessSets from plain string sets."""
    w = AccessSet.from_set(writes or set())
    return NodeAnalysis(
        name=name,
        reads=AccessSet.from_set(reads or set()),
        writes=w,
        mutations=w,
        confidence=confidence,
        special_calls=special_calls or set(),
    )

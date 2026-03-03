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
"""
Resolution protocol and data structures for speculation strategies.

Defines the contract between speculation plans and executors:
a ``ResolutionPolicy`` determines what to keep, cancel, or re-run
after a decision node completes.

Each speculation strategy implements its own ``ResolutionPolicy``
(e.g. ``RouterBranchResolution`` for full-branch router speculation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import runtime_checkable


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a speculative bet after a decision is known.

    Produced by ``ResolutionPolicy.resolve()``, consumed by executors.
    """

    keep: frozenset[str]
    """Nodes whose speculative results should be kept and merged."""

    cancel: frozenset[str]
    """Nodes to cancel (unchosen or wrong-prediction paths)."""

    rerun: frozenset[str]
    """Nodes that must be re-executed sequentially (prediction misses)."""


@runtime_checkable
class ResolutionPolicy(Protocol):
    """Determines what to keep/cancel after a decision node completes.

    Each speculation strategy provides its own implementation.
    The executor calls ``resolve()`` once the decision node's result
    is available, then acts on the returned ``Resolution``.
    """

    def resolve(self, decision_result: Any) -> Resolution:
        """Resolve speculation given the decision node's output.

        Args:
            decision_result: The result from the decision node, or
                a pre-extracted decision label (strategy-dependent).

        Returns:
            A ``Resolution`` describing what to keep, cancel, and rerun.
        """
        ...

    def is_on_chosen_path(self, node: str, decision_result: Any) -> bool:
        """Check whether *node* belongs to the chosen path.

        Args:
            node: Node name to test.
            decision_result: Decision label or raw result.

        Returns:
            ``True`` if the node is on the chosen path.
        """
        ...

    def get_cancel_set(self, decision_result: Any) -> frozenset[str]:
        """Return the set of nodes to cancel for the given decision.

        Convenience accessor equivalent to ``resolve(decision_result).cancel``.

        Args:
            decision_result: Decision label or raw result.

        Returns:
            Set of node names to cancel.
        """
        ...

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Runtime context management for prediction trie lookups.

Provides tracking of LLM call indices per function invocation,
enabling accurate lookups in the prediction trie at runtime.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field


@dataclass
class LLMCallTracker:
    """Tracks LLM call counts per function invocation."""

    counts: dict[str, int] = field(default_factory=dict)

    def increment(self, parent_function_id: str) -> int:
        """
        Increment and return the call index for this parent.

        Args:
            parent_function_id: Unique ID of the parent function invocation

        Returns:
            The call index (1-indexed) for this LLM call within the parent
        """
        self.counts[parent_function_id] = self.counts.get(parent_function_id, 0) + 1
        return self.counts[parent_function_id]

    def reset(self, parent_function_id: str) -> None:
        """
        Reset call count when a function invocation completes.

        Args:
            parent_function_id: Unique ID of the parent function invocation
        """
        self.counts.pop(parent_function_id, None)


# Thread/async-safe context variable for the call tracker
_llm_call_tracker: ContextVar[LLMCallTracker] = ContextVar("llm_call_tracker")


def get_call_tracker() -> LLMCallTracker:
    """
    Get the LLMCallTracker for the current context.

    Creates a new tracker if one doesn't exist in the current context.

    Returns:
        The LLMCallTracker for this context
    """
    try:
        return _llm_call_tracker.get()
    except LookupError:
        tracker = LLMCallTracker()
        _llm_call_tracker.set(tracker)
        return tracker

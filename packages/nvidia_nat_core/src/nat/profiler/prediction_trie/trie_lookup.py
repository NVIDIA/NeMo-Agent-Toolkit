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

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


class PredictionTrieLookup:
    """Looks up predictions in a prediction trie with graceful fallback."""

    def __init__(self, root: PredictionTrieNode) -> None:
        self._root = root

    def find(self, path: list[str], call_index: int) -> LLMCallPrediction | None:
        """
        Find the best matching prediction for the given path and call index.

        Walks the trie as far as possible along the path, then returns the deepest
        match. Falls back to aggregated predictions when exact call_index isn't found.

        Args:
            path: Function ancestry path (e.g., ["my_workflow", "react_agent"])
            call_index: The Nth LLM call within the current parent function

        Returns:
            Best matching prediction, or None if trie is empty
        """
        node = self._root
        deepest_match: LLMCallPrediction | None = None

        # Check root node first
        deepest_match = self._get_prediction(node, call_index) or deepest_match

        # Walk the trie as far as we can match
        for func_name in path:
            if func_name not in node.children:
                break
            node = node.children[func_name]
            # Update deepest match at each level
            match = self._get_prediction(node, call_index)
            if match is not None:
                deepest_match = match

        return deepest_match

    def _get_prediction(self, node: PredictionTrieNode, call_index: int) -> LLMCallPrediction | None:
        """Get prediction from node, preferring exact call_index, falling back to aggregated."""
        if call_index in node.predictions_by_call_index:
            return node.predictions_by_call_index[call_index]
        return node.predictions_any_index

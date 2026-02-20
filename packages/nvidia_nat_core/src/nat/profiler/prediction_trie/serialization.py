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

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode

CURRENT_VERSION = "1.0"


def save_prediction_trie(
    trie: PredictionTrieNode,
    path: Path,
    workflow_name: str = "unknown",
) -> None:
    """
    Save a prediction trie to a JSON file.

    Args:
        trie: The prediction trie root node
        path: Path to save the JSON file
        workflow_name: Name of the workflow this trie was built from
    """
    data = {
        "version": CURRENT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "workflow_name": workflow_name,
        "root": _serialize_node(trie),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_prediction_trie(path: Path) -> PredictionTrieNode:
    """
    Load a prediction trie from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        The deserialized prediction trie root node
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return _deserialize_node(data["root"])


def _serialize_node(node: PredictionTrieNode) -> dict[str, Any]:
    """Serialize a trie node to a dictionary."""
    result: dict[str, Any] = {
        "name": node.name,
        "predictions_by_call_index": {
            str(k): v.model_dump()
            for k, v in node.predictions_by_call_index.items()
        },
        "predictions_any_index": node.predictions_any_index.model_dump() if node.predictions_any_index else None,
        "children": {
            k: _serialize_node(v)
            for k, v in node.children.items()
        },
    }
    return result


def _deserialize_node(data: dict[str, Any]) -> PredictionTrieNode:
    """Deserialize a dictionary to a trie node."""
    predictions_by_call_index: dict[int, LLMCallPrediction] = {}
    for k, v in data.get("predictions_by_call_index", {}).items():
        predictions_by_call_index[int(k)] = LLMCallPrediction(
            remaining_calls=PredictionMetrics(**v["remaining_calls"]),
            interarrival_ms=PredictionMetrics(**v["interarrival_ms"]),
            output_tokens=PredictionMetrics(**v["output_tokens"]),
        )

    predictions_any_index = None
    if data.get("predictions_any_index"):
        v = data["predictions_any_index"]
        predictions_any_index = LLMCallPrediction(
            remaining_calls=PredictionMetrics(**v["remaining_calls"]),
            interarrival_ms=PredictionMetrics(**v["interarrival_ms"]),
            output_tokens=PredictionMetrics(**v["output_tokens"]),
        )

    children: dict[str, PredictionTrieNode] = {}
    for k, v in data.get("children", {}).items():
        children[k] = _deserialize_node(v)

    return PredictionTrieNode(
        name=data["name"],
        predictions_by_call_index=predictions_by_call_index,
        predictions_any_index=predictions_any_index,
        children=children,
    )

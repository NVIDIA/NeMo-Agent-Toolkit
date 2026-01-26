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

import tempfile
from pathlib import Path

import pytest

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.profiler import PredictionTrieConfig
from nat.data_models.profiler import ProfilerConfig
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.profile_runner import ProfilerRunner


@pytest.fixture(name="sample_traces")
def fixture_sample_traces() -> list[list[IntermediateStep]]:
    """Create sample traces for testing profiler integration."""

    def make_trace() -> list[IntermediateStep]:
        return [
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id="workflow-1",
                    function_name="my_workflow",
                    parent_id=None,
                    parent_name=None,
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_START,
                    event_timestamp=1000.0,
                    UUID="llm-1",
                ),
            ),
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id="workflow-1",
                    function_name="my_workflow",
                    parent_id=None,
                    parent_name=None,
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    event_timestamp=1001.0,
                    span_event_timestamp=1000.0,
                    UUID="llm-1",
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=100)),
                ),
            ),
        ]

    return [make_trace(), make_trace()]


async def test_profiler_generates_prediction_trie(sample_traces):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        config = ProfilerConfig(
            base_metrics=True,
            prediction_trie=PredictionTrieConfig(enable=True),
        )

        runner = ProfilerRunner(config, output_dir)
        await runner.run(sample_traces)

        trie_path = output_dir / "prediction_trie.json"
        assert trie_path.exists()

        trie = load_prediction_trie(trie_path)
        assert trie.name == "root"
        assert "my_workflow" in trie.children

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.data_models.intermediate_step import StreamEventData
from aiq.data_models.invocation_node import InvocationNode
from aiq.profiler.inference_optimization.prompt_caching import get_common_prefixes
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture(name="minimal_valid_df")
def minimal_valid_df_fixture():
    """
    Provide a minimal DataFrame with columns
    [framework, llm_name, llm_text_input].
    """
    # df = pd.DataFrame(data)

    events = [[
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llama-3", function_id="test-llama-3"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         event_timestamp=100.0,
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello world!"))),
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llama-3", function_id="test-llama-3"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                                         event_timestamp=105.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         name="llama-3",
                                                         data=StreamEventData(output="Hello world!"))),
        IntermediateStep(parent_id="root",
                         function_ancestry=InvocationNode(function_name="llama-2", function_id="test-llama-2"),
                         payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_START,
                                                         event_timestamp=200.0,
                                                         framework=LLMFrameworkEnum.LLAMA_INDEX,
                                                         name="llama-2",
                                                         data=StreamEventData(input="Hola mundo!"))),
    ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


###############################################################################
# Tests
###############################################################################


def test_get_common_prefixes_minimal(minimal_valid_df):
    """
    Basic run with minimal valid data => expect some prefix info for each llm_name.
    """
    from aiq.profiler.inference_optimization.data_models import CommonPrefixesOutput
    from aiq.profiler.inference_optimization.data_models import FrameworkLLMPrefixData

    result = get_common_prefixes(minimal_valid_df)
    assert isinstance(result, CommonPrefixesOutput)
    # The root is a dict of {llm_name => FrameworkLLMPrefixData}, though your code uses `f"{llm_name}"` keys
    assert len(result.root) >= 1

    # For example, we might have "llama-3" => ...
    # Check one typical key
    if "llama-3" in result.root:
        flm_data = result.root["llama-3"]
        assert isinstance(flm_data, FrameworkLLMPrefixData)
        # total_calls => 2 for gpt-3.5
        assert flm_data.total_calls == 2
        # prefix_info => list of PrefixInfo
        for pfx in flm_data.prefix_info:
            assert pfx.prefix_length == len(pfx.prefix)
            assert 0.0 <= pfx.calls_percentage <= 1.0


def test_get_common_prefixes_min_call_percentage(minimal_valid_df):
    """
    If we set min_call_percentage=0.6 => only keep prefixes that appear >= 60% of calls in that llm_name group.
    """

    result = get_common_prefixes(minimal_valid_df, min_call_percentage=0.6)

    # Possibly we see fewer prefixes. Let's just check the data structure is valid and we have some filtering done.
    for _, v in result.root.items():
        # Each v => FrameworkLLMPrefixData
        # If it has prefix_info => those are filtered
        for pfx_obj in v.prefix_info:
            # calls_percentage >= 0.6
            assert pfx_obj.calls_percentage >= 0.6, "Expected calls_percentage >= 0.6"

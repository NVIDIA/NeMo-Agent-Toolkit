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
from aiq.data_models.intermediate_step import IntermediateStepType as WorkflowEventEnum
from aiq.data_models.intermediate_step import StreamEventData
from aiq.profiler.inference_optimization.experimental.prefix_span_analysis import prefixspan_subworkflow_with_text
from aiq.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

###############################################################################
# Reuse or define minimal_valid_df fixture
###############################################################################


@pytest.fixture(name="minimal_valid_df")
def minimal_valid_df_fixture():
    """
    The minimal valid DataFrame provided in the prompt, plus the extra columns
    needed by your script: 'num_llm_calls' etc.
    """
    events = [[
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_START,
                                                         event_timestamp=1.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1",
                                                         name="llama-3",
                                                         data=StreamEventData(input="Hello world"))),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                         event_timestamp=1.5,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2",
                                                         name="weather-search")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                         event_timestamp=1.6,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u2",
                                                         name="weather-search")),
        IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.LLM_END,
                                                         event_timestamp=2.0,
                                                         framework=LLMFrameworkEnum.LANGCHAIN,
                                                         UUID="u1",
                                                         name="llama-3")),
    ],
              [
                  IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_START,
                                                                   event_timestamp=10.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3",
                                                                   name="google-search")),
                  IntermediateStep(payload=IntermediateStepPayload(event_type=WorkflowEventEnum.TOOL_END,
                                                                   event_timestamp=11.0,
                                                                   framework=LLMFrameworkEnum.LANGCHAIN,
                                                                   UUID="u3",
                                                                   name="google-search")),
              ]]

    return [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps] for steps in events]


###############################################################################
# Test Cases
###############################################################################


def test_prefixspan_subworkflow_with_text_basic(minimal_valid_df):
    """
    Minimal valid data => check we get a PrefixSpanSubworkflowResult with some
    patterns or possibly empty, but not an error.
    """
    from aiq.profiler.inference_optimization.data_models import PrefixSpanSubworkflowResult

    result = prefixspan_subworkflow_with_text(minimal_valid_df, min_support=1, top_k=5)
    assert isinstance(result, PrefixSpanSubworkflowResult), "Should return a PrefixSpanSubworkflowResult"

    # textual_report must mention "PrefixSpan Sub-Workflow Mining w/ LLM Text"
    assert "PrefixSpan Sub-Workflow Mining w/ LLM Text" in result.textual_report
    # patterns => list
    assert isinstance(result.patterns, list)


def test_prefixspan_subworkflow_with_text_min_coverage(minimal_valid_df):
    """
    If we set min_coverage=1.0 => pattern must appear in 100% examples => might yield 0 patterns.
    """

    # minimal_valid_df has 2 examples (#1, #2). If a pattern doesn't appear in both => it won't pass coverage=1.0
    result = prefixspan_subworkflow_with_text(minimal_valid_df, min_support=1, top_k=10, min_coverage=1.0)
    # It's possible no pattern is in 100% of examples => we get "No patterns passed coverage/duration thresholds."
    if result.patterns:
        # If we do have a pattern that appears in both examples, coverage must be 1.0
        for pat in result.patterns:
            assert pat.coverage == 1.0, "All patterns must appear in 100% of examples."
    else:
        # We fail quietly if it returns an empty set
        assert "No patterns passed coverage/duration thresholds." in result.textual_report


def test_prefixspan_subworkflow_with_text_prefix_list(minimal_valid_df):
    """
    Provide a prefix_list => ensures text truncation or prefix replacement doesn't crash.
    This isn't a thorough test of string replacement, just that the function runs.
    """

    # We'll add a column llm_text_input to minimal_valid_df for testing
    df_test = minimal_valid_df.copy()
    # Suppose the LLM had some text in example #1
    prefix_list = ["Hello w", "otherprefix"]

    result = prefixspan_subworkflow_with_text(df_test, prefix_list=prefix_list, min_support=1, top_k=5)
    # We don't deeply parse the patterns here, but we ensure it doesn't crash
    # And we might do a quick check on textual_report or patterns
    assert "PrefixSpan Sub-Workflow Mining w/ LLM Text" in result.textual_report
    for pat in result.patterns:
        assert isinstance(pat.pattern, list)
        # The pattern tokens might contain <common_prefix> if the text was replaced
        # This is just a partial check:
        # e.g. "LLM:llama-3|<common_prefix>orld"
        # or if prefix wasn't recognized, we won't see replacement.


def test_prefixspan_subworkflow_with_text_numeric_fields(minimal_valid_df):
    """
    Check coverage, average_duration, frequency in the returned patterns for negative or nonsense values.
    """

    result = prefixspan_subworkflow_with_text(minimal_valid_df, min_support=1)
    for pattern_obj in result.patterns:
        assert pattern_obj.frequency >= 1
        assert 0.0 <= pattern_obj.coverage <= 1.0
        assert pattern_obj.average_duration >= 0.0


def test_prefixspan_subworkflow_with_text_top_k(minimal_valid_df):
    """
    If top_k=1 => we only get 1 pattern in the result (if any).
    """

    result = prefixspan_subworkflow_with_text(minimal_valid_df, min_support=1, top_k=1)
    # If there's at least 1 pattern, we only expect 1 in the list
    if len(result.patterns) > 1:
        pytest.fail("Expected top_k=1 => only 1 pattern in the result.")


def test_prefixspan_subworkflow_with_text_no_patterns(minimal_valid_df):
    """
    If min_support is extremely high => we might get no patterns => textual report says so.
    """
    # Suppose we set min_support=999 => guaranteed no pattern meets this support
    res = prefixspan_subworkflow_with_text(minimal_valid_df, min_support=999)
    assert len(res.patterns) == 0
    assert ("No frequent patterns found" in res.textual_report
            or "No patterns passed coverage/duration thresholds." in res.textual_report)

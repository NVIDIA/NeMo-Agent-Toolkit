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

from nat.data_models.token_usage import TokenUsageBaseModel


def test_flat_counters_are_kept():
    usage = TokenUsageBaseModel(prompt_tokens=10, completion_tokens=5, cached_tokens=4, reasoning_tokens=2)
    assert (usage.cached_tokens, usage.reasoning_tokens) == (4, 2)


def test_openai_nested_details_populate_cached_and_reasoning_tokens():
    usage = TokenUsageBaseModel(
        **{
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {
                "cached_tokens": 64
            },
            "completion_tokens_details": {
                "reasoning_tokens": 20
            },
        })
    assert usage.cached_tokens == 64
    assert usage.reasoning_tokens == 20


def test_nested_details_with_none_values_are_ignored():
    usage = TokenUsageBaseModel(**{
        "prompt_tokens_details": None, "completion_tokens_details": {
            "reasoning_tokens": None
        }
    })
    assert usage.cached_tokens == 0
    assert usage.reasoning_tokens == 0


def test_flat_counters_win_over_nested_details():
    usage = TokenUsageBaseModel(**{"cached_tokens": 7, "prompt_tokens_details": {"cached_tokens": 64}})
    assert usage.cached_tokens == 7

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

from pydantic import BaseModel


class PromptCachingConfig(BaseModel):
    enable: bool = False
    min_frequency: float = 0.5


class BottleneckConfig(BaseModel):
    enable_simple_stack: bool = False
    enable_nested_stack: bool = False


class ConcurrencySpikeConfig(BaseModel):
    enable: bool = False
    spike_threshold: int = 1


class PrefixSpanConfig(BaseModel):
    enable: bool = False
    min_support: float = 2
    min_coverage: float = 0
    max_text_len: int = 1000
    top_k: int = 10
    chain_with_common_prefixes: bool = False


class ProfilerConfig(BaseModel):

    base_metrics: bool = False
    token_usage_forecast: bool = False
    token_uniqueness_forecast: bool = False
    workflow_runtime_forecast: bool = False
    compute_llm_metrics: bool = False
    csv_exclude_io_text: bool = False
    prompt_caching_prefixes: PromptCachingConfig = PromptCachingConfig()
    bottleneck_analysis: BottleneckConfig = BottleneckConfig()
    concurrency_spike_analysis: ConcurrencySpikeConfig = ConcurrencySpikeConfig()
    prefix_span_analysis: PrefixSpanConfig = PrefixSpanConfig()

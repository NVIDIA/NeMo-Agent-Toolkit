# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from nat.data_models.intermediate_step import TokenUsageBaseModel
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.span import Span

logger = logging.getLogger(__name__)


def extract_token_usage(span: Span) -> TokenUsageBaseModel:
    # Extract usage information from span attributes using structured models
    token_usage = TokenUsageBaseModel(prompt_tokens=span.attributes.get("llm.token_count.prompt", 0),
                                      completion_tokens=span.attributes.get("llm.token_count.completion", 0),
                                      total_tokens=span.attributes.get("llm.token_count.total", 0))

    return token_usage


def extract_usage_info(span: Span) -> UsageInfo:
    # Get additional usage metrics from span attributes
    token_usage = extract_token_usage(span)
    num_llm_calls = span.attributes.get("nat.usage.num_llm_calls", 0)
    seconds_between_calls = span.attributes.get("nat.usage.seconds_between_calls", 0)

    usage_info = UsageInfo(token_usage=token_usage,
                           num_llm_calls=num_llm_calls,
                           seconds_between_calls=seconds_between_calls)

    return usage_info


def extract_timestamp(span: Span) -> int:
    timestamp = span.attributes.get("nat.event_timestamp", 0)
    try:
        timestamp_int = int(float(str(timestamp)))
    except (ValueError, TypeError) as e:
        logger.warning("Invalid timestamp in span %s: %s, using 0", span.name, str(e))
        timestamp_int = 0

    return timestamp_int

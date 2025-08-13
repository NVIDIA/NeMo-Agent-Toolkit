# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .trace_source_adapter import TraceSourceAdapter
from .langchain import LangChainNimAdapter
from .langchain import LangChainOpenAIAdapter

# Auto-register default adapters
# Import here to avoid circular dependencies
from ..trace_adapter_registry import register_span_adapter

register_span_adapter(LangChainNimAdapter())
register_span_adapter(LangChainOpenAIAdapter())

__all__ = [
    "TraceSourceAdapter",
    "LangChainNimAdapter",
    "LangChainOpenAIAdapter",
]

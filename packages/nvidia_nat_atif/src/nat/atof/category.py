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


"""Category vocabulary for ATOF events (spec §4).

ATOF v0.1 uses a CLOSED vocabulary for ``category`` with a ``custom`` +
``category_profile.subtype`` escape hatch for vendor extensions and an
``unknown`` value for tier-1 pass-through producers that cannot classify
the work.

The ``Category`` name is a ``Literal`` type alias so that typed Python
consumers can annotate expected values; validator logic in ``events.py``
tolerates any non-empty string on the wire (consumers MUST NOT reject
unknown ``category`` values per spec §4.3), but producers using the
canonical vocabulary through ``Category`` get static-analysis coverage.

Canonical vocabulary:

- ``agent``      — top-level agent or workflow scope
- ``function``   — generic function or application step
- ``llm``        — LLM call (populates ``category_profile.model_name``)
- ``tool``       — tool invocation (populates ``category_profile.tool_call_id``)
- ``retriever``  — retrieval step (document search, index lookup)
- ``embedder``   — embedding-generation step
- ``reranker``   — result reranking step
- ``guardrail``  — guardrail or validation step
- ``evaluator``  — evaluation or scoring step
- ``custom``     — vendor-defined category; REQUIRES ``category_profile.subtype`` to name it
- ``unknown``    — producer does not know or cannot classify the work
"""

from __future__ import annotations

from typing import Literal

Category = Literal[
    "agent",
    "function",
    "llm",
    "tool",
    "retriever",
    "embedder",
    "reranker",
    "guardrail",
    "evaluator",
    "custom",
    "unknown",
]

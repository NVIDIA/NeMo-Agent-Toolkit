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

import logging
import os
from collections.abc import Mapping
from enum import Enum

from nat.data_models.span import Span
from nat.observability.processor.processor import Processor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class SpanTaggingProcessor(Processor[Span, Span]):
    """Processor that tags spans with multiple key-value metadata attributes.

    This processor adds custom tags to spans by setting attributes with a configurable prefix.
    Tags are applied for each key-value pair in the tags dictionary. The processor uses
    a span prefix (configurable via NAT_SPAN_PREFIX environment variable) to namespace
    the tag attributes.

        Args:
            tags: Mapping of tag keys to their values. Values can be enums (converted to strings) or strings
            span_prefix: The prefix to use for tag attributes (default: from NAT_SPAN_PREFIX env var or "nat")
    """

    def __init__(self, tags: Mapping[str, Enum | str] | None = None, span_prefix: str | None = None):
        self.tags = tags or {}

        if span_prefix is None:
            span_prefix = os.getenv("NAT_SPAN_PREFIX", "nat").strip() or "nat"

        self._span_prefix = span_prefix

    @override
    async def process(self, item: Span) -> Span:
        """Tag the span with all configured tags.

        Args:
            item (Span): The span to tag

        Returns:
            Span: The tagged span with all configured tags applied
        """
        for tag_key, tag_value in self.tags.items():
            key = str(tag_key).strip()
            if not key:
                continue
            value_str = str(tag_value.value) if isinstance(tag_value, Enum) else str(tag_value)
            if value_str == "":
                continue
            item.set_attribute(f"{self._span_prefix}.{key}", value_str)

        return item

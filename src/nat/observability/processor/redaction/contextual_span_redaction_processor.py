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

from collections.abc import Callable
from typing import Any

from nat.data_models.span import Span
from nat.observability.processor.redaction.contextual_redaction_processor import ContextualRedactionProcessor
from nat.observability.processor.redaction.redaction_processor import RedactionDataT
from nat.utils.type_utils import override


class ContextualSpanRedactionProcessor(ContextualRedactionProcessor[Span, RedactionDataT]):
    """Processor that redacts the Span based on the Span attributes.

    Args:
        attributes: List of span attribute keys to redact
        callback: Callable that determines if redaction should occur
        enabled: Whether the processor is enabled
        force_redact: If True, always redact regardless of callback
        redaction_value: The value to replace redacted attributes with
    """

    def __init__(self,
                 attributes: list[str],
                 callback: Callable[..., Any],
                 enabled: bool,
                 force_redact: bool,
                 redaction_value: str,
                 redaction_tag: str | None = None):
        super().__init__(callback=callback, enabled=enabled, force_redact=force_redact, redaction_value=redaction_value)
        self.attributes = attributes
        self.redaction_tag = redaction_tag

    @override
    async def redact_item(self, item: Span) -> Span:
        """Redact specified attributes in the span.

        Replaces the values of configured attributes with the redaction value.

        Args:
            item (Span): The span to redact

        Returns:
            Span: The span with redacted attributes
        """
        for key in self.attributes:
            if key in item.attributes:
                item.set_attribute(key, self.redaction_value)

        if self.redaction_tag:
            item.set_attribute(self.redaction_tag, True)

        return item

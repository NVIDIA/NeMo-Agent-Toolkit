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
from abc import ABC
from abc import abstractmethod
from typing import TypeVar

from nat.builder.context import Context
from nat.data_models.span import Span
from nat.observability.processor.processor import Processor
from nat.utils.type_utils import override

RedactionItemT = TypeVar('RedactionItemT')

logger = logging.getLogger(__name__)


class RedactionProcessor(Processor[RedactionItemT, RedactionItemT], ABC):
    """Abstract base class for redaction processors."""

    @abstractmethod
    def should_redact(self, item: RedactionItemT, context: Context) -> bool:
        """Determine if this item should be redacted.

        Args:
            item (RedactionItemT): The item to check.
            context (Context): The current context.

        Returns:
            bool: True if the item should be redacted, False otherwise.
        """
        pass

    @abstractmethod
    def redact_item(self, item: RedactionItemT) -> RedactionItemT:
        """Redact the item.

        Args:
            item (RedactionItemT): The item to redact.

        Returns:
            RedactionItemT: The redacted item.
        """
        pass

    @override
    async def process(self, item: RedactionItemT) -> RedactionItemT:
        """Perform redaction on the item if it should be redacted.

        Args:
            item (RedactionItemT): The item to process.

        Returns:
            RedactionItemT: The processed item.
        """
        context = Context.get()
        if self.should_redact(item, context):
            return self.redact_item(item)
        return item


class SpanRedactionProcessor(RedactionProcessor[Span]):
    """Abstract base class for span redaction processors."""
    pass

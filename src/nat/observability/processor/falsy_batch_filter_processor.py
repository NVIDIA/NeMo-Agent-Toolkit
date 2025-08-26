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
from typing import TypeVar

from nat.observability.processor.processor import Processor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

FalsyT = TypeVar("FalsyT")


class FalsyBatchFilterProcessor(Processor[list[FalsyT], list[FalsyT]]):
    """Processor that filters out falsy items from a batch."""

    @override
    async def process(self, item: list[FalsyT]) -> list[FalsyT]:
        """Filter out falsy items from a batch.

        Args:
            item (list[FalsyT]): The batch of items to filter.

        Returns:
            list[FalsyT]: The filtered batch.
        """
        return [i for i in item if i]


class DictBatchFilterProcessor(FalsyBatchFilterProcessor[dict]):
    """Processor that filters out empty dict items from a batch."""
    pass


class ListBatchFilterProcessor(FalsyBatchFilterProcessor[list]):
    """Processor that filters out empty list items from a batch."""
    pass


class SetBatchFilterProcessor(FalsyBatchFilterProcessor[set]):
    """Processor that filters out empty set items from a batch."""
    pass

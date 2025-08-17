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
from typing import cast

from pydantic import BaseModel

from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import Span
from nat.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin
from nat.observability.processor.processor import Processor
from nat.plugins.data_flywheel.observability.processor.trace_conversion import span_to_dfw_record
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

DFWRecordT = TypeVar("DFWRecordT", bound=BaseModel)


class DFWToDictProcessor(Processor[DFWRecordT, dict]):
    """Processor that converts a Span to an OtelSpan."""

    @override
    async def process(self, item: DFWRecordT | None) -> dict:
        """Convert a DFW record to a dictionary.

        Args:
            item (DFWRecordT | None): The DFW record to convert.

        Returns:
            dict: The converted dictionary.
        """
        if item is None:
            logger.debug("Cannot process 'None' item, returning empty dict")
            return {}

        return item.model_dump(by_alias=True)


class SpanToDFWRecordProcessor(Processor[Span, DFWRecordT], TypeIntrospectionMixin):
    """Processor that converts a Span to an OtelSpan."""

    def __init__(self, client_id: str):
        self._client_id = client_id

    @override
    async def process(self, item: Span) -> DFWRecordT | None:
        """Convert a Span to a DFW record.

        Args:
            item (Span): The Span to convert.

        Returns:
            DFWRecordT | None: The converted DFW record.
        """

        match item.attributes.get("nat.event_type"):
            case IntermediateStepType.LLM_START:
                dfw_record = span_to_dfw_record(span=item, to_type=self.output_type, client_id=self._client_id)
                return cast(DFWRecordT | None, dfw_record)
            case _:
                logger.debug("Unsupported event type: '%s'", item.attributes.get("nat.event_type"))
                return None

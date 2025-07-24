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
from abc import abstractmethod
from typing import TypeVar

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.observability.exporter.processing_exporter import ProcessingExporter
from aiq.utils.type_utils import override

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class RawExporter(ProcessingExporter[InputT, OutputT]):
    """A base class for exporting raw intermediate steps.

    This class provides a base implementation for telemetry exporters that
    work directly with IntermediateStep objects. It can optionally process
    them through a pipeline before export.

    The flow is: IntermediateStep -> [Processing Pipeline] -> OutputT -> Export

    Args:
        context_state (AIQContextState, optional): The context state to use for the exporter. Defaults to None.
    """

    @abstractmethod
    async def export_processed(self, item: OutputT):
        pass

    @override
    def export(self, event: IntermediateStep):
        if not isinstance(event, IntermediateStep):
            return

        self._create_export_task(self._export_with_processing(event))  # type: ignore

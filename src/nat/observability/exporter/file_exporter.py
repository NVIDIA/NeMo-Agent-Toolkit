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

from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.observability.exporter.raw_exporter import RawExporter
from nat.observability.mixin.file_mixin import FileExportMixin
from nat.observability.processor.intermediate_step_serializer import IntermediateStepSerializer

logger = logging.getLogger(__name__)


class FileExporter(FileExportMixin, RawExporter[IntermediateStep, str]):
    """A File exporter that exports telemetry traces to a local file."""

    def __init__(self, context_state: ContextState | None = None, **file_kwargs):
        super().__init__(context_state=context_state, **file_kwargs)
        self._processor = IntermediateStepSerializer()
        self.add_processor(self._processor)

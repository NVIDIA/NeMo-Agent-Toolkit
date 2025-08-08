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

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.observability.mixin.serialize_mixin import SerializeMixin
from aiq.observability.processor.processor import Processor
from aiq.utils.type_utils import override


class IntermediateStepSerializer(SerializeMixin, Processor[IntermediateStep, str]):
    """A File processor that exports telemetry traces to a local file."""

    @override
    async def process(self, item: IntermediateStep) -> str:
        serialized_payload, _ = self._serialize_payload(item)
        return serialized_payload

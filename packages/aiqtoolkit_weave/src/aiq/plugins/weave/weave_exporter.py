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

from aiq.data_models.span import Span
from aiq.observability.exporter.span_exporter import SpanExporter
from aiq.plugins.weave.mixins.weave_mixin import WeaveMixin

logger = logging.getLogger(__name__)


class WeaveExporter(WeaveMixin, SpanExporter[Span, Span]):  # pylint: disable=R0901
    """A Weave exporter that exports telemetry traces to Weights & Biases Weave using OpenTelemetry."""

    def __init__(self, context_state=None, **weave_kwargs):
        super().__init__(context_state=context_state, **weave_kwargs)

    async def _cleanup(self) -> None:
        await self._cleanup_weave_calls()
        await super()._cleanup()

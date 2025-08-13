# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.plugins.data_flywheel.observability.schema.dfw_record import DFWRecord
from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource

logger = logging.getLogger(__name__)


class TraceSourceAdapter(ABC):
    """Abstract base class for trace source adapters."""

    @abstractmethod
    def can_handle(self, trace_source: TraceSource) -> bool:
        """Check if this adapter can handle the given trace source."""
        pass

    @abstractmethod
    def convert(self, trace_source: TraceSource, client_id: str) -> DFWRecord | None:
        """Convert trace source to DFW record."""
        pass

    @property
    @abstractmethod
    def framework_identifier(self) -> str:
        """Return the framework identifier this adapter handles."""
        pass

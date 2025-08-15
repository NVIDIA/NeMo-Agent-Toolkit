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
from typing import Generic
from typing import TypeVar

from nat.plugins.data_flywheel.observability.schema.trace_source import TraceSource
from nat.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)

OutputT = TypeVar("OutputT")


class TraceSourceAdapter(ABC, Generic[OutputT]):
    """Abstract base class for trace source adapters."""

    @abstractmethod
    def can_handle(self, trace_source: TraceSource) -> bool:
        """Check if this adapter can handle the given trace source.

        Args:
            trace_source (TraceSource): The trace source to check

        Returns:
            bool: True if the adapter can handle the trace source, False otherwise
        """
        pass

    @abstractmethod
    def convert(self, trace_source: TraceSource, client_id: str) -> OutputT | None:
        """Convert trace source to DFW record.

        Args:
            trace_source (TraceSource): The trace source to convert
            client_id (str): The client ID to use for the DFW record

        Returns:
            OutputT | None: The converted DFW record
        """
        pass

    @property
    @abstractmethod
    def framework_identifier(self) -> str:
        """Return the framework identifier this adapter handles."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the adapter."""
        pass

    @property
    def output_type(self) -> type[OutputT]:
        """Return the output type this adapter produces."""
        params = DecomposedType.extract_generic_parameters_from_class(self.__class__, expected_param_count=1)
        return params[0]

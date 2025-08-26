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

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import TypeAdapter
from pydantic import field_validator
from pydantic import model_validator

from nat.data_models.span import Span


class TraceContainer(BaseModel):
    """Base TraceContainer model with dynamic union support.

    The source field uses a dynamic union that automatically includes
    all types registered via TraceAdapterRegistry.register_adapter().
    """
    source: Any = Field(..., description="The matched source of the trace")
    span: Span = Field(..., description="The span of the trace")

    @field_validator('source', mode='before')
    @classmethod
    def validate_source_via_union(cls, v):
        """Validate source field using dynamic union."""
        if isinstance(v, dict):
            # Use the dynamic union to validate and select the correct schema
            try:
                from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
                    TraceAdapterRegistry,  # yapf: disable
                )

                current_union = TraceAdapterRegistry.get_current_union()
                if current_union != Any:  # Only validate if union is available
                    adapter = TypeAdapter(current_union)
                    return adapter.validate_python(v)
            except ImportError:
                # Registry not available - return original value
                pass
            except Exception as e:
                # Union validation failed - this should trigger fail-fast in get_trace_source
                raise ValueError(
                    f"Union validation failed: none of the registered schemas match this data structure. {e}") from e
        return v

    @model_validator(mode='before')
    @classmethod
    def ensure_union_built(cls, data):
        """Ensure union is built before validation."""
        # Trigger union building on first instantiation if needed
        try:
            from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
                TraceAdapterRegistry,  # yapf: disable
            )
            TraceAdapterRegistry.get_current_union()  # This ensures union is built and model updated
        except ImportError:
            pass  # Registry not available
        return data

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Update source annotation with current dynamic union when subclassed."""
        super().__init_subclass__(**kwargs)
        # This ensures subclasses get the latest union
        cls.model_rebuild()

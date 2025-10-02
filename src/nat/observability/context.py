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

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class WorkflowMetadata:
    """Metadata for a workflow execution."""
    workflow_name: str
    workflow_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "running"
    parent_workflow_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityContext:
    """
    Context for propagating observability information across workflow executions.

    This class manages trace IDs, span hierarchies, and workflow metadata to enable
    cross-workflow observability tracking.
    """

    trace_id: str
    root_span_id: str
    current_span_id: str
    workflow_chain: List[WorkflowMetadata] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_root_context(cls,
                            workflow_name: str = "root",
                            trace_id: Optional[str] = None,
                            root_span_id: Optional[str] = None) -> "ObservabilityContext":
        """Create a new root observability context."""
        trace_id = trace_id or str(uuid.uuid4())
        root_span_id = root_span_id or str(uuid.uuid4())

        root_workflow = WorkflowMetadata(workflow_name=workflow_name, workflow_id=root_span_id)

        return cls(trace_id=trace_id,
                   root_span_id=root_span_id,
                   current_span_id=root_span_id,
                   workflow_chain=[root_workflow])

    def create_child_context(self, workflow_name: str, workflow_id: Optional[str] = None) -> "ObservabilityContext":
        """Create a child observability context for a sub-workflow."""
        workflow_id = workflow_id or str(uuid.uuid4())

        child_workflow = WorkflowMetadata(workflow_name=workflow_name,
                                          workflow_id=workflow_id,
                                          parent_workflow_id=self.current_span_id)

        # Create new context with extended workflow chain
        new_chain = self.workflow_chain + [child_workflow]

        return ObservabilityContext(trace_id=self.trace_id,
                                    root_span_id=self.root_span_id,
                                    current_span_id=workflow_id,
                                    workflow_chain=new_chain,
                                    custom_attributes=self.custom_attributes.copy())

    def create_span_context(self, span_id: str) -> "ObservabilityContext":
        """Create a new context with a different current span ID."""
        return ObservabilityContext(trace_id=self.trace_id,
                                    root_span_id=self.root_span_id,
                                    current_span_id=span_id,
                                    workflow_chain=self.workflow_chain.copy(),
                                    custom_attributes=self.custom_attributes.copy())

    def add_attribute(self, key: str, value: Any) -> None:
        """Add a custom attribute to the observability context."""
        self.custom_attributes[key] = value

    def get_current_workflow(self) -> Optional[WorkflowMetadata]:
        """Get the currently executing workflow metadata."""
        return self.workflow_chain[-1] if self.workflow_chain else None

    def get_workflow_depth(self) -> int:
        """Get the depth of workflow nesting."""
        return len(self.workflow_chain)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the observability context to a dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "current_span_id": self.current_span_id,
            "workflow_chain": [{
                "workflow_name": w.workflow_name,
                "workflow_id": w.workflow_id,
                "start_time": w.start_time,
                "end_time": w.end_time,
                "status": w.status,
                "parent_workflow_id": w.parent_workflow_id,
                "tags": w.tags
            } for w in self.workflow_chain],
            "custom_attributes": self.custom_attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservabilityContext":
        """Create an observability context from a dictionary."""
        workflow_chain = [
            WorkflowMetadata(workflow_name=w["workflow_name"],
                             workflow_id=w["workflow_id"],
                             start_time=w.get("start_time"),
                             end_time=w.get("end_time"),
                             status=w.get("status", "running"),
                             parent_workflow_id=w.get("parent_workflow_id"),
                             tags=w.get("tags", {})) for w in data.get("workflow_chain", [])
        ]

        return cls(trace_id=data["trace_id"],
                   root_span_id=data["root_span_id"],
                   current_span_id=data["current_span_id"],
                   workflow_chain=workflow_chain,
                   custom_attributes=data.get("custom_attributes", {}))


class ObservabilityContextManager:
    """
    Manager for observability context state using ContextVars.

    This class integrates with NAT's existing ContextState system to provide
    thread-safe observability context propagation.
    """

    _observability_context: ContextVar[Optional[ObservabilityContext]] = ContextVar("observability_context",
                                                                                    default=None)

    @classmethod
    def get_current_context(cls) -> Optional[ObservabilityContext]:
        """Get the current observability context."""
        return cls._observability_context.get()

    @classmethod
    def set_context(cls, context: ObservabilityContext) -> None:
        """Set the current observability context."""
        cls._observability_context.set(context)

    @classmethod
    def create_root_context(cls, workflow_name: str = "root", trace_id: Optional[str] = None) -> ObservabilityContext:
        """Create and set a new root observability context."""
        context = ObservabilityContext.create_root_context(workflow_name=workflow_name, trace_id=trace_id)
        cls.set_context(context)
        return context

    @classmethod
    def create_child_context(cls, workflow_name: str) -> Optional[ObservabilityContext]:
        """Create a child context from the current context."""
        current = cls.get_current_context()
        if current is None:
            return None

        child_context = current.create_child_context(workflow_name)
        cls.set_context(child_context)
        return child_context

    @classmethod
    def propagate_context(cls, context: ObservabilityContext) -> None:
        """Propagate an existing observability context."""
        cls.set_context(context)

    @classmethod
    def clear_context(cls) -> None:
        """Clear the current observability context."""
        cls._observability_context.set(None)

    @classmethod
    def get_trace_id(cls) -> Optional[str]:
        """Get the current trace ID."""
        context = cls.get_current_context()
        return context.trace_id if context else None

    @classmethod
    def get_current_span_id(cls) -> Optional[str]:
        """Get the current span ID."""
        context = cls.get_current_context()
        return context.current_span_id if context else None

    @classmethod
    def add_workflow_attribute(cls, key: str, value: Any) -> None:
        """Add an attribute to the current observability context."""
        context = cls.get_current_context()
        if context:
            context.add_attribute(key, value)

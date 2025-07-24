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

import json
import logging
import time
import traceback
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.sdk import util
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event
from opentelemetry.sdk.trace import InstrumentationScope
from opentelemetry.trace import Context
from opentelemetry.trace import Link
from opentelemetry.trace import SpanContext
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import TraceFlags
from opentelemetry.trace.span import Span
from opentelemetry.util import types

logger = logging.getLogger(__name__)


class MimeTypes(Enum):
    """Mime types for the span."""
    TEXT = "text/plain"
    JSON = "application/json"


class OtelSpan(Span):  # pylint: disable=too-many-public-methods
    """A manually created OpenTelemetry span.

    This class is a wrapper around the OpenTelemetry Span class.
    It provides a more convenient interface for creating and manipulating spans.

    Args:
        name (str): The name of the span.
        context (Context | SpanContext | None): The context of the span.
        parent (Span | None): The parent span.
        attributes (dict[str, Any] | None): The attributes of the span.
        events (list | None): The events of the span.
        links (list | None): The links of the span.
        kind (int | None): The kind of the span.
        start_time (int | None): The start time of the span in nanoseconds.
        end_time (int | None): The end time of the span in nanoseconds.
        status (Status | None): The status of the span.
        resource (Resource | None): The resource of the span.
        instrumentation_scope (InstrumentationScope | None): The instrumentation scope of the span.
    """

    def __init__(
        self,
        name: str,
        context: Context | SpanContext | None,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        events: list | None = None,
        links: list | None = None,
        kind: int | SpanKind | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        status: Status | None = None,
        resource: Resource | None = None,
        instrumentation_scope: InstrumentationScope | None = None,
    ):
        """Initialize the OtelSpan with the specified values."""
        self._name = name
        # Create a new SpanContext if none provided or if Context is provided
        if context is None or isinstance(context, Context):
            trace_id = uuid.uuid4().int & ((1 << 128) - 1)
            span_id = uuid.uuid4().int & ((1 << 64) - 1)
            self._context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_flags=TraceFlags(1),  # SAMPLED
            )
        else:
            self._context = context
        self._parent = parent
        self._attributes = attributes or {}
        self._events = events or []
        self._links = links or []
        self._kind = kind or SpanKind.INTERNAL
        self._start_time = start_time or int(time.time() * 1e9)  # Convert to nanoseconds
        self._end_time = end_time
        self._status = status or Status(StatusCode.UNSET)
        self._ended = False
        self._resource = resource or Resource.create()
        self._instrumentation_scope = instrumentation_scope or InstrumentationScope("aiq", "1.0.0")
        self._dropped_attributes = 0
        self._dropped_events = 0
        self._dropped_links = 0
        self._status_description = None

        # Add parent span as a link if provided
        if parent is not None:
            parent_context = parent.get_span_context()
            # Create a new span context that inherits the trace ID from the parent
            self._context = SpanContext(
                trace_id=parent_context.trace_id,
                span_id=self._context.span_id,
                is_remote=False,
                trace_flags=parent_context.trace_flags,
                trace_state=parent_context.trace_state,
            )
            # Create a proper link object instead of a dictionary
            self._links.append(Link(context=parent_context, attributes={"parent.name": self._name}))

    @property
    def resource(self) -> Resource:
        """Get the resource associated with this span.

        Returns:
            Resource: The resource.
        """
        return self._resource

    def set_resource(self, resource: Resource) -> None:
        """Set the resource associated with this span.

        Args:
            resource (Resource): The resource to set.
        """
        self._resource = resource

    @property
    def instrumentation_scope(self) -> InstrumentationScope:
        """Get the instrumentation scope associated with this span.

        Returns:
            InstrumentationScope: The instrumentation scope.
        """
        return self._instrumentation_scope

    @property
    def parent(self) -> Span | None:
        """Get the parent span.

        Returns:
            Span | None: The parent span.
        """
        return self._parent

    @property
    def name(self) -> str:
        """Get the name of the span.

        Returns:
            str: The name of the span.
        """
        return self._name

    @property
    def kind(self) -> int | SpanKind:
        """Get the kind of the span.

        Returns:
            int | SpanKind: The kind of the span.
        """
        return self._kind

    @property
    def start_time(self) -> int:
        """Get the start time of the span in nanoseconds.

        Returns:
            int: The start time of the span in nanoseconds.
        """
        return self._start_time

    @property
    def end_time(self) -> int | None:
        """Get the end time of the span in nanoseconds.

        Returns:
            int | None: The end time of the span in nanoseconds.
        """
        return self._end_time

    @property
    def attributes(self) -> dict[str, Any]:
        """Get all attributes of the span.

        Returns:
            dict[str, Any]: The attributes of the span.
        """
        return self._attributes

    @property
    def events(self) -> list:
        """Get all events of the span.

        Returns:
            list: The events of the span.
        """
        return self._events

    @property
    def links(self) -> list:
        """Get all links of the span.

        Returns:
            list: The links of the span.
        """
        return self._links

    @property
    def status(self) -> Status:
        """Get the status of the span.

        Returns:
            Status: The status of the span.
        """
        return self._status

    @property
    def dropped_attributes(self) -> int:
        """Get the number of dropped attributes.

        Returns:
            int: The number of dropped attributes.
        """
        return self._dropped_attributes

    @property
    def dropped_events(self) -> int:
        """Get the number of dropped events.

        Returns:
            int: The number of dropped events.
        """
        return self._dropped_events

    @property
    def dropped_links(self) -> int:
        """Get the number of dropped links.

        Returns:
            int: The number of dropped links.
        """
        return self._dropped_links

    @property
    def span_id(self) -> int:
        """Get the span ID.

        Returns:
            int: The span ID.
        """
        return self._context.span_id

    @property
    def trace_id(self) -> int:
        """Get the trace ID.

        Returns:
            int: The trace ID.
        """
        return self._context.trace_id

    @property
    def is_remote(self) -> bool:
        """Get whether this span is remote.

        Returns:
            bool: True if the span is remote, False otherwise.
        """
        return self._context.is_remote

    def end(self, end_time: int | None = None) -> None:
        """End the span.

        Args:
            end_time (int | None): The end time of the span in nanoseconds.
        """
        if not self._ended:
            self._ended = True
            self._end_time = end_time or int(time.time() * 1e9)

    def is_recording(self) -> bool:
        """Check if the span is recording.

        Returns:
            bool: True if the span is recording, False otherwise.
        """
        return not self._ended

    def get_span_context(self) -> SpanContext:
        """Get the span context.

        Returns:
            SpanContext: The span context.
        """
        return self._context

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        Args:
            key (str): The key of the attribute.
            value (Any): The value of the attribute.
        """
        self._attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes on the span.

        Args:
            attributes (dict[str, Any]): The attributes to set.
        """
        self._attributes.update(attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None, timestamp: int | None = None) -> None:
        """Add an event to the span.

        Args:
            name (str): The name of the event.
            attributes (dict[str, Any] | None): The attributes of the event.
            timestamp (int | None): The timestamp of the event in nanoseconds.
        """
        if timestamp is None:
            timestamp = int(time.time() * 1e9)
        self._events.append({"name": name, "attributes": attributes or {}, "timestamp": timestamp})

    def update_name(self, name: str) -> None:
        """Update the span name.

        Args:
            name (str): The name to set.
        """
        self._name = name

    def set_status(self, status: Status, description: str | None = None) -> None:
        """Set the span status.

        Args:
            status (Status): The status to set.
            description (str | None): The description of the status.
        """
        self._status = status
        self._status_description = description

    def get_links(self) -> list:
        """Get all links of the span.

        Returns:
            list: The links of the span.
        """
        return self._links

    def get_end_time(self) -> int | None:
        """Get the end time of the span.

        Returns:
            int | None: The end time of the span in nanoseconds.
        """
        return self._end_time

    def get_status(self) -> Status:
        """Get the status of the span.

        Returns:
            Status: The status of the span.
        """
        return self._status

    def get_parent(self) -> Span | None:
        """Get the parent span.

        Returns:
            Span | None: The parent span.
        """
        return self._parent

    def record_exception(self,
                         exception: Exception,
                         attributes: dict[str, Any] | None = None,
                         timestamp: int | None = None,
                         escaped: bool = False) -> None:
        """
        Record an exception on the span.

        Args:
            exception: The exception to record
            attributes: Optional dictionary of attributes to add to the event
            timestamp: Optional timestamp for the event
            escaped: Whether the exception was escaped
        """

        if timestamp is None:
            timestamp = int(time.time() * 1e9)

        # Get the exception type and message
        exc_type = type(exception).__name__
        exc_message = str(exception)

        # Get the stack trace
        exc_traceback = traceback.format_exception(type(exception), exception, exception.__traceback__)
        stack_trace = "".join(exc_traceback)

        # Create the event attributes
        event_attrs = {
            "exception.type": exc_type,
            "exception.message": exc_message,
            "exception.stacktrace": stack_trace,
        }

        # Add any additional attributes
        if attributes:
            event_attrs.update(attributes)

        # Add the event to the span
        self.add_event("exception", event_attrs)

        # Set the span status to error
        self.set_status(Status(StatusCode.ERROR, exc_message))

    def copy(self) -> "OtelSpan":
        """
        Create a new OtelSpan instance with the same values as this one.
        Note that this is not a deep copy - mutable objects like attributes, events, and links
        will be shared between the original and the copy.

        Returns:
            A new OtelSpan instance with the same values
        """
        return OtelSpan(
            name=self._name,
            context=self._context,
            parent=self._parent,
            attributes=self._attributes.copy(),
            events=self._events.copy(),
            links=self._links.copy(),
            kind=self._kind,
            start_time=self._start_time,
            end_time=self._end_time,
            status=self._status,
            resource=self._resource,
            instrumentation_scope=self._instrumentation_scope,
        )

    @staticmethod
    def _format_context(context: SpanContext) -> dict[str, str]:
        return {
            "trace_id": f"0x{trace_api.format_trace_id(context.trace_id)}",
            "span_id": f"0x{trace_api.format_span_id(context.span_id)}",
            "trace_state": repr(context.trace_state),
        }

    @staticmethod
    def _format_attributes(attributes: types.Attributes, ) -> dict[str, Any] | None:
        if attributes is not None and not isinstance(attributes, dict):
            return dict(attributes)
        return attributes

    @staticmethod
    def _format_events(events: Sequence[Event]) -> list[dict[str, Any]]:
        return [{
            "name": event.name,
            "timestamp": util.ns_to_iso_str(event.timestamp),
            "attributes": OtelSpan._format_attributes(event.attributes),
        } for event in events]

    @staticmethod
    def _format_links(links: Sequence[trace_api.Link]) -> list[dict[str, Any]]:
        return [{
            "context": OtelSpan._format_context(link.context),
            "attributes": OtelSpan._format_attributes(link.attributes),
        } for link in links]

    def to_json(self, indent: int | None = 4):
        parent_id = None
        if self.parent is not None:
            parent_id = f"0x{trace_api.format_span_id(self.parent.span_id)}"  # type: ignore

        start_time = None
        if self._start_time:
            start_time = util.ns_to_iso_str(self._start_time)

        end_time = None
        if self._end_time:
            end_time = util.ns_to_iso_str(self._end_time)

        status = {
            "status_code": str(self._status.status_code.name),
        }
        if self._status.description:
            status["description"] = self._status.description

        f_span = {
            "name": self._name,
            "context": (self._format_context(self._context) if self._context else None),
            "kind": str(self.kind),
            "parent_id": parent_id,
            "start_time": start_time,
            "end_time": end_time,
            "status": status,
            "attributes": self._format_attributes(self._attributes),
            "events": self._format_events(self._events),
            "links": self._format_links(self._links),
            "resource": json.loads(self.resource.to_json()),
        }

        return json.dumps(f_span, indent=indent)

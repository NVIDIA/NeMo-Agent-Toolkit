import logging
import time
import traceback
import uuid
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import InstrumentationScope
from opentelemetry.trace import Context
from opentelemetry.trace import Link
from opentelemetry.trace import SpanContext
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import TraceFlags
from opentelemetry.trace.span import Span

logger = logging.getLogger(__name__)

EVENT_TYPE_TO_SPAN_KIND_MAP = {
    "LLM_START": OpenInferenceSpanKindValues.LLM,
    "LLM_END": OpenInferenceSpanKindValues.LLM,
    "LLM_NEW_TOKEN": OpenInferenceSpanKindValues.LLM,
    "TOOL_START": OpenInferenceSpanKindValues.TOOL,
    "TOOL_END": OpenInferenceSpanKindValues.TOOL,
    "FUNCTION_START": OpenInferenceSpanKindValues.CHAIN,
    "FUNCTION_END": OpenInferenceSpanKindValues.CHAIN,
}


def event_type_to_span_kind(event_type: str) -> OpenInferenceSpanKindValues:
    return EVENT_TYPE_TO_SPAN_KIND_MAP.get(event_type, OpenInferenceSpanKindValues.UNKNOWN)


class OtelSpan(Span):  # pylint: disable=too-many-public-methods
    """A manually created OpenTelemetry span."""

    def __init__(
        self,
        name: str,
        context: Context | SpanContext | None,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        events: list | None = None,
        links: list | None = None,
        kind: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        status: Status | None = None,
        resource: Resource | None = None,
        instrumentation_scope: InstrumentationScope | None = None,
    ):
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
        """Get the resource associated with this span."""
        return self._resource

    def set_resource(self, resource: Resource) -> None:
        """Set the resource associated with this span."""
        self._resource = resource

    @property
    def instrumentation_scope(self) -> InstrumentationScope:
        """Get the instrumentation scope associated with this span."""
        return self._instrumentation_scope

    @property
    def parent(self) -> Span | None:
        """Get the parent span."""
        return self._parent

    @property
    def name(self) -> str:
        """Get the name of the span."""
        return self._name

    @property
    def kind(self) -> int:
        """Get the kind of the span."""
        return self._kind

    @property
    def start_time(self) -> int:
        """Get the start time of the span in nanoseconds."""
        return self._start_time

    @property
    def end_time(self) -> int | None:
        """Get the end time of the span in nanoseconds."""
        return self._end_time

    @property
    def attributes(self) -> dict[str, Any]:
        """Get all attributes of the span."""
        return self._attributes

    @property
    def events(self) -> list:
        """Get all events of the span."""
        return self._events

    @property
    def links(self) -> list:
        """Get all links of the span."""
        return self._links

    @property
    def status(self) -> Status:
        """Get the status of the span."""
        return self._status

    @property
    def dropped_attributes(self) -> int:
        """Get the number of dropped attributes."""
        return self._dropped_attributes

    @property
    def dropped_events(self) -> int:
        """Get the number of dropped events."""
        return self._dropped_events

    @property
    def dropped_links(self) -> int:
        """Get the number of dropped links."""
        return self._dropped_links

    @property
    def span_id(self) -> int:
        """Get the span ID."""
        return self._context.span_id

    @property
    def trace_id(self) -> int:
        """Get the trace ID."""
        return self._context.trace_id

    @property
    def is_remote(self) -> bool:
        """Get whether this span is remote."""
        return self._context.is_remote

    def end(self, end_time: int | None = None) -> None:
        """End the span."""
        if not self._ended:
            self._ended = True
            self._end_time = end_time or int(time.time() * 1e9)

    def is_recording(self) -> bool:
        """Check if the span is recording."""
        return not self._ended

    def get_span_context(self) -> SpanContext:
        """Get the span context."""
        return self._context

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self._attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes on the span."""
        self._attributes.update(attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None, timestamp: int | None = None) -> None:
        """Add an event to the span."""
        if timestamp is None:
            timestamp = int(time.time() * 1e9)
        self._events.append({"name": name, "attributes": attributes or {}, "timestamp": timestamp})

    def update_name(self, name: str) -> None:
        """Update the span name."""
        self._name = name

    def set_status(self, status: Status, description: str | None = None) -> None:
        """Set the span status."""
        self._status = status
        self._status_description = description

    def get_links(self) -> list:
        """Get all links of the span."""
        return self._links

    def get_end_time(self) -> int | None:
        """Get the end time of the span."""
        return self._end_time

    def get_status(self) -> Status:
        """Get the status of the span."""
        return self._status

    def get_parent(self) -> Span | None:
        """Get the parent span."""
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

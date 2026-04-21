# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spec-compliance tests for the ATOF Pydantic models.

Every test pins a specific behavior claimed by ``atof-event-format.md`` so
that a regression in the Pydantic model or the I/O layer is caught
immediately. Tests are grouped by spec section (§2 envelope, §2.1
attributes, §3 event kinds, §4 category, §1/§5 wire + stream semantics).

Where the implementation is deliberately looser than the spec (e.g. ``data``
typed as ``Any`` vs. the spec's "object or null"), a test named
``*_impl_drift_*`` pins current behavior and documents the gap.

Runnable either via pytest or as a standalone script:
    uv run pytest packages/nvidia_nat_atif/tests/test_spec_compliance.py
    uv run python packages/nvidia_nat_atif/tests/test_spec_compliance.py
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterator

from pydantic import TypeAdapter
from pydantic import ValidationError

from nat.atof import Event
from nat.atof import Flags
from nat.atof import MarkEvent
from nat.atof import ScopeEvent
from nat.atof import read_jsonl
from nat.atof import write_jsonl

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@contextmanager
def expect_validation_error(match: str | None = None) -> Iterator[None]:
    """Standalone replacement for ``pytest.raises(ValidationError)``.

    Keeps the suite runnable without pytest. ``match`` is a case-insensitive
    substring check against the error message.
    """
    try:
        yield
    except ValidationError as e:
        if match is not None and match.lower() not in str(e).lower():
            raise AssertionError(f"expected {match!r} in error, got: {e}") from None
        return
    raise AssertionError("expected ValidationError but no exception was raised")


def _scope_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal kwargs for a valid ScopeEvent — overrides merge on top."""
    base: dict[str, Any] = dict(
        scope_category="start",
        uuid="u-1",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00Z",
        name="test",
        category="unknown",
    )
    base.update(overrides)
    return base


def _mark_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal kwargs for a valid MarkEvent — overrides merge on top."""
    base: dict[str, Any] = dict(
        uuid="m-1",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00Z",
        name="checkpoint",
    )
    base.update(overrides)
    return base


# ===========================================================================
# §2 Base Event Envelope
# ===========================================================================


def test_envelope_atof_version_defaults_to_0_1() -> None:
    """§2: atof_version defaults to '0.1'."""
    e = ScopeEvent(**_scope_kwargs())
    assert e.atof_version == "0.1"


def test_envelope_atof_version_accepts_0_minor_values() -> None:
    """§5.6: any '0.MINOR' value in the v0 family is accepted."""
    for v in ("0.1", "0.2", "0.10", "0.99"):
        e = ScopeEvent(**_scope_kwargs(atof_version=v))
        assert e.atof_version == v


def test_envelope_atof_version_rejects_invalid_patterns() -> None:
    """§5.6: non-v0 values and malformed strings raise ValidationError.

    Consumers that want forward compat MUST dispatch on the major version
    and fail fast on unknown majors — this model is a v0 consumer.
    """
    for bad in ("1.0", "0", "0.1.2", "1", "v0.1", "", "0.x"):
        with expect_validation_error("atof_version"):
            ScopeEvent(**_scope_kwargs(atof_version=bad))


def test_envelope_uuid_required_and_non_empty() -> None:
    """§2: uuid is required and must be non-empty."""
    with expect_validation_error("uuid"):
        ScopeEvent(**_scope_kwargs(uuid=""))


def test_envelope_parent_uuid_accepts_none() -> None:
    """§2: parent_uuid MAY be None — root scope / unparented mark."""
    e = ScopeEvent(**_scope_kwargs(parent_uuid=None))
    assert e.parent_uuid is None


def test_envelope_parent_uuid_accepts_uuid_string() -> None:
    """§2: parent_uuid accepts any non-empty string when populated."""
    e = ScopeEvent(**_scope_kwargs(parent_uuid="parent-xyz"))
    assert e.parent_uuid == "parent-xyz"


def test_envelope_parent_uuid_rejects_empty_string() -> None:
    """§2: parent_uuid must be non-empty when populated."""
    with expect_validation_error("parent_uuid"):
        ScopeEvent(**_scope_kwargs(parent_uuid=""))


def test_envelope_timestamp_accepts_rfc3339_string() -> None:
    """§5.1: timestamp accepts an RFC 3339 string."""
    e = ScopeEvent(**_scope_kwargs(timestamp="2026-01-01T00:00:00Z"))
    assert e.timestamp == "2026-01-01T00:00:00Z"


def test_envelope_timestamp_accepts_integer_microseconds() -> None:
    """§5.1: timestamp accepts int epoch microseconds."""
    e = ScopeEvent(**_scope_kwargs(timestamp=1767225600000000))
    assert e.timestamp == 1767225600000000


def test_envelope_ts_micros_computed_from_rfc3339_string() -> None:
    """§5.1: ts_micros is the string timestamp normalized to int microseconds."""
    e = ScopeEvent(**_scope_kwargs(timestamp="2026-01-01T00:00:00Z"))
    # 2026-01-01T00:00:00Z == 1767225600 seconds since epoch
    assert e.ts_micros == 1767225600 * 1_000_000


def test_envelope_ts_micros_passes_through_integer_timestamp() -> None:
    """§5.1: ts_micros passes through when the wire form is already int µs."""
    e = ScopeEvent(**_scope_kwargs(timestamp=1767225600123456))
    assert e.ts_micros == 1767225600123456


def test_envelope_extra_fields_allowed_for_lossless_passthrough() -> None:
    """§7: ConfigDict(extra='allow') keeps unknown fields for round-trip."""
    e = ScopeEvent(**_scope_kwargs(producer_version="v1.2.3", custom_field={"nested": True}))
    assert e.model_extra == {"producer_version": "v1.2.3", "custom_field": {"nested": True}}


def test_envelope_data_accepts_object() -> None:
    """§2: data is typically an object (the spec-conformant case)."""
    e = ScopeEvent(**_scope_kwargs(data={"key": 1}))
    assert e.data == {"key": 1}


def test_envelope_data_accepts_none() -> None:
    """§2: data may be null."""
    e = ScopeEvent(**_scope_kwargs(data=None))
    assert e.data is None


def test_envelope_data_impl_drift_accepts_primitives() -> None:
    """IMPL DRIFT: spec §2 declares ``data: object or null`` but the Pydantic
    model is ``Any | None`` — primitives validate at runtime. This test pins
    current lax behavior.

    If the spec is loosened to "any or null" this test documents parity. If
    the impl is tightened to "dict or null", flip these asserts to
    ``expect_validation_error``.
    """
    assert ScopeEvent(**_scope_kwargs(data="plain string")).data == "plain string"
    assert ScopeEvent(**_scope_kwargs(data=42)).data == 42
    assert ScopeEvent(**_scope_kwargs(data=[1, 2, 3])).data == [1, 2, 3]


def test_envelope_data_schema_accepts_name_version_dict() -> None:
    """§2: data_schema wire shape is ``{name: string, version: string}``."""
    ds = {"name": "openai/chat-completions", "version": "1"}
    e = ScopeEvent(**_scope_kwargs(data_schema=ds))
    assert e.data_schema == ds


def test_envelope_data_schema_accepts_none() -> None:
    """§2: data_schema is optional."""
    e = ScopeEvent(**_scope_kwargs(data_schema=None))
    assert e.data_schema is None


def test_envelope_metadata_accepts_dict_and_none() -> None:
    """§2: metadata is a tracing/correlation envelope, optional dict."""
    e1 = ScopeEvent(**_scope_kwargs(metadata={"trace_id": "abc", "span_id": "def"}))
    assert e1.metadata == {"trace_id": "abc", "span_id": "def"}
    e2 = ScopeEvent(**_scope_kwargs(metadata=None))
    assert e2.metadata is None


# ===========================================================================
# §2.1 Attributes
# ===========================================================================


def test_attributes_defaults_to_empty_list() -> None:
    """§2.1: attributes is required on scope events; defaults to []."""
    e = ScopeEvent(**_scope_kwargs())
    assert e.attributes == []


def test_attributes_canonicalized_sorted() -> None:
    """§2.1: producers MUST emit attributes in lexicographic order."""
    e = ScopeEvent(**_scope_kwargs(attributes=["streaming", "parallel", "remote"]))
    assert e.attributes == ["parallel", "remote", "streaming"]


def test_attributes_canonicalized_deduplicated() -> None:
    """§2.1: duplicates MUST be removed."""
    e = ScopeEvent(**_scope_kwargs(attributes=["remote", "remote", "parallel"]))
    assert e.attributes == ["parallel", "remote"]


def test_attributes_preserves_unknown_flag_names() -> None:
    """§2.1: unknown flag names MUST be preserved — vendor extensions are forward-compat."""
    e = ScopeEvent(**_scope_kwargs(attributes=["nvidia.speculative", "streaming"]))
    assert e.attributes == ["nvidia.speculative", "streaming"]  # 'n' < 's'


def test_attributes_accepts_flags_enum_members() -> None:
    """Flags StrEnum members serialize as their string values."""
    e = ScopeEvent(**_scope_kwargs(attributes=[Flags.STREAMING, Flags.REMOTE]))
    assert e.attributes == ["remote", "streaming"]


def test_attributes_rejects_non_string_entries() -> None:
    """§2.1: attributes MUST be an array of strings."""
    with expect_validation_error():
        ScopeEvent(**_scope_kwargs(attributes=[1, 2, 3]))


def test_attributes_rejects_non_list_value() -> None:
    """§2.1: attributes MUST be a list (not a scalar)."""
    with expect_validation_error():
        ScopeEvent(**_scope_kwargs(attributes="streaming"))


# ===========================================================================
# §3.1 ScopeEvent
# ===========================================================================


def test_scope_kind_is_literal_scope() -> None:
    """§3.1: kind is the literal string 'scope'."""
    e = ScopeEvent(**_scope_kwargs())
    assert e.kind == "scope"


def test_scope_kind_cannot_be_overridden() -> None:
    """§3.1: passing any other kind value raises ValidationError."""
    with expect_validation_error("kind"):
        ScopeEvent(kind="mark", **_scope_kwargs())


def test_scope_category_required() -> None:
    """§3.1: scope_category is a required enum field."""
    kwargs = _scope_kwargs()
    del kwargs["scope_category"]
    with expect_validation_error("scope_category"):
        ScopeEvent(**kwargs)


def test_scope_category_accepts_start_and_end() -> None:
    """§3.1: scope_category ∈ {'start', 'end'}."""
    assert ScopeEvent(**_scope_kwargs(scope_category="start")).scope_category == "start"
    assert ScopeEvent(**_scope_kwargs(scope_category="end")).scope_category == "end"


def test_scope_category_rejects_other_values() -> None:
    """§3.1: values outside {'start', 'end'} are invalid."""
    for bad in ("middle", "START", "", "running"):
        with expect_validation_error("scope_category"):
            ScopeEvent(**_scope_kwargs(scope_category=bad))


def test_scope_category_field_required() -> None:
    """§3.1: category is required on scope events (§4)."""
    kwargs = _scope_kwargs()
    del kwargs["category"]
    with expect_validation_error("category"):
        ScopeEvent(**kwargs)


def test_scope_category_rejects_empty_string() -> None:
    """§4: category must be non-empty."""
    with expect_validation_error("category"):
        ScopeEvent(**_scope_kwargs(category=""))


def test_scope_no_deprecated_v0_0_fields() -> None:
    """Regression guard: v0.0 fields removed during the v0.1 consolidation.

    status, error, input, output, scope_type, profile, annotated_request,
    annotated_response, and the StreamHeader-specific schemas field must not
    reappear on ScopeEvent.
    """
    removed = {
        "status", "error", "input", "output",
        "scope_type", "profile",
        "annotated_request", "annotated_response",
        "schemas",
    }
    for field in removed:
        assert field not in ScopeEvent.model_fields, f"removed field {field!r} reappeared on ScopeEvent"


def test_scope_has_all_required_v0_1_fields() -> None:
    """Regression guard: every v0.1 ScopeEvent field is present."""
    expected = {
        "kind", "scope_category", "atof_version", "uuid", "parent_uuid",
        "timestamp", "name", "attributes", "category", "category_profile",
        "data", "data_schema", "metadata",
    }
    assert expected.issubset(set(ScopeEvent.model_fields)), (
        f"missing fields on ScopeEvent: {expected - set(ScopeEvent.model_fields)}"
    )


# ===========================================================================
# §3.2 MarkEvent
# ===========================================================================


def test_mark_kind_is_literal_mark() -> None:
    """§3.2: kind is the literal string 'mark'."""
    e = MarkEvent(**_mark_kwargs())
    assert e.kind == "mark"


def test_mark_does_not_carry_scope_fields() -> None:
    """§3.2: mark does NOT carry scope_category, attributes, or v0.0 fields."""
    forbidden = {
        "scope_category",  # §3.2 explicit
        "attributes",       # §3.2 explicit
        # v0.0 removed:
        "status", "error", "input", "output",
        "scope_type", "profile",
        "annotated_request", "annotated_response",
        "schemas",
    }
    for field in forbidden:
        assert field not in MarkEvent.model_fields, f"forbidden field {field!r} on MarkEvent"


def test_mark_category_defaults_to_none() -> None:
    """§4: category is OPTIONAL on marks; default is None (generic checkpoint)."""
    e = MarkEvent(**_mark_kwargs())
    assert e.category is None


def test_mark_category_accepts_populated_value() -> None:
    """§4: a mark MAY carry a category to tag the checkpoint."""
    e = MarkEvent(**_mark_kwargs(category="llm", category_profile={"model_name": "gpt-4.1"}))
    assert e.category == "llm"
    assert e.category_profile == {"model_name": "gpt-4.1"}


def test_mark_category_profile_defaults_to_none() -> None:
    """§4.4: category_profile is optional on marks."""
    e = MarkEvent(**_mark_kwargs())
    assert e.category_profile is None


def test_mark_preserves_data_schema_and_data() -> None:
    """§3.2 + §2: mark carries data, data_schema, metadata like scope events do."""
    e = MarkEvent(
        **_mark_kwargs(
            data={"session_id": "s1"},
            data_schema={"name": "myco/session", "version": "1"},
            metadata={"trace_id": "t-1"},
        )
    )
    assert e.data == {"session_id": "s1"}
    assert e.data_schema == {"name": "myco/session", "version": "1"}
    assert e.metadata == {"trace_id": "t-1"}


# ===========================================================================
# §4 Category vocabulary
# ===========================================================================


def test_canonical_categories_all_accepted() -> None:
    """§4: every canonical category value constructs successfully."""
    canonical = (
        "agent", "function", "llm", "tool", "retriever",
        "embedder", "reranker", "guardrail", "evaluator", "unknown",
    )
    for cat in canonical:
        e = ScopeEvent(**_scope_kwargs(category=cat))
        assert e.category == cat


def test_unknown_category_values_accepted() -> None:
    """§4.3: consumers MUST NOT reject unknown category values."""
    e = ScopeEvent(**_scope_kwargs(category="some_future_vendor_category"))
    assert e.category == "some_future_vendor_category"


# ===========================================================================
# §4.2 custom + subtype rule
# ===========================================================================


def test_custom_on_scope_requires_subtype() -> None:
    """§4.2: scope with category='custom' MUST have category_profile.subtype."""
    with expect_validation_error("subtype"):
        ScopeEvent(**_scope_kwargs(category="custom"))
    with expect_validation_error("subtype"):
        ScopeEvent(**_scope_kwargs(category="custom", category_profile={}))
    with expect_validation_error("subtype"):
        ScopeEvent(**_scope_kwargs(category="custom", category_profile={"other": "value"}))


def test_custom_on_scope_with_subtype_succeeds() -> None:
    """§4.2: 'custom' + non-empty subtype constructs successfully."""
    e = ScopeEvent(
        **_scope_kwargs(category="custom", category_profile={"subtype": "nvidia.speculative_decode"})
    )
    assert e.category_profile == {"subtype": "nvidia.speculative_decode"}


def test_custom_on_mark_requires_subtype() -> None:
    """§4.2: subtype rule applies to mark events too (spec explicit)."""
    with expect_validation_error("subtype"):
        MarkEvent(**_mark_kwargs(category="custom"))


def test_custom_on_mark_with_subtype_succeeds() -> None:
    """§4.2: mark with 'custom' + subtype is valid."""
    e = MarkEvent(
        **_mark_kwargs(category="custom", category_profile={"subtype": "vendor.custom_checkpoint"})
    )
    assert e.category == "custom"
    assert e.category_profile == {"subtype": "vendor.custom_checkpoint"}


def test_custom_subtype_rejects_empty_string() -> None:
    """§4.2: subtype must be a non-empty string."""
    with expect_validation_error("subtype"):
        ScopeEvent(**_scope_kwargs(category="custom", category_profile={"subtype": ""}))


def test_non_custom_categories_do_not_require_subtype() -> None:
    """§4.2: subtype is only required when category='custom'."""
    # llm with no subtype: valid
    e1 = ScopeEvent(**_scope_kwargs(category="llm", category_profile={"model_name": "gpt-4.1"}))
    assert e1.category == "llm"
    # unknown with null profile: valid
    e2 = ScopeEvent(**_scope_kwargs(category="unknown", category_profile=None))
    assert e2.category_profile is None


# ===========================================================================
# §4.4 category_profile shapes
# ===========================================================================


def test_llm_category_profile_carries_model_name() -> None:
    """§4.4: llm profile shape is {model_name: str}."""
    e = ScopeEvent(**_scope_kwargs(category="llm", category_profile={"model_name": "gpt-4.1"}))
    assert e.category_profile == {"model_name": "gpt-4.1"}


def test_tool_category_profile_carries_tool_call_id() -> None:
    """§4.4: tool profile shape is {tool_call_id: str}."""
    e = ScopeEvent(**_scope_kwargs(category="tool", category_profile={"tool_call_id": "call_abc"}))
    assert e.category_profile == {"tool_call_id": "call_abc"}


def test_category_profile_preserves_extra_keys() -> None:
    """§4.4: unknown profile keys MUST be preserved verbatim."""
    e = ScopeEvent(
        **_scope_kwargs(
            category="llm",
            category_profile={"model_name": "gpt-4.1", "future_key": "future_value"},
        )
    )
    assert e.category_profile["future_key"] == "future_value"


def test_category_profile_accepts_null_for_tier1() -> None:
    """§4.4: null is legal for tier-1 opaque events and categories without defined keys."""
    # tier-1 unknown
    e1 = ScopeEvent(**_scope_kwargs(category="unknown", category_profile=None))
    assert e1.category_profile is None
    # agent (reserved, no defined keys)
    e2 = ScopeEvent(**_scope_kwargs(category="agent", category_profile=None))
    assert e2.category_profile is None


# ===========================================================================
# §3 Discriminated Event union
# ===========================================================================


def test_event_union_dispatches_scope() -> None:
    """§3: a dict with kind='scope' validates to a ScopeEvent."""
    raw: dict[str, Any] = {
        "kind": "scope",
        "scope_category": "start",
        "atof_version": "0.1",
        "uuid": "u-1",
        "parent_uuid": None,
        "timestamp": "2026-01-01T00:00:00Z",
        "name": "test",
        "attributes": [],
        "category": "unknown",
    }
    adapter = TypeAdapter(Event)
    evt = adapter.validate_python(raw)
    assert isinstance(evt, ScopeEvent)
    assert evt.scope_category == "start"


def test_event_union_dispatches_mark() -> None:
    """§3: a dict with kind='mark' validates to a MarkEvent."""
    raw: dict[str, Any] = {
        "kind": "mark",
        "atof_version": "0.1",
        "uuid": "m-1",
        "parent_uuid": None,
        "timestamp": "2026-01-01T00:00:00Z",
        "name": "checkpoint",
    }
    adapter = TypeAdapter(Event)
    evt = adapter.validate_python(raw)
    assert isinstance(evt, MarkEvent)


def test_event_union_rejects_removed_kinds() -> None:
    """§3: old kinds (ScopeStart, ScopeEnd, Mark capitalised, StreamHeader) are invalid."""
    adapter = TypeAdapter(Event)
    for bad_kind in ("ScopeStart", "ScopeEnd", "Mark", "StreamHeader", "Unknown", ""):
        raw = {
            "kind": bad_kind,
            "atof_version": "0.1",
            "uuid": "u-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "name": "test",
        }
        with expect_validation_error():
            adapter.validate_python(raw)


# ===========================================================================
# §1 Wire envelope + §7 lossless pass-through
# ===========================================================================


def test_wire_round_trip_scope_event_rfc3339() -> None:
    """Write → read yields an equivalent ScopeEvent for every spec-governed field."""
    original = ScopeEvent(
        scope_category="start",
        uuid="u-rt-1",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00Z",
        name="rt_test",
        attributes=["remote", "streaming"],
        category="tool",
        category_profile={"tool_call_id": "call_xyz"},
        data={"a": 1, "nested": {"b": 2}},
        data_schema={"name": "myco/tool", "version": "1"},
        metadata={"trace_id": "t-rt"},
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rt.jsonl"
        write_jsonl([original], path)
        restored = read_jsonl(path)
    assert len(restored) == 1
    r = restored[0]
    assert isinstance(r, ScopeEvent)
    for field in ("scope_category", "uuid", "parent_uuid", "timestamp", "name",
                  "attributes", "category", "category_profile", "data",
                  "data_schema", "metadata"):
        assert getattr(r, field) == getattr(original, field), f"field {field} diverged on round-trip"


def test_wire_round_trip_integer_timestamp() -> None:
    """§5.1: int microsecond timestamps survive JSON round-trip."""
    e = ScopeEvent(**_scope_kwargs(timestamp=1767225600123456))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ts.jsonl"
        write_jsonl([e], path)
        restored = read_jsonl(path)
    assert restored[0].timestamp == 1767225600123456


def test_wire_round_trip_mark_event() -> None:
    """Write → read yields an equivalent MarkEvent."""
    original = MarkEvent(
        uuid="m-rt-1",
        parent_uuid=None,
        timestamp="2026-01-01T00:00:00Z",
        name="session_boundary",
        category="llm",
        category_profile={"model_name": "gpt-4.1"},
        data={"tokens_used": 42},
        data_schema={"name": "myco/session_mark", "version": "1"},
        metadata={"trace_id": "t-m"},
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mark.jsonl"
        write_jsonl([original], path)
        restored = read_jsonl(path)
    assert len(restored) == 1
    r = restored[0]
    assert isinstance(r, MarkEvent)
    assert r.category == original.category
    assert r.category_profile == original.category_profile
    assert r.data == original.data
    assert r.data_schema == original.data_schema


def test_wire_emits_explicit_null_for_optional_none_fields() -> None:
    """§1 wire envelope example: optional None fields serialize as explicit ``null``."""
    event = ScopeEvent(
        scope_category="start",
        uuid="u-null",
        timestamp="2026-01-01T00:00:00Z",
        name="test",
        category="unknown",
        # parent_uuid, data, data_schema, metadata, category_profile all default None
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nulls.jsonl"
        write_jsonl([event], path)
        content = path.read_text()
    # Explicit nulls in the serialized JSON, not dropped keys
    for expected in ('"parent_uuid": null', '"data": null', '"data_schema": null',
                     '"metadata": null', '"category_profile": null'):
        assert expected in content, f"expected {expected!r} literally on the wire; got: {content}"


def test_wire_excludes_computed_ts_micros_field() -> None:
    """§2: ts_micros is a computed sorting convenience; MUST NOT appear on the wire."""
    event = ScopeEvent(**_scope_kwargs())
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nomicros.jsonl"
        write_jsonl([event], path)
        content = path.read_text()
    assert "ts_micros" not in content


def test_wire_preserves_unknown_fields_lossless() -> None:
    """§7: lossless pass-through — unknown fields round-trip unchanged."""
    raw: dict[str, Any] = {
        "kind": "scope",
        "scope_category": "start",
        "atof_version": "0.1",
        "uuid": "u-unknown",
        "parent_uuid": None,
        "timestamp": "2026-01-01T00:00:00Z",
        "name": "test",
        "attributes": [],
        "category": "unknown",
        "category_profile": None,
        "data": None,
        "data_schema": None,
        "metadata": None,
        "vendor_extension": {"nested": "value"},
        "producer_trace_id": "pt-1",
    }
    adapter = TypeAdapter(Event)
    evt = adapter.validate_python(raw)
    assert evt.model_extra == {
        "vendor_extension": {"nested": "value"},
        "producer_trace_id": "pt-1",
    }
    dumped = evt.model_dump(exclude={"ts_micros"}, mode="json", by_alias=True)
    assert dumped["vendor_extension"] == {"nested": "value"}
    assert dumped["producer_trace_id"] == "pt-1"


# ===========================================================================
# §5.1 Stream ordering
# ===========================================================================


def test_read_jsonl_sorts_events_by_ts_micros() -> None:
    """§5.1: read_jsonl returns events sorted by their normalized microsecond timestamp."""
    later = ScopeEvent(**_scope_kwargs(uuid="u-later", timestamp="2026-01-01T00:00:02Z"))
    earlier = ScopeEvent(**_scope_kwargs(uuid="u-earlier", timestamp="2026-01-01T00:00:01Z"))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "unsorted.jsonl"
        # Write out-of-order — later event first
        write_jsonl([later, earlier], path)
        restored = read_jsonl(path)
    # read_jsonl must normalize order
    assert restored[0].uuid == "u-earlier"
    assert restored[1].uuid == "u-later"


def test_read_jsonl_handles_mixed_timestamp_forms() -> None:
    """§5.1: mixed RFC 3339 and int-µs timestamps sort correctly via ts_micros."""
    string_ts = ScopeEvent(**_scope_kwargs(uuid="u-str", timestamp="2026-01-01T00:00:03Z"))
    int_ts = ScopeEvent(**_scope_kwargs(uuid="u-int", timestamp=1767225601_000_000))  # 00:00:01
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mixed.jsonl"
        write_jsonl([string_ts, int_ts], path)
        restored = read_jsonl(path)
    assert restored[0].uuid == "u-int"
    assert restored[1].uuid == "u-str"


# ===========================================================================
# Main runner (standalone mode)
# ===========================================================================


if __name__ == "__main__":
    import sys

    module = sys.modules[__name__]
    tests = [(name, fn) for name, fn in vars(module).items() if name.startswith("test_") and callable(fn)]
    failures: list[tuple[str, BaseException]] = []
    for name, fn in tests:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — surface every failure
            failures.append((name, exc))
            print(f"FAIL: {name}: {exc}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} spec-compliance tests FAILED.")
        sys.exit(1)
    print(f"All {len(tests)} spec-compliance tests passed.")

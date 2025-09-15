import pytest


from nat.utils.optional_imports import (
    DummyBatchSpanProcessor,
    DummySpan,
    DummySpanExporter,
    DummyTrace,
    DummyTracerProvider,
    OptionalImportError,
    TelemetryOptionalImportError,
    optional_import,
    telemetry_optional_import,
)


def test_optional_import_success():
    assert optional_import("math").sqrt(4) == 2


def test_optional_import_failure():
    with pytest.raises(OptionalImportError):
        optional_import("nonexistent___module___xyz")


def test_telemetry_optional_import_failure_has_guidance():
    with pytest.raises(TelemetryOptionalImportError) as ei:
        telemetry_optional_import("not_real_otel_mod")
    assert "Optional dependency" in str(ei.value)
    assert "telemetry" in str(ei.value).lower()


def test_dummy_tracer_stack():
    tracer = DummyTracerProvider.get_tracer()
    span = tracer.start_span("op")
    assert isinstance(span, DummySpan)
    span.set_attribute("k", "v")
    span.end()
    DummyBatchSpanProcessor().shutdown()
    DummySpanExporter.export()
    DummySpanExporter.shutdown()
    assert DummyTrace.get_tracer_provider() is not None
    DummyTrace.set_tracer_provider(None)
    assert DummyTrace.get_tracer("name") is not None

"""Tests for utils.trace_exporters (C12)."""

from rtsa.utils.trace_exporters import (
    NullTraceExporter, OtlpTraceExporter, make_trace_exporter,
)


def test_null_exporter_span_is_noop():
    exporter = NullTraceExporter()
    with exporter.span("x", {"a": 1}) as span:
        span.record("k", "v")  # must not raise


def test_make_exporter_default_is_null(monkeypatch):
    monkeypatch.delenv("RTSA_TRACE_EXPORTER", raising=False)
    assert isinstance(make_trace_exporter(), NullTraceExporter)


def test_make_exporter_none_kind():
    assert isinstance(make_trace_exporter("none"), NullTraceExporter)


def test_make_exporter_unknown_kind_degrades():
    assert isinstance(make_trace_exporter("totally_unknown"), NullTraceExporter)


def test_make_exporter_missing_backend_degrades():
    """otlp/langfuse are optional deps; a missing backend must degrade
    gracefully instead of crashing the experiment."""
    exporter = make_trace_exporter("otlp")
    assert isinstance(exporter, (NullTraceExporter, OtlpTraceExporter))


def test_env_var_drives_kind(monkeypatch):
    monkeypatch.setenv("RTSA_TRACE_EXPORTER", "none")
    assert isinstance(make_trace_exporter(), NullTraceExporter)

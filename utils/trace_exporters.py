"""Observability adapters (C12): OTLP and Langfuse trace export.

Wraps OpenTelemetry (OTLP) and Langfuse behind a tiny uniform interface so
pipeline code can emit execution spans without hard-wiring a vendor:

    from utils.trace_exporters import make_trace_exporter

    exporter = make_trace_exporter()   # honours RTSA_TRACE_EXPORTER env var
    with exporter.span("prune", {"dataset": "gsm8k"}) as span:
        ...                            # span.record("n_nodes", 42)

Both backends are optional imports: when the package is missing (or the
exporter name is unknown) :func:`make_trace_exporter` degrades to a no-op
``NullTraceExporter`` with a warning instead of crashing the experiment.
Select the backend with ``RTSA_TRACE_EXPORTER=otlp|langfuse`` or the
``kind`` argument.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_VAR = "RTSA_TRACE_EXPORTER"


class _NullSpan(AbstractContextManager):
    """No-op span: records nothing, never fails."""

    def __init__(self) -> None:
        pass

    def record(self, key: str, value: Any) -> None:
        pass

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class TraceExporter:
    """Minimal span-emitting interface implemented by all backends."""

    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        raise NotImplementedError


class NullTraceExporter(TraceExporter):
    """Backend used when no observability dependency is installed."""

    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        return _NullSpan()


class OtlpTraceExporter(TraceExporter):
    """OpenTelemetry backend sending spans to an OTLP collector (gRPC)."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        service_name: str = "rtsa",
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            raise ImportError(
                "OpenTelemetry packages required. Install: "
                "pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint)
        ))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(service_name)

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        with self._tracer.start_as_current_span(
            name, attributes=attributes
        ) as otlp_span:
            yield _OtlpSpan(otlp_span)


class _OtlpSpan(AbstractContextManager):
    """Adapter mapping ``record`` onto OTel ``set_attribute``."""

    def __init__(self, span) -> None:
        self._span = span

    def record(self, key: str, value: Any) -> None:
        try:
            self._span.set_attribute(key, value)
        except Exception as exc:  # never let observability break the run
            logger.debug("Failed to record %s=%s: %s", key, value, exc)

    def __enter__(self) -> "_OtlpSpan":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class LangfuseTraceExporter(TraceExporter):
    """Langfuse backend (cloud or self-hosted)."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        try:
            from langfuse import Langfuse
        except ImportError:
            raise ImportError(
                "langfuse package required. Install: pip install langfuse"
            )
        self._client = Langfuse(
            public_key=public_key, secret_key=secret_key, host=host,
        )

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        trace = self._client.trace(name=name, input=attributes or {})
        span = trace.span(name=name, input=attributes or {})
        try:
            yield _LangfuseSpan(span)
        except Exception as exc:
            span.end(output={"error": str(exc)})
            raise
        else:
            span.end(output=span._input)  # keep attributes as output summary
        finally:
            self._client.flush()


class _LangfuseSpan(AbstractContextManager):
    """Adapter mapping ``record`` onto Langfuse span ``update``."""

    def __init__(self, span) -> None:
        self._span = span
        self._attrs: Dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self._attrs[key] = value
        try:
            self._span.update(input=self._attrs)
        except Exception as exc:
            logger.debug("Failed to update langfuse span: %s", exc)

    def __enter__(self) -> "_LangfuseSpan":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def make_trace_exporter(
    kind: Optional[str] = None, **kwargs
) -> TraceExporter:
    """Build an exporter from *kind* or ``RTSA_TRACE_EXPORTER``.

    Accepted kinds: ``"otlp"``, ``"langfuse"``, ``"none"`` (default).
    Missing backends and unknown kinds degrade to ``NullTraceExporter``
    with a warning — experiments never crash because of observability.
    """
    kind = kind or os.environ.get(_ENV_VAR, "none")
    if kind == "none":
        return NullTraceExporter()
    try:
        if kind == "otlp":
            return OtlpTraceExporter(**kwargs)
        if kind == "langfuse":
            return LangfuseTraceExporter(**kwargs)
    except ImportError as exc:
        logger.warning(
            "Trace exporter '%s' unavailable (%s) — using no-op", kind, exc
        )
        return NullTraceExporter()
    logger.warning("Unknown trace exporter '%s' — using no-op", kind)
    return NullTraceExporter()

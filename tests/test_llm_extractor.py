"""Tests for LLM extractor with mock client (no API key needed)."""

import pytest
from rtsa.extractors.llm_extractor import (
    LLMClient, LLMExtractor, MockLLMClient,
    create_mock_extractor, EXTRACTION_PROMPT,
)
from rtsa.core.types import ReasoningTraceGraph


class TestMockLLMClient:
    def test_returns_fixed_response(self):
        client = MockLLMClient()
        resp = client.complete("some prompt")
        assert '"Retrieve"' in resp
        assert '"Transform"' in resp
        assert '"Verify"' in resp

    def test_custom_response(self):
        custom = '{"nodes": [{"id": 1, "type": "Branch"}], "edges": [], "domain": "logic"}'
        client = MockLLMClient(response=custom)
        assert client.complete() == custom


class TestLLMExtractorWithMock:
    def test_extract_returns_graph(self):
        ext = create_mock_extractor()
        cot = "According to theorem, compute x. Verify result."
        graph = ext.extract(cot, trace_id="test1")
        assert isinstance(graph, ReasoningTraceGraph)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.trace_id == "test1"
        assert graph.extractor == "mock-llm"

    def test_extract_node_types_from_mock(self):
        ext = create_mock_extractor()
        graph = ext.extract("ignored")
        types = [n.type.value for n in graph.nodes]
        assert types == ["Retrieve", "Transform", "Verify"]

    def test_extract_edges_from_mock(self):
        ext = create_mock_extractor()
        graph = ext.extract("ignored")
        assert set(graph.edges) == {(1, 2), (2, 3)}

    def test_extract_domain_from_mock(self):
        ext = create_mock_extractor()
        graph = ext.extract("ignored")
        assert graph.domain == "math"

    def test_extract_empty_response(self):
        ext = create_mock_extractor(response='{"nodes": [], "edges": []}')
        graph = ext.extract("empty")
        assert len(graph.nodes) == 0
        assert graph.metadata.get("extraction_rate") == 0.0

    def test_extract_retry_on_parse_failure(self):
        """Invalid JSON on first try, valid on second (but mock is deterministic)."""
        ext = create_mock_extractor(response="not json")
        graph = ext.extract("bad", trace_id="parse_fail")
        assert len(graph.nodes) == 0
        assert graph.metadata.get("extraction_rate") == 0.0
        assert "error" in graph.metadata

    def test_custom_response_parse(self):
        custom = '{"nodes": [{"id": 1, "type": "Branch"}, {"id": 2, "type": "Transform"}, {"id": 3, "type": "Transform"}], "edges": [[1, 2], [1, 3]], "domain": "logic"}'
        ext = create_mock_extractor(response=custom)
        graph = ext.extract("if x > 0 then y = 1 else y = -1")
        assert len(graph.nodes) == 3
        assert graph.domain == "logic"

    def test_prompt_template(self):
        cot = "Test CoT text here."
        prompt = EXTRACTION_PROMPT.format(cot_text=cot)
        assert cot in prompt
        assert "Reasoning Trace Graph" in prompt or "reasoning trace" in prompt.lower()


class TestLLMClient:
    def test_init_defaults(self):
        client = LLMClient()
        assert client.provider == "openai"
        assert client.model == "gpt-4"
        assert client.temperature == 0.0
        assert client.max_tokens == 2048

    def test_invalid_provider_raises_on_complete(self):
        client = LLMClient(provider="unknown")
        with pytest.raises(ValueError, match="Unknown provider"):
            client.complete("prompt")


class TestLLMExtractorInit:
    def test_create_e4_factory(self):
        from rtsa.extractors.llm_extractor import create_extractor_e4
        ext = create_extractor_e4(api_key="sk-test")
        assert ext.name == "gpt-4"
        assert ext.client.provider == "openai"
        assert ext.client.model == "gpt-4"

    def test_create_e5_factory(self):
        from rtsa.extractors.llm_extractor import create_extractor_e5
        ext = create_extractor_e5(api_key="sk-test")
        assert ext.name == "claude-3.5"
        assert ext.client.provider == "anthropic"

    def test_create_e6_factory(self):
        from rtsa.extractors.llm_extractor import create_extractor_e6
        ext = create_extractor_e6()
        assert ext.name == "qwen-2.5"
        assert ext.client.provider == "local"

    def test_create_mock_factory(self):
        ext = create_mock_extractor()
        assert ext.name == "mock-llm"
        assert isinstance(ext.client, MockLLMClient)

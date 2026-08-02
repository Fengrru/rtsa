"""
LLM-Based Extractor — Extractors E4, E5, E6, E7.

Uses LLM prompting to extract RTG from CoT text.
Supports OpenAI, Anthropic, DeepSeek, and local models via transformers.

E4: GPT-4 Extractor (Low independence — shared LLM bias)
E5: Claude-3.5 Extractor (Low independence — shared LLM bias)
E6: Qwen-2.5 Extractor (Medium independence — different training mix)
E7: DeepSeek Extractor (Medium independence — different training mix)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Template (Section 3.5)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_V2 = """You are a reasoning trace analyzer. Your job is to decompose a Chain-of-Thought reasoning trace into a fine-grained graph of atomic reasoning operations.

Each sentence or clause in the trace should typically become its OWN node. Split multi-step reasoning into separate nodes.

Node types:
- Retrieve: Recalling a fact, formula, theorem, or definition from memory
- Transform: Performing a calculation, simplification, substitution, or algebraic manipulation
- Compare: Comparing two values, approaches, or methods
- Verify: Checking consistency, confirming a result, testing a condition
- Branch: Conditional split, case exploration ("if", "when", "Case 1:", "suppose")
- Backtrack: Self-correction, error detection, changing approach ("Wait", "Actually", "Let me rethink")

Rules:
1. Split each distinct reasoning step into its OWN node. Do NOT merge steps.
   Example input: "Calculate 5 * 4 = 20. Then multiply 20 * 5.5 = 110."
   Correct: [Transform(id=1), Transform(id=2)] with edge 1→2

2. The SAME type can repeat in sequence if they are separate steps.

3. For a simple calculation like "6 * 2 = 12", it's ONE Transform node.
   But if the trace says "6 * 2 = 12. Then 12 * 3 = 36.", that's TWO Transform nodes connected sequentially.

4. Recognize the reasoning pattern of the trace:
   - "So", "thus", "therefore" → result step, usually Transform
   - "Recall", "remember", "by the theorem" → Retrieve
   - "Check", "verify", "does this satisfy" → Verify
   - "Wait", "actually", "let me re-" → Backtrack
   - "If", "suppose", "consider the case" → Branch
   - "Compare", "which is larger" → Compare

Output JSON format:
{{"nodes": [{{"id": 1, "type": "Transform"}}, {{"id": 2, "type": "Verify"}}], "edges": [[1, 2]], "domain": "math"}}

Only output valid JSON. No explanation.

CoT: {cot_text}"""

# Backward-compatible alias used by tests and legacy integrations
EXTRACTION_PROMPT = EXTRACTION_PROMPT_V2


# ---------------------------------------------------------------------------
# LLM Client Abstraction
# ---------------------------------------------------------------------------

class LLMClient:
    """Abstracted LLM client supporting multiple backends."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install: pip install anthropic")
        elif self.provider == "deepseek":
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com/v1",
                )
            except ImportError:
                raise ImportError("openai package required for DeepSeek. Install: pip install openai")
        elif self.provider == "local":
            self._client = None  # Will use transformers directly
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and return completion text."""
        client = self._get_client()

        if self.provider in ("openai", "deepseek") and client is not None:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "anthropic" and client is not None:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""

        elif self.provider == "local":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                if not hasattr(self, "_local_model"):
                    self._local_tokenizer = AutoTokenizer.from_pretrained(self.model)
                    self._local_model = AutoModelForCausalLM.from_pretrained(
                        self.model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    )

                inputs = self._local_tokenizer(prompt, return_tensors="pt").to(
                    self._local_model.device
                )
                outputs = self._local_model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
                return self._local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except ImportError:
                raise ImportError("transformers and torch required. Install: pip install transformers torch")

        raise RuntimeError(f"Failed to get completion from provider: {self.provider}")


# ---------------------------------------------------------------------------
# LLM Extractor
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    LLM-powered reasoning trace graph extractor.

    Uses the standardized prompt template (Section 3.5) to extract
    RTG JSON from CoT text via LLM API calls.

    E4: provider="openai", model="gpt-4"
    E5: provider="anthropic", model="claude-3-5-sonnet-20241022"
    E6: provider="local", model="Qwen/Qwen2.5-7B-Instruct"
    """

    def __init__(
        self,
        name: str,
        client: LLMClient,
        max_retries: int = 3,
    ):
        self.name = name
        self.client = client
        self.max_retries = max_retries

    def _parse_response(self, response: str) -> Tuple[List[Dict], List[Tuple[int, int]], str]:
        """
        Parse LLM JSON response into nodes and edges.
        Returns (nodes_list, edges_list, domain).
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match is None:
            raise ValueError(f"No JSON found in response: {response[:200]}...")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        nodes_raw = data.get("nodes", [])
        edges_raw = data.get("edges", [])
        domain = data.get("domain", "")

        return nodes_raw, edges_raw, domain

    def extract(self, cot_text: str, trace_id: str = "", **metadata) -> ReasoningTraceGraph:
        """
        Extract RTG via LLM prompting.

        Retries on parse failure up to max_retries times.
        """
        prompt = EXTRACTION_PROMPT_V2.format(cot_text=cot_text)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.complete(prompt)
                nodes_raw, edges_raw, domain = self._parse_response(response)

                # Build nodes
                nodes = []
                for n in nodes_raw:
                    nid = int(n.get("id", len(nodes) + 1))
                    ntype_str = n.get("type", "Transform")
                    try:
                        ntype = NodeType.from_string(ntype_str)
                    except ValueError:
                        ntype = NodeType.TRANSFORM  # fallback
                    span = n.get("span", [0, 0])
                    if isinstance(span, list) and len(span) == 2:
                        span_tuple = (int(span[0]), int(span[1]))
                    else:
                        span_tuple = (0, 0)
                    nodes.append(GraphNode(id=nid, type=ntype, span=span_tuple))

                # Build edges
                edges = []
                for e in edges_raw:
                    if isinstance(e, list) and len(e) == 2:
                        edges.append((int(e[0]), int(e[1])))

                if not nodes:
                    raise ValueError("No nodes extracted from LLM response")

                return ReasoningTraceGraph(
                    trace_id=trace_id or f"{self.name}_{hashlib.md5(cot_text.encode('utf-8')).hexdigest()[:12]}",
                    model=self.client.model,
                    extractor=self.name,
                    domain=domain,
                    nodes=nodes,
                    edges=edges,
                    metadata={
                        "cot_length_tokens": len(cot_text.split()),
                        "extraction_rate": 1.0 if nodes else 0.0,
                        "llm_provider": self.client.provider,
                        "llm_model": self.client.model,
                        **metadata,
                    },
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM extraction attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )

        # All retries exhausted
        logger.error(f"LLM extraction failed after {self.max_retries} attempts")
        return ReasoningTraceGraph(
            trace_id=trace_id or f"{self.name}_failed_{hashlib.md5(cot_text.encode('utf-8')).hexdigest()[:12]}",
            extractor=self.name,
            nodes=[],
            edges=[],
            metadata={
                "extraction_rate": 0.0,
                "error": str(last_error),
                **metadata,
            },
        )


# ---------------------------------------------------------------------------
# Factory for creating standard extractor configurations
# ---------------------------------------------------------------------------

def create_extractor_e4(api_key: Optional[str] = None) -> LLMExtractor:
    """Create GPT-4 extractor (E4)."""
    return LLMExtractor(
        name="gpt-4",
        client=LLMClient(provider="openai", model="gpt-4", api_key=api_key),
    )


def create_extractor_e5(api_key: Optional[str] = None) -> LLMExtractor:
    """Create Claude-3.5 extractor (E5)."""
    return LLMExtractor(
        name="claude-3.5",
        client=LLMClient(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key=api_key),
    )


def create_extractor_e6(model_path: str = "Qwen/Qwen2.5-7B-Instruct") -> LLMExtractor:
    """Create Qwen-2.5 local extractor (E6)."""
    return LLMExtractor(
        name="qwen-2.5",
        client=LLMClient(provider="local", model=model_path),
    )


def create_extractor_e7(api_key: Optional[str] = None, model: str = "deepseek-chat") -> LLMExtractor:
    """Create DeepSeek extractor (E7). Uses OpenAI-compatible API via deepseek.com."""
    return LLMExtractor(
        name="deepseek",
        client=LLMClient(provider="deepseek", model=model, api_key=api_key),
    )


def create_extractor_deepseek(api_key: Optional[str] = None, model: str = "deepseek-chat") -> LLMExtractor:
    """Create DeepSeek extractor with the v2 prompt (finer-grained extraction).
    
    Auto-detects DEEPSEEK_API_KEY from environment if not provided.
    """
    import os
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return LLMExtractor(
        name="deepseek-v2",
        client=LLMClient(provider="deepseek", model=model, api_key=api_key),
    )


# ---------------------------------------------------------------------------
# Mock LLM Client for offline testing
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = """{
    "nodes": [
        {"id": 1, "type": "Retrieve"},
        {"id": 2, "type": "Transform"},
        {"id": 3, "type": "Verify"}
    ],
    "edges": [[1, 2], [2, 3]],
    "domain": "math"
}"""


class MockLLMClient:
    """Returns fixed JSON response for offline testing (no API key needed)."""

    def __init__(self, response: str = _MOCK_RESPONSE, model: str = "mock", provider: str = "mock"):
        self.response = response
        self.model = model
        self.provider = provider

    def complete(self, prompt: str = "") -> str:
        return self.response


def create_mock_extractor(response: Optional[str] = None) -> LLMExtractor:
    """Create an LLMExtractor backed by MockLLMClient for testing."""
    return LLMExtractor(
        name="mock-llm",
        client=MockLLMClient(response or _MOCK_RESPONSE),
    )

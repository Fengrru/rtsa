"""
Core type definitions for RTSA (v3.2 merged).

Defines the foundational data structures: node types, graph schema,
operation space, validation rules, motif catalog, and canonical JSON Schema.
All other modules depend on these.

Merged from v3.1 + v3.2:
- NodeType enum with six primitive operations
- GraphNode + ReasoningTraceGraph (Pydantic v2)
- TraceMetadata dataclass for trace-level metadata
- MotifEntry + MOTIF_CATALOG + MOTIF_LOOKUP
- CANONICAL_JSON_SCHEMA for standard interchange (from v3.1)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from pydantic import BaseModel, Field, field_validator


# Schema version
SCHEMA_VERSION = "3.2.0"


# ---------------------------------------------------------------------------
# Primitive Operation Space (A)
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """
    The six primitive atomic reasoning operations.

    These are the ONLY valid node types in an RTG. Domain-specific tags
    (e.g., Math, Code) are graph-level metadata, NOT node attributes.
    """
    RETRIEVE = "Retrieve"
    TRANSFORM = "Transform"
    COMPARE = "Compare"
    VERIFY = "Verify"
    BRANCH = "Branch"
    BACKTRACK = "Backtrack"

    @classmethod
    def valid_set(cls) -> set:
        return {t.value for t in cls}

    @classmethod
    def from_string(cls, s: str) -> "NodeType":
        """Normalize case-insensitive string to NodeType."""
        mapping = {t.value.lower(): t for t in cls}
        result = mapping.get(s.strip().lower())
        if result is None:
            raise ValueError(
                f"Unknown node type '{s}'. Must be one of {sorted(cls.valid_set())}"
            )
        return result


# ---------------------------------------------------------------------------
# Graph Node
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    """A single node in a Reasoning Trace Graph (RTG)."""
    id: int = Field(..., ge=1)
    type: NodeType
    span: Tuple[int, int] = Field(default=(0, 0), description="[start_idx, end_idx) in source text")
    text: str = Field(default="", description="Text content of this node")

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v[0] < 0 or v[1] < 0:
            raise ValueError(f"Span indices must be non-negative, got {v}")
        if v[0] > v[1] and v != (0, 0):
            raise ValueError(f"Span start ({v[0]}) must be <= end ({v[1]})")
        return v

    def model_dump_simple(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type.value, "span": list(self.span)}


# ---------------------------------------------------------------------------
# Reasoning Trace Graph (RTG)
# ---------------------------------------------------------------------------

class ReasoningTraceGraph(BaseModel):
    """
    A complete Reasoning Trace Graph: DAG of atomic reasoning operations.

    Schema v3.2 - compliant with the JSON Schema from the implementation manual.
    """
    trace_id: str
    model: str = ""
    question_id: str = ""
    domain: str = ""  # e.g., "math", "code", "logic"
    extractor: str = ""  # e.g., "rbe", "sbe", "gpt-4"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    nodes: List[GraphNode]
    edges: List[Tuple[int, int]]

    @field_validator("edges")
    @classmethod
    def edges_valid(cls, v: List[Tuple[int, int]], info) -> List[Tuple[int, int]]:
        """Validate edges reference existing nodes."""
        data = info.data
        if "nodes" in data:
            node_ids = {n.id for n in data["nodes"]}
            for u, w in v:
                if u not in node_ids:
                    raise ValueError(f"Edge source {u} not in node set {node_ids}")
                if w not in node_ids:
                    raise ValueError(f"Edge target {w} not in node set {node_ids}")
        return v

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for topological analysis."""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id, type=node.type.value, span=node.span)
        G.add_edges_from(self.edges)
        return G

    def validate_dag(self) -> bool:
        """Check that the graph is a Directed Acyclic Graph."""
        G = self.to_networkx()
        return nx.is_directed_acyclic_graph(G)

    def validate_no_isolates(self) -> bool:
        """Check no isolated nodes (except single-node graphs)."""
        if len(self.nodes) <= 1:
            return True
        G = self.to_networkx()
        return len(list(nx.isolates(G))) == 0

    def validate_types(self) -> bool:
        valid = NodeType.valid_set()
        return all(n.type.value in valid for n in self.nodes)

    def is_valid(self) -> Tuple[bool, List[str]]:
        """Run all validation checks. Returns (is_valid, list_of_errors)."""
        errors = []
        if not self.validate_dag():
            errors.append("Graph contains cycles (must be a DAG)")
        if not self.validate_no_isolates():
            errors.append("Graph contains isolated nodes")
        if not self.validate_types():
            errors.append("Graph contains invalid node types")
        if len(self.nodes) == 0:
            errors.append("Graph has no nodes")
        return len(errors) == 0, errors

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Export in the standard JSON Schema format (Section 4.1)."""
        return {
            "trace_id": self.trace_id,
            "model": self.model,
            "question_id": self.question_id,
            "domain": self.domain,
            "extractor": self.extractor,
            "metadata": self.metadata,
            "graph": {
                "nodes": [n.model_dump_simple() for n in self.nodes],
                "edges": [list(e) for e in self.edges],
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to canonical JSON string."""
        return json.dumps(self.to_canonical_dict(), ensure_ascii=False, indent=indent)

    def to_jsonl_line(self) -> str:
        """Serialize as a single JSONL line."""
        return json.dumps(self.to_canonical_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: Union[str, Dict[str, Any]]) -> "ReasoningTraceGraph":
        """Deserialize from JSON string or dict."""
        if isinstance(data, str):
            data = json.loads(data)
        gd = data.get("graph", data)
        nodes = [
            GraphNode(
                id=int(n["id"]),
                type=NodeType.from_string(n["type"]),
                span=tuple(n["span"]) if n.get("span") is not None else (0, 0),
            )
            for n in gd.get("nodes", [])
        ]
        edges = [tuple(e) for e in gd.get("edges", [])]
        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            model=data.get("model", ""),
            question_id=data.get("question_id", ""),
            domain=data.get("domain", ""),
            extractor=data.get("extractor", ""),
            metadata=data.get("metadata", {}),
            nodes=nodes,
            edges=edges,
        )

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> List["ReasoningTraceGraph"]:
        """Load a JSONL file containing multiple RTGs."""
        schemas = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    schemas.append(cls.from_json(line))
        return schemas

    @classmethod
    def from_networkx(
        cls,
        G: nx.DiGraph,
        trace_id: str = "",
        model: str = "",
        question_id: str = "",
        domain: str = "",
        extractor: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ReasoningTraceGraph":
        """Build a ReasoningTraceGraph from a networkx.DiGraph."""
        nodes = []
        for nid, attrs in G.nodes(data=True):
            ntype = NodeType.from_string(attrs.get("type", "Transform"))
            span = attrs.get("span", (0, 0))
            nodes.append(GraphNode(id=int(nid), type=ntype, span=span))
        edges = [(int(u), int(v)) for u, v in G.edges()]
        return cls(
            trace_id=trace_id or str(uuid.uuid4()),
            model=model,
            question_id=question_id,
            domain=domain,
            extractor=extractor,
            metadata=metadata or {},
            nodes=nodes,
            edges=edges,
        )


# ---------------------------------------------------------------------------
# TraceMetadata (from v3.1)
# ---------------------------------------------------------------------------

@dataclass
class TraceMetadata:
    """Metadata for a reasoning trace (stored as graph-level attributes)."""
    cot_length_tokens: int = 0
    cot_length_chars: Optional[int] = None
    answer_correct: Optional[bool] = None
    domain: Optional[str] = None
    question_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extraction_rate: float = 1.0


# ---------------------------------------------------------------------------
# Motif Catalog Entry
# ---------------------------------------------------------------------------

class MotifEntry(BaseModel):
    """A discovered or predefined motif pattern."""
    motif_id: str
    pattern_name: str = ""
    description: str = ""
    size: int  # number of nodes
    node_types: List[NodeType]  # ordered list of types
    edge_list: List[Tuple[int, int]]  # edges as (from_idx, to_idx), 0-indexed
    frequency: float = 0.0  # fraction of graphs containing this motif
    discovery_method: str = "preset"  # "preset" | "gspan" | "exhaustive"


# ---------------------------------------------------------------------------
# Experiment Phase Constants
# ---------------------------------------------------------------------------

MOTIF_CATALOG: List[MotifEntry] = [
    MotifEntry(
        motif_id="M1", pattern_name="Chain(3)",
        description="A -> B -> C",
        size=3, node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY],
        edge_list=[(0, 1), (1, 2)],
    ),
    MotifEntry(
        motif_id="M2", pattern_name="Fork(3)",
        description="A -> B, A -> C (general fork: one source, two targets)",
        size=3, node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY],
        edge_list=[(0, 1), (0, 2)],
    ),
    MotifEntry(
        motif_id="M3", pattern_name="Diamond(4)",
        description="A -> B -> D, A -> C -> D",
        size=4, node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.COMPARE, NodeType.VERIFY],
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
    ),
    MotifEntry(
        motif_id="M4", pattern_name="Loop(3)",
        description="A -> B -> C -> A (detected but rejected for DAGs)",
        size=3, node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.BACKTRACK],
        edge_list=[(0, 1), (1, 2), (2, 0)],
    ),
    MotifEntry(
        motif_id="M5", pattern_name="Verify-After-Transform",
        description="Transform -> Verify",
        size=2, node_types=[NodeType.TRANSFORM, NodeType.VERIFY],
        edge_list=[(0, 1)],
    ),
    MotifEntry(
        motif_id="M6", pattern_name="Backtrack-Recover",
        description="Backtrack -> Retrieve/Transform",
        size=2, node_types=[NodeType.BACKTRACK, NodeType.RETRIEVE],
        edge_list=[(0, 1)],
    ),
    MotifEntry(
        motif_id="M7", pattern_name="Branch-Explore",
        description="Branch -> Transform, Branch -> Transform",
        size=3, node_types=[NodeType.BRANCH, NodeType.TRANSFORM, NodeType.TRANSFORM],
        edge_list=[(0, 1), (0, 2)],
    ),
    MotifEntry(
        motif_id="M8", pattern_name="Multi-Verify",
        description="Verify <- multiple parents",
        size=3, node_types=[NodeType.TRANSFORM, NodeType.TRANSFORM, NodeType.VERIFY],
        edge_list=[(0, 2), (1, 2)],
    ),
    MotifEntry(
        motif_id="M9", pattern_name="Branch-Transform",
        description="Branch -> Transform",
        size=2, node_types=[NodeType.BRANCH, NodeType.TRANSFORM],
        edge_list=[(0, 1)],
    ),
    MotifEntry(
        motif_id="M10", pattern_name="Transform-Transform",
        description="Transform -> Transform",
        size=2, node_types=[NodeType.TRANSFORM, NodeType.TRANSFORM],
        edge_list=[(0, 1)],
    ),
    MotifEntry(
        motif_id="M11", pattern_name="Retrieve-Transform",
        description="Retrieve -> Transform",
        size=2, node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM],
        edge_list=[(0, 1)],
    ),
    MotifEntry(
        motif_id="M12", pattern_name="Transform-Branch",
        description="Transform -> Branch",
        size=2, node_types=[NodeType.TRANSFORM, NodeType.BRANCH],
        edge_list=[(0, 1)],
    ),
]

# Precomputed motif_id -> MotifEntry mapping
MOTIF_LOOKUP: Dict[str, MotifEntry] = {m.motif_id: m for m in MOTIF_CATALOG}


# ---------------------------------------------------------------------------
# Canonical JSON Schema (from v3.1 — Section 4.1)
# ---------------------------------------------------------------------------

CANONICAL_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://fengrru.github.io/rtsa/v3.2/canonical.schema.json",
    "title": "RTSA Canonical Reasoning Trace Graph Schema",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": ["trace_id", "model", "question_id", "domain", "extractor", "metadata", "graph"],
    "properties": {
        "trace_id": {"type": "string"},
        "model": {"type": "string", "minLength": 1},
        "question_id": {"type": "string", "minLength": 1},
        "domain": {"type": "string", "enum": ["math", "code", "logic"]},
        "extractor": {
            "type": "string",
            "enum": ["rbe", "sbe", "rbe-rand", "gpt-4", "claude-3.5", "qwen-2.5", "human"],
        },
        "metadata": {
            "type": "object",
            "required": ["cot_length_tokens"],
            "properties": {
                "cot_length_tokens": {"type": "integer", "minimum": 0},
                "cot_length_chars": {"type": "integer", "minimum": 0},
                "answer_correct": {"type": "boolean"},
                "domain": {"type": "string"},
                "question_id": {"type": "string"},
                "model": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "extraction_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
        "graph": {
            "type": "object",
            "required": ["nodes", "edges"],
            "properties": {
                "nodes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "type"],
                        "properties": {
                            "id": {"type": "integer", "minimum": 1},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "Retrieve", "Transform", "Compare",
                                    "Verify", "Branch", "Backtrack",
                                ],
                            },
                            "span": {
                                "type": ["array", "null"],
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "integer", "minimum": 1},
                    },
                },
            },
        },
    },
}

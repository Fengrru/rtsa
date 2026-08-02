"""Tests for Rule-Based, Syntax-Based, and Random Baseline extractors."""

import pytest
from rtsa.core.types import NodeType, ReasoningTraceGraph
from rtsa.extractors.rule_based import RuleBasedExtractor, KEYWORD_RULES
from rtsa.extractors.syntax_based import SyntaxBasedExtractor
from rtsa.extractors.random_baseline import RandomBaselineExtractor, ShuffledTypeExtractor


# ─── RuleBasedExtractor ───────────────────────────────────────────────────────

class TestRuleBasedExtractor:
    @pytest.fixture
    def rbe(self):
        return RuleBasedExtractor()

    def test_classify_retrieve(self, rbe):
        assert rbe.classify_sentence("According to the Pythagorean theorem, a^2 + b^2 = c^2.") == NodeType.RETRIEVE

    def test_classify_transform(self, rbe):
        assert rbe.classify_sentence("Calculate the value of x given y = 5.") == NodeType.TRANSFORM

    def test_classify_compare(self, rbe):
        assert rbe.classify_sentence("We compare approach A which is larger than approach B.") == NodeType.COMPARE

    def test_classify_verify(self, rbe):
        assert rbe.classify_sentence("Let me check the result: 3^2 + 4^2 = 9 + 16 = 25.") == NodeType.VERIFY
        assert rbe.classify_sentence("Indeed, this is consistent with the theorem.") == NodeType.VERIFY

    def test_classify_branch(self, rbe):
        assert rbe.classify_sentence("If x > 0, then we proceed with the positive case.") == NodeType.BRANCH
        assert rbe.classify_sentence("Suppose for contradiction that n is composite.") == NodeType.BRANCH

    def test_classify_backtrack(self, rbe):
        assert rbe.classify_sentence("Wait, actually, I made a mistake in the previous step.") == NodeType.BACKTRACK
        assert rbe.classify_sentence("No, correction: the answer should be 5, not 7.") == NodeType.BACKTRACK

    def test_classify_fallback_to_transform(self, rbe):
        result = rbe.classify_sentence("The quick brown fox jumps over the lazy dog.")
        assert result == NodeType.TRANSFORM

    def test_classify_uncertainty_fallback(self, rbe):
        result = rbe.classify_sentence("I am not sure about this step?")
        assert result == NodeType.BACKTRACK

    def test_extract_returns_graph(self, rbe):
        cot = "According to Pythagoras, a^2 + b^2 = c^2. Calculate: 3^2 + 4^2 = 25. Check: c = 5."
        graph = rbe.extract(cot, trace_id="test1")
        assert isinstance(graph, ReasoningTraceGraph)
        assert graph.trace_id == "test1"
        assert graph.extractor == "rbe"
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_extract_empty_text(self, rbe):
        graph = rbe.extract("")
        assert len(graph.nodes) == 0

    def test_extract_all_node_types_present(self, rbe):
        cot = (
            "According to the theorem, x = 5. "
            "If x > 0, we compute y = x + 1. "
            "Otherwise, y = x - 1. "
            "Compare the two results. "
            "Wait, actually I need to verify. "
            "Check: both results are correct."
        )
        graph = rbe.extract(cot)
        types = {n.type for n in graph.nodes}
        assert len(types) >= 3  # should have at least 3 different types

    def test_merge_consecutive_transforms(self, rbe):
        types = [NodeType.TRANSFORM, NodeType.TRANSFORM, NodeType.VERIFY, NodeType.TRANSFORM, NodeType.TRANSFORM]
        merged = rbe._merge_consecutive_transforms(types)
        assert len(merged) == 3
        assert merged == [NodeType.TRANSFORM, NodeType.VERIFY, NodeType.TRANSFORM]

    def test_merge_no_consecutive(self, rbe):
        types = [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY]
        merged = rbe._merge_consecutive_transforms(types)
        assert merged == types

    def test_merge_empty(self, rbe):
        assert rbe._merge_consecutive_transforms([]) == []

    def test_split_sentences(self, rbe):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = rbe._split_sentences(text)
        assert len(sentences) >= 3

    def test_split_sentences_with_newlines(self, rbe):
        text = "Line one.\nLine two.\nLine three."
        sentences = rbe._split_sentences(text)
        assert len(sentences) >= 3

    def test_branch_node_has_extra_edges(self, rbe):
        cot = "If x > 0, then compute y. Otherwise compute z."
        graph = rbe.extract(cot)
        branch_nodes = [n for n in graph.nodes if n.type == NodeType.BRANCH]
        if branch_nodes:
            out_edges = [e for e in graph.edges if e[0] == branch_nodes[0].id]
            assert len(out_edges) >= 1

    def test_compiled_patterns(self, rbe):
        assert len(rbe._compiled) == 6
        assert all(isinstance(p, list) for _, p in rbe._compiled)


# ─── RandomBaselineExtractor ──────────────────────────────────────────────────

class TestRandomBaselineExtractor:
    @pytest.fixture
    def rbr(self):
        return RandomBaselineExtractor()

    def test_classify_by_length_short(self, rbr):
        assert rbr.classify_by_length("Short.") == NodeType.RETRIEVE
        assert rbr.classify_by_length("Five words total here.") == NodeType.RETRIEVE

    def test_classify_by_length_medium(self, rbr):
        assert rbr.classify_by_length("This is a medium length sentence with several words.") == NodeType.TRANSFORM

    def test_classify_by_length_long(self, rbr):
        long_sent = "This is a very long sentence that has many many many words in it for testing purposes."
        assert rbr.classify_by_length(long_sent) == NodeType.VERIFY

    def test_classify_by_length_very_long(self, rbr):
        very_long = " ".join(["word"] * 25)
        assert rbr.classify_by_length(very_long) == NodeType.BACKTRACK

    def test_extract_returns_graph(self, rbr):
        cot = "Short. A medium length sentence here. This is a much longer sentence with more words in it."
        graph = rbr.extract(cot)
        assert isinstance(graph, ReasoningTraceGraph)
        assert graph.extractor == "rbe_rand"
        assert graph.metadata.get("is_zero_information") is True

    def test_extract_empty(self, rbr):
        graph = rbr.extract("")
        assert len(graph.nodes) == 0

    def test_seed_reproducibility(self):
        rbr1 = RandomBaselineExtractor(seed=42)
        rbr2 = RandomBaselineExtractor(seed=42)
        cot = "Some text with a few sentences. Here is another sentence."
        g1 = rbr1.extract(cot)
        g2 = rbr2.extract(cot)
        types1 = [n.type for n in g1.nodes]
        types2 = [n.type for n in g2.nodes]
        assert types1 == types2


class TestRuleBasedExtractorChinese:
    """RBE with Chinese CoT text."""

    @pytest.fixture
    def rbe(self):
        return RuleBasedExtractor()

    def test_classify_cn_retrieve(self, rbe):
        assert rbe.classify_sentence("根据勾股定理，a平方加b平方等于c平方。") == NodeType.RETRIEVE

    def test_classify_cn_transform(self, rbe):
        assert rbe.classify_sentence("计算 x 加 3 等于 7。") == NodeType.TRANSFORM

    def test_classify_cn_verify(self, rbe):
        assert rbe.classify_sentence("验证结果是否正确。") == NodeType.VERIFY

    def test_classify_cn_branch(self, rbe):
        assert rbe.classify_sentence("如果 x 为负，则取绝对值。") == NodeType.BRANCH

    def test_classify_cn_backtrack(self, rbe):
        assert rbe.classify_sentence("等等，我搞错了。") == NodeType.BACKTRACK

    def test_classify_cn_compare(self, rbe):
        assert rbe.classify_sentence("比较两种方法的优劣。") == NodeType.COMPARE

    def test_extract_cn_chain(self, rbe):
        cot = "根据勾股定理，a平方加b平方等于c平方。代入a=3，b=4，计算c。验证c等于5是否正确。"
        graph = rbe.extract(cot)
        assert len(graph.nodes) >= 2
        types = [n.type for n in graph.nodes]
        assert NodeType.RETRIEVE in types
        assert NodeType.TRANSFORM in types or NodeType.VERIFY in types

    def test_extract_cn_mixed(self, rbe):
        """English + Chinese mixed CoT."""
        cot = "Recall the formula. 代入 a=1, b=-5, c=6。计算判别式。Verify the result."
        graph = rbe.extract(cot)
        assert len(graph.nodes) >= 2

    def test_split_sentences_cn(self, rbe):
        text = "第一句。第二句！第三句？第四句。"
        sentences = rbe._split_sentences(text)
        assert len(sentences) >= 4
        assert sentences[0] == "第一句。"

    def test_split_sentences_mixed(self, rbe):
        text = "First step. 第二步。 Third step."
        sentences = rbe._split_sentences(text)
        assert len(sentences) >= 3


class TestShuffledTypeExtractor:
    def test_preserves_structure(self):
        from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
        nodes = [
            GraphNode(id=1, type=NodeType.RETRIEVE),
            GraphNode(id=2, type=NodeType.TRANSFORM),
            GraphNode(id=3, type=NodeType.VERIFY),
        ]
        ref = ReasoningTraceGraph(trace_id="ref", nodes=nodes, edges=[(1, 2), (2, 3)])

        ste = ShuffledTypeExtractor(seed=42)
        shuffled = ste.extract(ref)
        assert len(shuffled.nodes) == 3
        assert len(shuffled.edges) == 2
        assert shuffled.extractor == "shuffled_type"

    def test_types_may_differ(self):
        from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
        nodes = [
            GraphNode(id=1, type=NodeType.RETRIEVE),
            GraphNode(id=2, type=NodeType.TRANSFORM),
            GraphNode(id=3, type=NodeType.VERIFY),
        ]
        ref = ReasoningTraceGraph(trace_id="ref", nodes=nodes, edges=[(1, 2), (2, 3)])

        ste = ShuffledTypeExtractor(seed=42)
        shuffled = ste.extract(ref)
        # Types might or might not differ; either is valid
        assert all(isinstance(n.type, NodeType) for n in shuffled.nodes)

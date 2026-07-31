"""Extractors module — all submodules unified export."""

from .rule_based import RuleBasedExtractor
from .syntax_based import SyntaxBasedExtractor
from .random_baseline import RandomBaselineExtractor, ShuffledTypeExtractor
from .llm_extractor import LLMClient, LLMExtractor, MockLLMClient, create_mock_extractor, create_extractor_e4, create_extractor_e5, create_extractor_e6, create_extractor_e7, create_extractor_deepseek
from .agreement import compute_full_iaa, detect_length_bias, detect_syntax_artifact, graph_level_iaa, motif_level_iaa, structure_level_iaa
from .gcp_validator import GCPValidator, GCPResult, GCSSentence, GCS_CORPUS_FULL, compute_gcs, make_gcp_adapter
from .inter_annotator import InterAnnotatorAgreement, ANNOTATION_GUIDELINES
from .synthetic_validation import SyntheticValidator, SyntheticValidationResult, SyntheticTrace
from .baselines import JPDirectedPreservingRandomizer, EdgeRewiringBaseline, PermutationBaseline, EnsembleBaseline, compute_stable_rate, compute_tsi_threshold
from .experiments import Phase05Pilot, PilotResult, ERCorrectedPredictor, ERCorrectedResult, PowerAnalysisResult, MODEL_REGISTRY, MODEL_FAMILIES, get_models_by_family, get_cross_family_models, power_analysis_logistic
from .analysis import CostEstimator, CostBreakdown, estimate_project_cost, kruskal_wallis_test, bootstrap_ci, cohens_d, partial_correlation

__all__ = [
    "RuleBasedExtractor", "SyntaxBasedExtractor", "RandomBaselineExtractor", "ShuffledTypeExtractor",
    "LLMClient", "LLMExtractor", "MockLLMClient", "create_mock_extractor",
    "create_extractor_e4", "create_extractor_e5", "create_extractor_e6", "create_extractor_e7", "create_extractor_deepseek",
    "compute_full_iaa", "detect_length_bias", "detect_syntax_artifact",
    "graph_level_iaa", "motif_level_iaa", "structure_level_iaa",
    "GCPValidator", "GCPResult", "GCSSentence", "GCS_CORPUS_FULL", "compute_gcs", "make_gcp_adapter",
    "InterAnnotatorAgreement", "ANNOTATION_GUIDELINES",
    "SyntheticValidator", "SyntheticValidationResult", "SyntheticTrace",
    "JPDirectedPreservingRandomizer", "EdgeRewiringBaseline", "PermutationBaseline", "EnsembleBaseline",
    "compute_stable_rate", "compute_tsi_threshold",
    "Phase05Pilot", "PilotResult", "ERCorrectedPredictor", "ERCorrectedResult", "PowerAnalysisResult",
    "MODEL_REGISTRY", "MODEL_FAMILIES", "get_models_by_family", "get_cross_family_models", "power_analysis_logistic",
    "CostEstimator", "CostBreakdown", "estimate_project_cost",
    "kruskal_wallis_test", "bootstrap_ci", "cohens_d", "partial_correlation",
]

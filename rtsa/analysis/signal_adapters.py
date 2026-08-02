"""
Signal adapters for external reasoning-quality projects.

Provides optional confidence / process-reward signals that *enhance*
RedundancyAnalyzer without adding new CLI commands.

If external libraries are installed, they are used; otherwise robust
textual and structural fallback heuristics are applied automatically.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class _CalibrationSignal(Protocol):
    def score(self, text: str, node_type: Optional[str] = None) -> float:
        ...


class _PRMSignal(Protocol):
    def score(self, text: str, prev_text: Optional[str] = None) -> float:
        ...


class FallbackCalibration:
    """Text-based heuristic mimicking metacognitive calibration.

    Lower confidence (-> higher redundancy risk) when:
    - hedge words dominate
    - length is extreme (too short / too long)
    - no concrete numbers or equations
    """

    _HEDGES = {
        "maybe",
        "perhaps",
        "possibly",
        "probably",
        "likely",
        "i think",
        "i believe",
        "it seems",
        "might",
        "could be",
        "unclear",
        "not sure",
        "guess",
        "assume",
        "suppose",
        "roughly",
        "approximately",
        "seems like",
        "appears to",
        "tentatively",
    }

    def score(self, text: str, node_type: Optional[str] = None) -> float:
        t = text.lower()
        hedge_hits = sum(1 for h in self._HEDGES if h in t)
        penalty = min(0.45, hedge_hits * 0.10)

        words = text.split()
        n = len(words)
        if n < 3:
            penalty += 0.20
        elif n > 60:
            penalty += 0.10

        bonus = 0.0
        if re.search(r"\d+", text):
            bonus += 0.08
        if any(op in text for op in ("=", "+", "-", "*", "/", "**", "^")):
            bonus += 0.06
        if node_type == "Verify" and re.search(r"correct|check|indeed|valid", t):
            bonus += 0.06

        return max(0.0, min(1.0, 1.0 - penalty + bonus))


class FallbackPRM:
    """Structural/textual heuristic mimicking a Process-Reward Model.

    Lower reward (-> higher redundancy risk) when:
    - step is almost pure repetition of previous step
    - starts with vague pronoun and lacks referent
    - contains no forward progress (no new info)
    """

    def score(self, text: str, prev_text: Optional[str] = None) -> float:
        reward = 0.72

        if prev_text:
            prev_words = set(prev_text.lower().split())
            curr_words = set(text.lower().split())
            if curr_words:
                novelty = len(curr_words - prev_words) / len(curr_words)
                if novelty < 0.15:
                    reward -= 0.22
                elif novelty > 0.60:
                    reward += 0.08

            first = text.split()[0].lower() if text.split() else ""
            if first in {"it", "this", "that", "they"}:
                if not any(
                    n in prev_text.lower()
                    for n in {"result", "value", "answer", "sum", "product"}
                ):
                    reward -= 0.12

        if len(text.split()) < 2:
            reward -= 0.15

        return max(0.0, min(1.0, reward))


def _try_import(path: str, name: str):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None


def make_calibration_adapter() -> _CalibrationSignal:
    cls = _try_import("metacognitive_calibration", "ConfidenceScorer")
    if cls is None:
        cls = _try_import("metacog_cal", "ConfidenceScorer")
    if cls is not None:
        logger.info("Loaded external metacognitive-calibration backend")
        return cls()
    return FallbackCalibration()


def make_prm_adapter() -> _PRMSignal:
    cls = _try_import("reasoning_navigation_engine", "ProcessRewardModel")
    if cls is None:
        cls = _try_import("rne", "ProcessRewardModel")
    if cls is not None:
        logger.info("Loaded external RNE/PRM backend")
        return cls()
    return FallbackPRM()

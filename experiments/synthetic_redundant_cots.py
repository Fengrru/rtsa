"""
Synthetic Redundant CoT Generator

Produces 50 diverse math-reasoning CoT traces with controlled redundancy
patterns for pruning utility experiments.

Each trace is a human-like chain-of-thought that intentionally contains
one or more redundancy motifs:
    - Excessive verification (Verify nodes repeated)
    - Dead branches (Branch/Backtrack with no productive children)
    - Long transform chains (single operation split into 3+ steps)
    - Circular retrieval (same theorem recalled multiple times)
    - Hedged backtracking ("Wait, maybe... No, it's fine")
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

# Module-level default RNG so helper functions stay deterministic when
# invoked standalone, WITHOUT seeding the global ``random`` module.
_DEFAULT_RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Base problems (clean, minimal reasoning)
# ---------------------------------------------------------------------------

BASE_PROBLEMS: List[Dict[str, str]] = [
    {"q": "What is 15 plus 27?", "a": "15 + 27 = 42"},
    {"q": "A rectangle has width 4 and length 7. What is its area?", "a": "4 * 7 = 28"},
    {"q": "Solve for x: 2x + 5 = 13.", "a": "x = 4"},
    {"q": "What is the average of 10, 20, and 30?", "a": "20"},
    {"q": "A car travels 60 miles per hour for 3 hours. How far?", "a": "180 miles"},
    {"q": "Compute 8 squared minus 6 squared.", "a": "28"},
    {"q": "If 3 apples cost $6, how much do 5 apples cost?", "a": "$10"},
    {"q": "What is 25 percent of 80?", "a": "20"},
    {"q": "Find the perimeter of a square with side 9.", "a": "36"},
    {"q": "Solve: x/4 = 7.", "a": "x = 28"},
]

# Templates for redundancy injection
VERIFY_TEMPLATES = [
    "Check: {expr} equals {result}. This is consistent.",
    "Verify: plugging {var} into the equation gives {result}. Correct.",
    "Double-check: {expr} is indeed {result}. Good.",
    "Confirm: {expr} = {result}. No errors found.",
]

DEAD_BRANCH_TEMPLATES = [
    "If we tried method A, we would get a different intermediate value. But method A is not needed here.",
    "Suppose the answer were {wrong}. Then {expr} would be {wrong_result}, which contradicts the given. So this path is discarded.",
    "One might consider a geometric approach, but algebraic methods are sufficient here.",
]

HEDGE_BACKTRACK_TEMPLATES = [
    "Wait, maybe I made an error. Let me reconsider... Actually no, the previous step is correct.",
    "I think the answer might be different. No, upon reflection, the calculation stands.",
    "Perhaps this is wrong. Re-examining... Yes, it is correct after all.",
]

CIRCULAR_RETRIEVAL_TEMPLATES = [
    "Recall the distributive property: a(b+c) = ab + ac.",
    "According to the definition of average: sum divided by count.",
    "By the Pythagorean theorem: a squared plus b squared equals c squared.",
]


def _build_clean_cot(problem: Dict[str, str], idx: int) -> str:
    """Minimal, non-redundant reasoning for a base problem."""
    q = problem["q"]
    a = problem["a"]
    # Simple 3-step pattern tailored to each problem style
    if "plus" in q or "minus" in q or "squared" in q:
        return (
            f"We need to compute {q.lower().replace('what is ', '').replace('compute ', '')}"
            f" Direct calculation gives {a}."
            f" Therefore, the answer is {a}."
        )
    if "area" in q or "perimeter" in q:
        return (
            f"Recall the formula for the requested geometric quantity."
            f" Substituting the given values yields {a}."
            f" Therefore, the answer is {a}."
        )
    if "average" in q or "percent" in q:
        return (
            f"Apply the standard formula."
            f" Substituting the numbers gives {a}."
            f" Therefore, the answer is {a}."
        )
    if "solve" in q.lower():
        return (
            f"Isolate the variable step by step."
            f" This simplifies to {a}."
            f" Therefore, the answer is {a}."
        )
    return (
        f"Set up the equation from the problem statement."
        f" Solving yields {a}."
        f" Therefore, the answer is {a}."
    )


def _inject_excessive_verify(
    cot: str, problem: Dict[str, str], rng: Optional[random.Random] = None,
) -> str:
    """Append 2-3 redundant Verify sentences."""
    rng = rng or _DEFAULT_RNG
    templates = rng.sample(VERIFY_TEMPLATES, k=rng.randint(2, 3))
    injected = " ".join(t.format(expr=problem["a"], result=problem["a"], var="x") for t in templates)
    return f"{cot} {injected}"


def _inject_dead_branch(
    cot: str, problem: Dict[str, str], rng: Optional[random.Random] = None,
) -> str:
    """Insert a dead-branch sentence in the middle."""
    rng = rng or _DEFAULT_RNG
    wrong = str(rng.randint(1, 100))
    tmpl = rng.choice(DEAD_BRANCH_TEMPLATES)
    branch = tmpl.format(wrong=wrong, expr=problem["a"], wrong_result=wrong)
    # Insert roughly in the middle
    parts = cot.split(" ")
    mid = len(parts) // 2
    parts.insert(mid, branch)
    return " ".join(parts)


def _inject_hedge_backtrack(
    cot: str, rng: Optional[random.Random] = None,
) -> str:
    """Insert a hedged backtrack sentence."""
    rng = rng or _DEFAULT_RNG
    tmpl = rng.choice(HEDGE_BACKTRACK_TEMPLATES)
    parts = cot.split(" ")
    mid = len(parts) // 2
    parts.insert(mid, tmpl)
    return " ".join(parts)


def _inject_long_transform_chain(
    cot: str, rng: Optional[random.Random] = None,
) -> str:
    """Replace a concise step with a drawn-out multi-step version."""
    filler = (
        "First, write down the given values. "
        "Next, identify the operation needed. "
        "Then, perform the arithmetic carefully. "
        "After that, simplify the intermediate result. "
        "Finally, obtain the final value."
    )
    # Replace "Direct calculation" or similar concise phrase
    if "Direct calculation" in cot:
        return cot.replace("Direct calculation", filler)
    if "Apply the standard formula" in cot:
        return cot.replace("Apply the standard formula", filler)
    # Fallback: prepend
    return f"{filler} {cot}"


def _inject_circular_retrieval(
    cot: str, rng: Optional[random.Random] = None,
) -> str:
    """Insert the same theorem/recall 2-3 times."""
    rng = rng or _DEFAULT_RNG
    tmpl = rng.choice(CIRCULAR_RETRIEVAL_TEMPLATES)
    repeats = " ".join([tmpl] * rng.randint(2, 3))
    return f"{repeats} {cot}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_cots(n: int = 50, seed: int = 42) -> List[Dict[str, str]]:
    """Generate *n* synthetic CoT traces with controlled redundancy.

    Returns list of dicts with keys: question_id, cot_text, answer, redundancy_types.
    """
    rng = random.Random(seed)
    traces: List[Dict[str, str]] = []

    redundancy_injectors = [
        ("ExcessiveVerify", _inject_excessive_verify),
        ("DeadBranch", _inject_dead_branch),
        ("HedgeBacktrack", _inject_hedge_backtrack),
        ("LongTransformChain", _inject_long_transform_chain),
        ("CircularRetrieve", _inject_circular_retrieval),
    ]

    for i in range(n):
        base = BASE_PROBLEMS[i % len(BASE_PROBLEMS)]
        clean = _build_clean_cot(base, i)

        # Each trace gets 1-3 redundancy types
        k = rng.randint(1, 3)
        selected = rng.sample(redundancy_injectors, k=k)
        types = []
        cot = clean
        for name, injector in selected:
            types.append(name)
            if name in ("DeadBranch", "ExcessiveVerify"):
                cot = injector(cot, base, rng=rng)
            else:
                cot = injector(cot, rng=rng)

        traces.append({
            "question_id": f"syn_redundant_{i:03d}",
            "cot_text": cot,
            "answer": base["a"],
            "redundancy_types": ",".join(types),
        })

    return traces


if __name__ == "__main__":
    cots = generate_synthetic_cots(10)
    for c in cots:
        print(f"{c['question_id']}  types={c['redundancy_types']}  words={len(c['cot_text'].split())}")

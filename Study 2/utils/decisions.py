# decisions.py

from typing import Dict, Set

DECISION_SPECS: Dict[str, Dict[str, object]] = {
    "binary": {
        "name": "BINARY",
        "allowed": {"include", "exclude"},  # exact only
        "doc": """Allowed outputs: include, exclude

INCLUDE: The abstract clearly meets at least one inclusion criterion and does not meet any exclusion criterion.
EXCLUDE: The abstract clearly fails the inclusion criteria or meets any exclusion criterion.

RATIONALE GUIDANCE:
Explain your decision.
"""
    },
    "ordinal": {
        "name": "ORDINAL",
        "allowed": {"yes", "maybe", "no"},  # exact only
        "doc": """Allowed outputs: yes, maybe, no

**YES**: Choose this when the abstract clearly meets the inclusion criteria and does NOT meet any exclusion criteria. The abstract provides sufficient information to confidently determine relevance.

**NO**: Choose this when the abstract clearly violates one or more exclusion criteria OR clearly fails to meet the inclusion criteria. The abstract provides sufficient information to confidently determine irrelevance.

**MAYBE**: Choose this when the abstract is potentially relevant but lacks sufficient detail to make a confident decision. Use this option when:
- The abstract mentions relevant concepts but lacks specific details about methodology, context, or scope
- Key information needed to apply the criteria is missing or ambiguous
- The study appears to be in the right domain but the connection to your research question is unclear
- You would need to see the full text to properly evaluate against the criteria

RATIONALE GUIDANCE:
Explain your decision. For MAYBE decisions, specifically describe what additional information from the full text would be needed to make a final determination.

"""
    },
    "likert": {
        "name": "LIKERT",
        "allowed": {"1", "2", "3", "4", "5"},  # exact only
        "doc": """Allowed outputs: 1, 2, 3, 4, 5 (integers)
**5 - Highly Relevant**: Abstract clearly and strongly meets all inclusion criteria. Definitely should be included.

**4 - Relevant**: Abstract meets most inclusion criteria with good alignment to the research focus. Should likely be included.

**3 - Moderately Relevant**: Abstract shows some relevance but has unclear aspects or partial alignment with criteria. Requires careful consideration.

**2 - Minimally Relevant**: Abstract has limited relevance, meets few criteria, or has significant gaps. Should likely be excluded.

**1 - Not Relevant**: Abstract clearly does not meet inclusion criteria or is completely outside the research scope. Definitely should be excluded.

Use whole numbers only.

RATIONALE GUIDANCE:
Explain your rating by specifically addressing how the abstract aligns with each inclusion/exclusion criterion. For scores of 3, clearly identify what information would be needed to make a definitive decision. For scores of 1-2, specify which criteria are not met. For scores of 4-5, highlight the strong alignment with criteria.
"""
    },
    "triage": {
        "name": "TRIAGE",
        "allowed": {"include", "maybe", "exclude"},  # exact only
        "doc": """Allowed outputs: include, maybe, exclude

include = clear match to inclusion; no exclusion applies.
maybe   = potentially relevant but abstract lacks key details to decide; flag for full-text review.
exclude = clear mismatch or explicit exclusion applies.
"""
    },
}

def decision_block(scheme: str) -> str:
    """Build the instruction block injected into the user prompt."""
    if scheme not in DECISION_SPECS:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose one of: {list(DECISION_SPECS.keys())}")
    spec = DECISION_SPECS[scheme]
    return (
        f"Expected Decision Type: {spec['name']}\n\n"
        f"{spec['doc']}\n"
        "Return format (strict):\n"
        "Reason: <1-3 sentences>\n"
        "Decision: <label or value>\n"
    )

def allowed_labels(scheme: str) -> Set[str]:
    """Convenience accessor used by the parser/validator."""
    if scheme not in DECISION_SPECS:
        raise ValueError(f"Unknown scheme '{scheme}'.")
    return set(DECISION_SPECS[scheme]["allowed"])  # type: ignore[return-value]

def validate_decision(value: str, scheme: str) -> bool:
    """Exact-match validation (no aliases)."""
    if not isinstance(value, str):
        return False
    alwd = allowed_labels(scheme)
    # keep case-insensitive check but require exact token membership
    if value.isdigit():
        return value in alwd
    return value.lower() in {a.lower() for a in alwd}

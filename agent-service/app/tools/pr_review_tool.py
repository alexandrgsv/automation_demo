"""Compatibility import for the PR review skill.

The skill lives in ``app.skills.pr_code_reviewer``.
"""

from app.skills.pr_code_reviewer.logic import (
    ANALYZE_TOOL,
    CYCLE_REDUCTION,
    REVIEW_TIME_SAVED_MINUTES,
    RISK_TOOL,
    STUB_TOOL,
    analyze_pr_diff,
    audit_code_changes,
    calculate_review_risk_score,
    generate_pytest_stub,
)

__all__ = [
    "ANALYZE_TOOL",
    "CYCLE_REDUCTION",
    "REVIEW_TIME_SAVED_MINUTES",
    "RISK_TOOL",
    "STUB_TOOL",
    "analyze_pr_diff",
    "audit_code_changes",
    "calculate_review_risk_score",
    "generate_pytest_stub",
]

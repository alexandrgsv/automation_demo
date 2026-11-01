"""Deterministic PR triage: diff checks, a pytest stub, and a merge risk.

The three tools scan text. They do not clone a repository and they do not
call a model. The same diff always produces the same audit.
"""

from typing import Any, Literal

from app.schemas import (
    DiffAnalysis,
    PRReviewInput,
    PRReviewResponse,
    PRReviewSummary,
    PytestStub,
    ReviewMetrics,
    RiskScore,
    RiskScoreInput,
    ToolInvocation,
)

ANALYZE_TOOL: str = "analyze_pr_diff"
STUB_TOOL: str = "generate_pytest_stub"
RISK_TOOL: str = "calculate_review_risk_score"

REVIEW_TIME_SAVED_MINUTES: int = 45
CYCLE_REDUCTION: str = "From 3-day backlog to 1-day turnaround"

_MISSING_ANNOTATION: str = "Missing return type annotations in modified functions"
_BROAD_EXCEPT: str = "Broad exception handling detected; specify exact exception types"
_UNRESOLVED_MARKER: str = "Unresolved TODO/FIXME markers in production diff"


def _module_stem(file_path: str) -> str:
    """Turn a file path into a test-name stem.

    Parameters
    ----------
    file_path : str
        Repository path. Only the final segment is used.

    Returns
    -------
    str
        Letters, digits, and underscores. A path with no usable characters
        becomes ``module``.
    """
    name: str = file_path.replace("\\", "/").rstrip("/").split("/")[-1]
    if name.endswith(".py"):
        name = name[: -len(".py")]
    cleaned: list[str] = [char if char.isalnum() or char == "_" else "_" for char in name]
    stem: str = "".join(cleaned).strip("_")
    return stem or "module"


def analyze_pr_diff(payload: PRReviewInput) -> DiffAnalysis:
    """Scan a diff for three fixed practice checks.

    The scan is case-insensitive and covers the whole text, not one function
    at a time. A single ``->`` anywhere clears the annotation check.

    Parameters
    ----------
    payload : PRReviewInput
        Pull request id, path, and diff text.

    Returns
    -------
    DiffAnalysis
        Findings in check order: annotations, bare or ``Exception`` handlers,
        then TODO or FIXME markers.
    """
    diff: str = payload.diff_content.lower()
    issues: list[str] = []
    if "def " in diff and "->" not in diff:
        issues.append(_MISSING_ANNOTATION)
    if "except:" in diff or "except exception" in diff:
        issues.append(_BROAD_EXCEPT)
    if "todo" in diff or "fixme" in diff:
        issues.append(_UNRESOLVED_MARKER)
    return DiffAnalysis(detected_issues=issues)


def generate_pytest_stub(payload: PRReviewInput) -> PytestStub:
    """Build two pytest names and a skeleton from the changed file name.

    Parameters
    ----------
    payload : PRReviewInput
        Pull request id, path, and diff text. Only ``file_path`` affects the names.

    Returns
    -------
    PytestStub
        ``test_<stem>_edge_cases``, ``test_<stem>_invalid_schema_handling``,
        and the source of both functions.
    """
    stem: str = _module_stem(payload.file_path)
    suggested_tests: list[str] = [
        f"test_{stem}_edge_cases",
        f"test_{stem}_invalid_schema_handling",
    ]
    blocks: list[str] = []
    for name in suggested_tests:
        blocks.append(
            f"def {name}() -> None:\n"
            f'    """Deterministic stub for {name}."""\n'
            f"    raise NotImplementedError\n"
        )
    return PytestStub(suggested_tests=suggested_tests, stub_source="\n".join(blocks))


def calculate_review_risk_score(payload: RiskScoreInput) -> RiskScore:
    """Map the issue count to a merge risk and a recommendation.

    Zero issues is Low and ``approve``. One issue is Medium. Two or more is
    High. Medium and High both recommend ``request_changes``.

    Parameters
    ----------
    payload : RiskScoreInput
        Findings from :func:`analyze_pr_diff`.

    Returns
    -------
    RiskScore
        Level, recommendation, and whether an automatic merge is allowed.
    """
    issue_count: int = len(payload.detected_issues)
    risk_level: Literal["Low", "Medium", "High"]
    if issue_count >= 2:
        risk_level = "High"
    elif issue_count == 1:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    can_auto_merge: bool = risk_level == "Low"
    recommendation: Literal["approve", "request_changes"] = (
        "approve" if can_auto_merge else "request_changes"
    )
    return RiskScore(
        risk_level=risk_level,
        recommendation=recommendation,
        can_auto_merge=can_auto_merge,
    )


def audit_code_changes(payload: PRReviewInput) -> PRReviewResponse:
    """Run the three review tools and assemble one response.

    Parameters
    ----------
    payload : PRReviewInput
        Validated diff from ``POST /api/v1/pr-review``.

    Returns
    -------
    PRReviewResponse
        Metrics, the combined summary, and the three tool traces.

    Raises
    ------
    RuntimeError
        If a tool returns a failure for an already validated payload.
    """
    # Imported here so this module can be registered without a circular import.
    from app.tools import dispatch_tool

    arguments: dict[str, Any] = payload.model_dump()
    analysis_call: ToolInvocation = dispatch_tool(ANALYZE_TOOL, arguments)
    stub_call: ToolInvocation = dispatch_tool(STUB_TOOL, arguments)
    issues: list[str] = []
    if analysis_call.ok and analysis_call.output is not None:
        issues = list(analysis_call.output["detected_issues"])
    risk_call: ToolInvocation = dispatch_tool(RISK_TOOL, {"detected_issues": issues})
    invocations: list[ToolInvocation] = [analysis_call, stub_call, risk_call]
    failed: ToolInvocation | None = next((item for item in invocations if not item.ok), None)
    if failed is not None or risk_call.output is None or stub_call.output is None:
        message: str = failed.error.message if failed is not None and failed.error is not None else "PR tool failed."
        raise RuntimeError(message)

    risk: RiskScore = RiskScore.model_validate(risk_call.output)
    summary: PRReviewSummary = PRReviewSummary(
        risk_level=risk.risk_level,
        detected_issues=issues,
        suggested_tests=list(stub_call.output["suggested_tests"]),
        can_auto_merge=risk.can_auto_merge,
        review_time_saved_minutes=REVIEW_TIME_SAVED_MINUTES,
        recommendation=risk.recommendation,
    )
    return PRReviewResponse(
        status="success",
        pr_id=payload.pr_id,
        mode="mock",
        metrics=ReviewMetrics(
            estimated_time_saved=f"{REVIEW_TIME_SAVED_MINUTES} min",
            cycle_reduction=CYCLE_REDUCTION,
        ),
        review=summary,
        tool_invocations=invocations,
    )

"""PR review skill: three tools and one audit entry point."""

from pydantic import BaseModel

from app.schemas import PRReviewResponse
from app.skills.base import BaseSkill, ToolBinding
from app.skills.pr_code_reviewer.logic import (
    ANALYZE_TOOL,
    RISK_TOOL,
    STUB_TOOL,
    analyze_pr_diff,
    audit_code_changes,
    calculate_review_risk_score,
    generate_pytest_stub,
)
from app.skills.pr_code_reviewer.schema import PRReviewInput, RiskScoreInput


class PRCodeReviewerSkill(BaseSkill):
    """Audit a diff, name two pytest stubs, and score merge risk.

    The three tools stay individually callable. :meth:`execute` runs them in
    order and returns the combined review.
    """

    name: str = "pr_code_reviewer"
    description: str = (
        "Audits a pull request diff for missing annotations, broad handlers, "
        "and TODO markers, then scores merge risk."
    )

    @property
    def tools(self) -> tuple[ToolBinding, ...]:
        """Return the analyze, stub, and risk tools.

        Returns
        -------
        tuple[ToolBinding, ...]
            The three review bindings, in pipeline order.
        """
        return (
            ToolBinding(
                name=ANALYZE_TOOL,
                description=(
                    "Scan a pull request diff for missing return annotations, "
                    "broad exception handlers, and TODO or FIXME markers."
                ),
                input_model=PRReviewInput,
                handler=analyze_pr_diff,
            ),
            ToolBinding(
                name=STUB_TOOL,
                description="Build a deterministic pytest skeleton from the changed file name.",
                input_model=PRReviewInput,
                handler=generate_pytest_stub,
            ),
            ToolBinding(
                name=RISK_TOOL,
                description=(
                    "Score merge risk from analyze_pr_diff issues. "
                    "Low approves. Medium and High request changes."
                ),
                input_model=RiskScoreInput,
                handler=calculate_review_risk_score,
            ),
        )

    def execute(self, params: BaseModel) -> PRReviewResponse:
        """Run the three review tools on one diff.

        Parameters
        ----------
        params : BaseModel
            A :class:`PRReviewInput`, or a model with the same fields.

        Returns
        -------
        PRReviewResponse
            Risk, findings, suggested tests, and the three tool traces.
        """
        parsed: PRReviewInput = PRReviewInput.model_validate(params, from_attributes=True)
        return audit_code_changes(parsed)


PR_CODE_REVIEWER_SKILL: PRCodeReviewerSkill = PRCodeReviewerSkill()

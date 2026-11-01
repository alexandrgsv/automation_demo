"""Argument and result contracts for the PR review skill."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PRReviewInput(BaseModel):
    """Diff submitted to the PR triage pipeline.

    Attributes
    ----------
    pr_id : int
        Pull request identifier. Zero and negative values are rejected.
    file_path : str
        Repository path of the changed file.
    diff_content : str
        Git diff text, or the source of the changed function.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "pr_id": 42,
                    "file_path": "app/agent.py",
                    "diff_content": (
                        "def run(data):\n"
                        "    try:\n"
                        "        return data\n"
                        "    except:\n"
                        "        pass\n"
                        "# TODO: narrow the handler\n"
                    ),
                }
            ]
        },
    )

    pr_id: int = Field(gt=0, description="Pull request identifier.", examples=[42])
    file_path: str = Field(
        min_length=1,
        max_length=512,
        description="Path of the changed file.",
        examples=["app/agent.py"],
    )
    diff_content: str = Field(
        min_length=1,
        max_length=20000,
        description="Git diff text, or the source of the changed function.",
    )


class DiffAnalysis(BaseModel):
    """Issues found by ``analyze_pr_diff``.

    Attributes
    ----------
    detected_issues : list[str]
        Stable messages, in check order. Empty when the diff passes.
    """

    detected_issues: list[str] = Field(
        default_factory=list,
        description="Practice findings, in check order. Empty when the diff passes.",
    )


class PytestStub(BaseModel):
    """Deterministic pytest skeleton for one changed file.

    Attributes
    ----------
    suggested_tests : list[str]
        Test function names derived from the file name.
    stub_source : str
        Python source of those tests. Each body raises ``NotImplementedError``.
    """

    suggested_tests: list[str] = Field(description="Test names derived from the file name.")
    stub_source: str = Field(description="Pytest source. Each test raises NotImplementedError.")


class RiskScoreInput(BaseModel):
    """Issues passed from the diff analysis into the risk score.

    Attributes
    ----------
    detected_issues : list[str]
        Messages produced by ``analyze_pr_diff``.
    """

    model_config = ConfigDict(extra="forbid")

    detected_issues: list[str] = Field(
        default_factory=list,
        description="Issues produced by analyze_pr_diff.",
    )


class RiskScore(BaseModel):
    """Merge risk and the review recommendation.

    Attributes
    ----------
    risk_level : {"Low", "Medium", "High"}
        Low for no issues, Medium for one, High for two or more.
    recommendation : {"approve", "request_changes"}
        ``approve`` only when the risk is Low.
    can_auto_merge : bool
        True only when the recommendation is ``approve``.
    """

    risk_level: Literal["Low", "Medium", "High"] = Field(description="Low, Medium, or High.")
    recommendation: Literal["approve", "request_changes"] = Field(
        description="approve when the risk is Low, otherwise request_changes."
    )
    can_auto_merge: bool = Field(description="True only when the recommendation is approve.")


class PRReviewSummary(BaseModel):
    """Combined audit returned by the PR triage pipeline.

    Attributes
    ----------
    risk_level : {"Low", "Medium", "High"}
        Score from ``calculate_review_risk_score``.
    detected_issues : list[str]
        Findings from ``analyze_pr_diff``.
    suggested_tests : list[str]
        Names from ``generate_pytest_stub``.
    can_auto_merge : bool
        True only for a Low risk.
    review_time_saved_minutes : int
        Fixed demo estimate of reviewer time removed by the pipeline.
    recommendation : {"approve", "request_changes"}
        Same decision as ``can_auto_merge``, spelled out for the review comment.
    """

    risk_level: Literal["Low", "Medium", "High"] = Field(description="Low, Medium, or High.")
    detected_issues: list[str] = Field(description="Findings from analyze_pr_diff.")
    suggested_tests: list[str] = Field(description="Test names from generate_pytest_stub.")
    can_auto_merge: bool = Field(description="True only when the risk is Low.")
    review_time_saved_minutes: int = Field(
        ge=0,
        description="Fixed estimate of reviewer minutes removed by this audit.",
    )
    recommendation: Literal["approve", "request_changes"] = Field(
        description="approve when the risk is Low, otherwise request_changes."
    )


class ReviewMetrics(BaseModel):
    """Headline numbers attached to a PR review response.

    Attributes
    ----------
    estimated_time_saved : str
        The saved-minutes figure, ready to read on a slide.
    cycle_reduction : str
        One sentence about the review cycle the demo claims.
    """

    estimated_time_saved: str = Field(description="Saved reviewer time, as a short label.")
    cycle_reduction: str = Field(description="Review-cycle claim for this demo.")

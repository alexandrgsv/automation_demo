"""Argument and result contracts for the developer-versus-QA skill."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TestExecutionInput(BaseModel):
    """Code pair executed by the isolated test runner.

    Attributes
    ----------
    source_code : str
        Implementation under test, usually from the developer step.
    test_code : str
        Assertions from the QA step. They run in the same namespace.
    """

    model_config = ConfigDict(extra="forbid")

    source_code: str = Field(
        min_length=1,
        max_length=20000,
        description="Implementation under test.",
    )
    test_code: str = Field(
        min_length=1,
        max_length=20000,
        description="Assertions that run after the implementation is loaded.",
    )


class TestExecutionResult(BaseModel):
    """Outcome of one sandbox run.

    Attributes
    ----------
    passed : bool
        True when the test script finishes without an exception.
    total_tests : int
        One when the test script was started, otherwise zero.
    failed_tests : int
        One when the run failed, otherwise zero.
    failure_details : list[str]
        Short failure lines. Empty when the run passed.
    """

    passed: bool = Field(description="True when the test script finished cleanly.")
    total_tests: int = Field(ge=0, description="How many test scripts were started.")
    failed_tests: int = Field(ge=0, description="How many of those scripts failed.")
    failure_details: list[str] = Field(
        default_factory=list,
        description="Short failure lines. Empty when the run passed.",
    )


class AdversarialCycleSummary(BaseModel):
    """Fixed developer-versus-QA loop result.

    Attributes
    ----------
    task : str
        Feature sentence supplied by the caller.
    iterations : int
        How many runner passes the loop performed.
    initial_flaw_detected : str
        Failure from the first pass, before the guard was added.
    final_status : {"Passed", "Failed"}
        Result of the second pass.
    generated_edge_cases : list[str]
        Boundary cases the QA step is aiming at.
    """

    task: str = Field(description="Feature sentence for this loop.")
    iterations: int = Field(ge=1, description="Runner passes in this loop.")
    initial_flaw_detected: str = Field(description="Failure text from the first pass.")
    final_status: Literal["Passed", "Failed"] = Field(description="Result of the hardened pass.")
    generated_edge_cases: list[str] = Field(description="Boundary cases named by the QA step.")


class LoopIteration(BaseModel):
    """One pass of the QA script against one implementation.

    Attributes
    ----------
    step : int
        1 is the naive implementation. 2 is the hardened one.
    role : {"qa_against_naive", "qa_against_hardened"}
        Which implementation this pass attacked.
    result : TestExecutionResult
        Runner output for this pass.
    """

    step: int = Field(ge=1, description="Pass number, starting at 1.")
    role: Literal["qa_against_naive", "qa_against_hardened"] = Field(
        description="Which implementation this pass attacked."
    )
    result: TestExecutionResult = Field(description="Runner output for this pass.")


class AdversarialReviewRequest(BaseModel):
    """Body of ``POST /api/v1/adversarial-review``.

    Attributes
    ----------
    task : str
        Sentence stored on the summary. It does not select the snippets.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"task": "Implement edge throughput calculation with zero-division safety"}
            ]
        },
    )

    task: str = Field(
        default="Implement edge throughput calculation with zero-division safety",
        min_length=1,
        max_length=2000,
        description="Feature sentence recorded on the summary. The snippets stay fixed.",
    )


class AdversarialReviewResponse(BaseModel):
    """Body returned by the developer-versus-QA loop.

    The developer and QA snippets are fixed. Only the runner executes them.
    ``mode`` stays ``mock`` because no chat provider writes the code.

    Attributes
    ----------
    status : {"completed"}
        Both passes finished.
    pattern : str
        Name of the loop, for the screen share.
    mode : {"mock"}
        Always mock.
    summary : AdversarialCycleSummary
        Flaw, final status, and the named edge cases.
    runs : list[LoopIteration]
        First the failing pass, then the hardened pass.
    """

    status: Literal["completed"] = Field(description="Both passes finished.")
    pattern: str = Field(description="Name of this feedback loop.")
    mode: Literal["mock"] = Field(description="The snippets are fixed. The runner is real.")
    summary: AdversarialCycleSummary = Field(description="Loop summary.")
    runs: list[LoopIteration] = Field(description="Naive pass, then hardened pass.")

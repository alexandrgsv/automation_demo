"""Scripted developer-versus-QA loop around the sandbox runner.

The developer step and the QA step are fixed snippets. The first snippet
divides by the duration with no guard. The QA script requires a ``ValueError``
when the duration is zero. The second snippet adds that guard, and the same
script is run again.
"""

from typing import Literal

from app.schemas import (
    AdversarialCycleSummary,
    AdversarialReviewResponse,
    LoopIteration,
    TestExecutionInput,
)
from app.tools.test_runner_tool import execute_in_sandbox

_PATTERN: str = "Two-Agent Adversarial Feedback Loop (Dev + Red Team QA)"

_NAIVE_CODE: str = """
def calculate_throughput(frame_count: int, duration_sec: float) -> float:
    return frame_count / duration_sec
"""

_HARDENED_CODE: str = """
def calculate_throughput(frame_count: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        raise ValueError("Duration must be strictly positive")
    if frame_count < 0:
        raise ValueError("Frame count cannot be negative")
    return frame_count / duration_sec
"""

_QA_TEST: str = """
assert calculate_throughput(100, 2.0) == 50.0
try:
    calculate_throughput(10, 0.0)
    assert False, "Should raise ValueError on duration <= 0"
except ValueError:
    pass
"""

_EDGE_CASES: list[str] = [
    "Zero duration handling (ZeroDivisionError)",
    "Negative timestamps / frame counts",
    "Float precision boundary",
]


def run_adversarial_dev_loop(task_description: str) -> AdversarialReviewResponse:
    """Run the naive implementation, then the hardened one, against one QA script.

    Parameters
    ----------
    task_description : str
        Feature sentence copied onto the summary. It does not change the snippets.

    Returns
    -------
    AdversarialReviewResponse
        The first failure, the second pass, and the named edge cases.
    """
    qa_input_naive: TestExecutionInput = TestExecutionInput(
        source_code=_NAIVE_CODE,
        test_code=_QA_TEST,
    )
    first_run = execute_in_sandbox(qa_input_naive)
    second_run = execute_in_sandbox(
        TestExecutionInput(source_code=_HARDENED_CODE, test_code=_QA_TEST)
    )
    initial_flaw: str = first_run.failure_details[0] if first_run.failure_details else "None"
    final_status: Literal["Passed", "Failed"] = "Passed" if second_run.passed else "Failed"
    summary: AdversarialCycleSummary = AdversarialCycleSummary(
        task=task_description,
        iterations=2,
        initial_flaw_detected=initial_flaw,
        final_status=final_status,
        generated_edge_cases=list(_EDGE_CASES),
    )
    return AdversarialReviewResponse(
        status="completed",
        pattern=_PATTERN,
        mode="mock",
        summary=summary,
        runs=[
            LoopIteration(step=1, role="qa_against_naive", result=first_run),
            LoopIteration(step=2, role="qa_against_hardened", result=second_run),
        ],
    )

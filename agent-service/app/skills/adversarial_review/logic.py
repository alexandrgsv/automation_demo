"""Scripted developer-versus-QA loop and the sandbox runner it calls.

The developer step and the QA step are fixed snippets. The first snippet
divides by the duration with no guard. The QA script requires a ``ValueError``
when the duration is zero. The second snippet adds that guard, and the same
script is run again.

The HTTP demo does not accept caller source. Imports and file access are
omitted from the namespace so a stray call cannot read the host.
"""

import traceback
from typing import Any, Literal

from app.skills.adversarial_review.schema import (
    AdversarialCycleSummary,
    AdversarialReviewResponse,
    LoopIteration,
    TestExecutionInput,
    TestExecutionResult,
)

EXECUTE_TOOL: str = "execute_in_sandbox"
_PATTERN: str = "Two-Agent Adversarial Feedback Loop (Dev + Red Team QA)"

_SANDBOX_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
    "AssertionError": AssertionError,
    "Exception": Exception,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}

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


def _namespace() -> dict[str, Any]:
    """Build a fresh execution namespace.

    Returns
    -------
    dict[str, Any]
        Globals that contain only the small builtin map.
    """
    return {"__builtins__": dict(_SANDBOX_BUILTINS)}


def _failure_line(prefix: str) -> str:
    """Take the last traceback line and prefix it.

    Parameters
    ----------
    prefix : str
        Short label such as ``Assertion failed``.

    Returns
    -------
    str
        One line suitable for ``failure_details``.
    """
    last_line: str = traceback.format_exc().splitlines()[-1]
    return f"{prefix}: {last_line}"


def execute_in_sandbox(payload: TestExecutionInput) -> TestExecutionResult:
    """Load an implementation and run one test script against it.

    A failure in the implementation load is reported and the test script is
    not started. One test script counts as one test.

    Parameters
    ----------
    payload : TestExecutionInput
        Implementation source and the QA script.

    Returns
    -------
    TestExecutionResult
        Pass or fail, plus a short failure line when the run did not pass.
    """
    env: dict[str, Any] = _namespace()
    try:
        exec(payload.source_code, env)  # noqa: S102
    except Exception as exc:
        return TestExecutionResult(
            passed=False,
            total_tests=0,
            failed_tests=1,
            failure_details=[f"Syntax/Import error in source code: {type(exc).__name__}: {exc}"],
        )

    try:
        exec(payload.test_code, env)  # noqa: S102
    except AssertionError:
        return TestExecutionResult(
            passed=False,
            total_tests=1,
            failed_tests=1,
            failure_details=[_failure_line("Assertion failed")],
        )
    except Exception as exc:
        return TestExecutionResult(
            passed=False,
            total_tests=1,
            failed_tests=1,
            failure_details=[f"Runtime exception: {type(exc).__name__}: {exc}"],
        )

    return TestExecutionResult(
        passed=True,
        total_tests=1,
        failed_tests=0,
        failure_details=[],
    )


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
    first_run: TestExecutionResult = execute_in_sandbox(qa_input_naive)
    second_run: TestExecutionResult = execute_in_sandbox(
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

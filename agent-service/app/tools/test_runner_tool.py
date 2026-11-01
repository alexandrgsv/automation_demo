"""Run one QA script against one implementation in a fresh namespace.

The HTTP demo does not accept caller source. The adversarial loop passes two
fixed snippets. Imports and file access are omitted from the namespace so a
stray call cannot read the host.
"""

import traceback
from typing import Any

from app.schemas import TestExecutionInput, TestExecutionResult

EXECUTE_TOOL: str = "execute_in_sandbox"

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

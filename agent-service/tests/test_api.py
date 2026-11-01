"""HTTP contract tests for health, run, and the SSE stream."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Build an isolated app forced into mock mode.

    Yields
    ------
    TestClient
        Client bound to a fresh runner with an empty API key.
    """
    application = create_app(Settings(environment="test", llm_api_key="", memory_window=8))
    with TestClient(application) as test_client:
        yield test_client


def test_health_reports_the_mock_worker(client: TestClient) -> None:
    """Expose worker readiness and the registered tool.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["llm_mode"] == "mock"
    assert body["workers_ready"] is True
    assert body["registered_tools"] == [
        "analyze_pr_diff",
        "calculate_review_risk_score",
        "evaluate_edge_config",
        "execute_in_sandbox",
        "generate_pytest_stub",
    ]


def test_root_redirects_to_the_docs(client: TestClient) -> None:
    """Send a browser that opens the origin to Swagger.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_run_returns_the_cuda_estimate(client: TestClient) -> None:
    """Return the deterministic cuda estimate for the sample payload.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/run",
        json={
            "message": "Check this edge config against the FPS target.",
            "session_id": "demo-1",
            "tool_arguments": {"batch_size": 2, "device": "cuda", "target_fps": 30},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "mock"
    assert body["session_id"] == "demo-1"
    assert body["steps_taken"] == 2
    assert body["latency_ms"] >= 0
    assert "40.0 FPS" in body["result"]
    assert body["tool_invocations"][0]["ok"] is True
    assert body["tool_invocations"][0]["output"]["estimated_fps"] == 40.0
    assert body["tool_invocations"][0]["output"]["meets_requirements"] is True


def test_run_repairs_a_gpu_alias(client: TestClient) -> None:
    """Record the failed call, rebuild the device, and then succeed.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/run",
        json={
            "message": "GPU alias should be rebuilt once.",
            "session_id": "repair-1",
            "tool_arguments": {"batch_size": 2, "device": "GPU", "target_fps": 30},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["steps_taken"] == 3
    assert len(body["tool_invocations"]) == 2
    assert body["tool_invocations"][0]["ok"] is False
    assert body["tool_invocations"][0]["error"]["type"] == "validation_error"
    assert body["tool_invocations"][1]["ok"] is True
    assert body["tool_invocations"][1]["arguments"]["device"] == "cuda"
    assert body["tool_invocations"][1]["output"]["estimated_fps"] == 40.0


def test_run_keeps_a_bad_tool_payload_on_http_200(client: TestClient) -> None:
    """Answer 200 when the tool schema fails and no repair is possible.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/run",
        json={
            "message": "This device is not in the contract.",
            "tool_arguments": {"batch_size": 0, "device": "tpu", "target_fps": 30},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["steps_taken"] == 2
    assert body["tool_invocations"][0]["ok"] is False
    assert body["tool_invocations"][0]["error"]["type"] == "validation_error"
    assert "was not executed" in body["result"]


def test_run_reads_json_embedded_in_the_message(client: TestClient) -> None:
    """Accept a benchmark object pasted into the user message.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/run",
        json={
            "message": 'Please check {"batch_size": 1, "device": "cpu", "target_fps": 10}.',
        },
    )
    assert response.status_code == 200
    output = response.json()["tool_invocations"][0]["output"]
    assert output["device"] == "cpu"
    assert output["estimated_fps"] == 22.22
    assert output["meets_requirements"] is True


def test_run_rejects_an_unknown_body_field(client: TestClient) -> None:
    """Reject a body that does not match RunRequest before the agent runs.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/run",
        json={"message": "hello", "nope": 1},
    )
    assert response.status_code == 422


def test_stream_emits_tool_and_result_events(client: TestClient) -> None:
    """Encode the mock trace as Server-Sent Events.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/stream",
        json={
            "message": "Stream the cuda check.",
            "session_id": "stream-1",
            "tool_arguments": {"batch_size": 2, "device": "cuda", "target_fps": 30},
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: step" in response.text
    assert "event: tool" in response.text
    assert "event: result" in response.text
    assert "40.0" in response.text


def test_pr_review_returns_a_high_risk_audit(client: TestClient) -> None:
    """Flag a bare except, a missing annotation, and a TODO as High risk.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/pr-review",
        json={
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
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["mode"] == "mock"
    assert body["pr_id"] == 42
    assert body["metrics"]["estimated_time_saved"] == "45 min"
    assert body["metrics"]["cycle_reduction"] == "From 3-day backlog to 1-day turnaround"
    assert body["review"]["risk_level"] == "High"
    assert body["review"]["can_auto_merge"] is False
    assert body["review"]["recommendation"] == "request_changes"
    assert body["review"]["review_time_saved_minutes"] == 45
    assert body["review"]["suggested_tests"] == [
        "test_agent_edge_cases",
        "test_agent_invalid_schema_handling",
    ]
    assert len(body["review"]["detected_issues"]) == 3
    assert [item["tool_name"] for item in body["tool_invocations"]] == [
        "analyze_pr_diff",
        "generate_pytest_stub",
        "calculate_review_risk_score",
    ]
    assert all(item["ok"] is True for item in body["tool_invocations"])


def test_pr_review_approves_a_typed_function(client: TestClient) -> None:
    """Approve a diff that already has a return annotation and a narrow handler.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/pr-review",
        json={
            "pr_id": 7,
            "file_path": "app/tools/metrics_tool.py",
            "diff_content": (
                "def score(value: int) -> int:\n"
                "    try:\n"
                "        return value\n"
                "    except ValueError:\n"
                "        return 0\n"
            ),
        },
    )
    assert response.status_code == 200
    review = response.json()["review"]
    assert review["risk_level"] == "Low"
    assert review["detected_issues"] == []
    assert review["can_auto_merge"] is True
    assert review["recommendation"] == "approve"
    assert review["suggested_tests"] == [
        "test_metrics_tool_edge_cases",
        "test_metrics_tool_invalid_schema_handling",
    ]


def test_pr_review_rejects_a_non_positive_id(client: TestClient) -> None:
    """Reject a pull request id of zero before the tools run.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/pr-review",
        json={"pr_id": 0, "file_path": "app/agent.py", "diff_content": "def run() -> None:\n    return\n"},
    )
    assert response.status_code == 422


def test_adversarial_review_fails_then_passes(client: TestClient) -> None:
    """Show the naive divide-by-zero, then the hardened pass.

    Parameters
    ----------
    client : TestClient
        Mock-mode application client.

    Returns
    -------
    None
        The test passes by assertion.
    """
    response = client.post(
        "/api/v1/adversarial-review",
        json={"task": "Implement edge throughput calculation with zero-division safety"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["mode"] == "mock"
    assert body["summary"]["iterations"] == 2
    assert body["summary"]["final_status"] == "Passed"
    assert "ZeroDivisionError" in body["summary"]["initial_flaw_detected"]
    assert body["runs"][0]["result"]["passed"] is False
    assert body["runs"][1]["result"]["passed"] is True
    assert body["runs"][1]["result"]["failed_tests"] == 0


def test_sandbox_reports_a_blocked_import() -> None:
    """Keep an import attempt inside the runner result.

    Returns
    -------
    None
        The test passes by assertion.
    """
    from app.schemas import TestExecutionInput
    from app.tools.test_runner_tool import execute_in_sandbox

    result = execute_in_sandbox(
        TestExecutionInput(source_code="import os\n", test_code="assert True\n")
    )
    assert result.passed is False
    assert result.total_tests == 0
    assert result.failure_details

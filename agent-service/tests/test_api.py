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
    assert body["registered_tools"] == ["evaluate_edge_config"]


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

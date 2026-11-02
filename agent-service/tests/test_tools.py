"""Schema and deterministic-tool tests."""

import asyncio

import pytest
from pydantic import ValidationError

from app.agent import AgentRunner, extract_json_object, repair_benchmark_arguments
from app.config import Settings
from app.schemas import BenchmarkInput, PRReviewInput, RunRequest, ToolError
from app.tools import SKILLS, TOOL_NAME, dispatch_tool
from app.tools.metrics_tool import evaluate_edge_config
from app.tools.pr_review_tool import audit_code_changes


def test_benchmark_input_rejects_bad_device_and_batch() -> None:
    """Reject a device outside the contract and a batch size of zero.

    Returns
    -------
    None
        The test passes by assertion.
    """
    with pytest.raises(ValidationError):
        BenchmarkInput(batch_size=1, device="gpu", target_fps=30)
    with pytest.raises(ValidationError):
        BenchmarkInput(batch_size=0, device="cuda", target_fps=30)
    with pytest.raises(ValidationError):
        BenchmarkInput(batch_size=65, device="cuda", target_fps=30)
    with pytest.raises(ValidationError):
        BenchmarkInput(batch_size=1, device="cuda", target_fps=0)


def test_benchmark_schema_exposes_the_device_pattern() -> None:
    """Publish the device pattern on the JSON schema used for tool calling.

    Returns
    -------
    None
        The test passes by assertion.
    """
    schema: dict[str, object] = BenchmarkInput.model_json_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    device = properties["device"]
    assert isinstance(device, dict)
    assert device["pattern"] == "^(cpu|cuda|tensorrt)$"


def test_evaluate_edge_config_cuda_meets_target() -> None:
    """Match the demo payload: cuda, batch 2, target 30 FPS.

    Latency is 12.5 ms times 2. FPS is 1000 / 25.

    Returns
    -------
    None
        The test passes by assertion.
    """
    result = evaluate_edge_config(
        BenchmarkInput(batch_size=2, device="cuda", target_fps=30)
    )
    assert result.estimated_latency_ms == 25.0
    assert result.estimated_fps == 40.0
    assert result.meets_requirements is True
    assert result.bottleneck_detected is False


def test_evaluate_edge_config_cpu_reports_a_bottleneck() -> None:
    """Show a miss when CPU latency stacks past the target.

    Returns
    -------
    None
        The test passes by assertion.
    """
    result = evaluate_edge_config(
        BenchmarkInput(batch_size=4, device="cpu", target_fps=30)
    )
    assert result.estimated_latency_ms == 180.0
    assert result.estimated_fps == 5.56
    assert result.meets_requirements is False
    assert result.bottleneck_detected is True


def test_evaluate_edge_config_exact_fps_boundary_passes() -> None:
    """Treat an estimate equal to the target as a pass.

    Returns
    -------
    None
        The test passes by assertion.
    """
    result = evaluate_edge_config(
        BenchmarkInput(batch_size=1, device="cuda", target_fps=80)
    )
    assert result.estimated_fps == 80.0
    assert result.meets_requirements is True


def test_dispatch_returns_validation_error_instead_of_raising() -> None:
    """Keep a bad tool payload inside a structured error.

    Returns
    -------
    None
        The test passes by assertion.
    """
    invocation = dispatch_tool(
        TOOL_NAME,
        {"batch_size": 0, "device": "tpu", "target_fps": 30},
    )
    assert invocation.ok is False
    assert invocation.error is not None
    assert invocation.error.type == "validation_error"
    assert "batch_size" in invocation.error.message
    assert invocation.output is None


def test_dispatch_unknown_tool_is_a_structured_error() -> None:
    """Return unknown_tool when the name is not in the registry.

    Returns
    -------
    None
        The test passes by assertion.
    """
    invocation = dispatch_tool("missing_tool", {"batch_size": 1})
    assert invocation.ok is False
    assert invocation.error is not None
    assert invocation.error.type == "unknown_tool"


def test_repair_rewrites_device_alias_once() -> None:
    """Map GPU onto cuda and leave an unknown device unchanged.

    Returns
    -------
    None
        The test passes by assertion.
    """
    error = ToolError(type="validation_error", message="bad device", details=[])
    repaired = repair_benchmark_arguments(
        {"batch_size": 2, "device": "GPU", "target_fps": 30},
        error,
    )
    assert repaired == {"batch_size": 2, "device": "cuda", "target_fps": 30}
    assert (
        repair_benchmark_arguments(
            {"batch_size": 2, "device": "tpu", "target_fps": 30},
            error,
        )
        is None
    )


def test_extract_json_object_skips_prose() -> None:
    """Read the first object when the user wraps it in a sentence.

    Returns
    -------
    None
        The test passes by assertion.
    """
    found = extract_json_object(
        'Check this config {"batch_size": 1, "device": "cpu", "target_fps": 10} please.'
    )
    assert found == {"batch_size": 1, "device": "cpu", "target_fps": 10}
    assert extract_json_object("no payload here") is None


def test_provider_failure_falls_back_to_mock() -> None:
    """Answer from the deterministic tool when the chat provider cannot be reached.

    Returns
    -------
    None
        The test passes by assertion.
    """
    runner = AgentRunner(
        Settings(
            llm_api_key="test-key",
            llm_base_url="http://127.0.0.1:9",
            llm_timeout_s=0.2,
            memory_window=4,
        )
    )

    async def scenario() -> None:
        response = await runner.run(
            RunRequest(
                message="Check the config.",
                session_id="fallback-1",
                tool_arguments={"batch_size": 2, "device": "cuda", "target_fps": 30},
            )
        )
        await runner.aclose()
        assert response.mode == "mock"
        assert response.steps_taken == 2
        assert "mock runner answered instead" in response.result
        assert "40.0 FPS" in response.result

    asyncio.run(scenario())


def test_one_issue_is_medium_and_two_issues_are_high() -> None:
    """Score one finding as Medium and two findings as High.

    Returns
    -------
    None
        The test passes by assertion.
    """
    medium = audit_code_changes(
        PRReviewInput(
            pr_id=1,
            file_path="app/agent.py",
            diff_content="def run(data: int) -> int:\n    return data\n# TODO: explain the cast\n",
        )
    )
    assert medium.review.risk_level == "Medium"
    assert medium.review.recommendation == "request_changes"
    assert medium.review.detected_issues == ["Unresolved TODO/FIXME markers in production diff"]

    high = audit_code_changes(
        PRReviewInput(
            pr_id=2,
            file_path="app/agent.py",
            diff_content="def run(data):\n    try:\n        return data\n    except Exception as exc:\n        raise\n",
        )
    )
    assert high.review.risk_level == "High"
    assert "Missing return type annotations in modified functions" in high.review.detected_issues
    assert "Broad exception handling detected; specify exact exception types" in high.review.detected_issues


def test_session_window_keeps_only_the_latest_turns() -> None:
    """Drop the oldest turns when the window is two messages.

    Returns
    -------
    None
        The test passes by assertion.
    """
    runner = AgentRunner(Settings(memory_window=2, llm_api_key=""))
    arguments: dict[str, object] = {"batch_size": 2, "device": "cuda", "target_fps": 30}

    async def scenario() -> None:
        await runner.run(
            RunRequest(message="first", session_id="session-a", tool_arguments=arguments)
        )
        second = await runner.run(
            RunRequest(message="second", session_id="session-a", tool_arguments=arguments)
        )
        stored = runner.memory.get("session-a")
        assert [item.role for item in stored] == ["user", "assistant"]
        assert stored[0].content == "second"
        assert stored[1].content == second.result

    asyncio.run(scenario())


def test_skills_publish_contract_protocol_and_logic() -> None:
    """Each skill folder exposes a schema, a protocol file, and an execute method.

    Returns
    -------
    None
        The test passes by assertion.
    """
    names: list[str] = [skill.name for skill in SKILLS]
    assert names == [
        "edge_fps_evaluator",
        "pr_code_reviewer",
        "adversarial_review",
        "algorithm_deployer",
    ]
    for skill in SKILLS:
        assert skill.description
        assert skill.tools
        assert skill.instructions()
    result = SKILLS[0].execute(BenchmarkInput(batch_size=2, device="cuda", target_fps=30))
    assert result.estimated_fps == 40.0
    assert "gpu" in SKILLS[0].instructions()


def test_algorithm_retrieval_picks_the_named_sample() -> None:
    """Map three demo sentences onto the three sample documents.

    Returns
    -------
    None
        The test passes by assertion.
    """
    from app.skills.algorithm_deployer.logic import retrieve_algorithm

    line = retrieve_algorithm("Продеплой детекцию пересечения линии на входе")
    zone = retrieve_algorithm("Нужно видеть нахождение человека в зоне торгового зала")
    heat = retrieve_algorithm("Собери хитмап перемещений человека")
    assert line.selected is not None and line.selected.algorithm_id == "line_crossing"
    assert zone.selected is not None and zone.selected.algorithm_id == "zone_presence"
    assert heat.selected is not None and heat.selected.algorithm_id == "movement_heatmap"
    assert heat.selected.document == "movement_heatmap.json"


def test_portainer_create_uses_the_sample_stack_and_hides_the_key() -> None:
    """Post the corpus config and keep the API key out of the result.

    Returns
    -------
    None
        The test passes by assertion.
    """
    import json

    import httpx

    from app.skills.algorithm_deployer.logic import deploy_from_request
    from app.skills.algorithm_deployer.schema import AlgorithmDeployRequest

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["endpoint"] = request.url.params.get("endpointId")
        seen["key"] = request.headers.get("X-API-Key")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"Id": 17, "Name": "sample-zone-presence"})

    settings = Settings(
        llm_api_key="",
        portainer_url="http://portainer.internal:9000",
        portainer_api_key="demo-key",
        portainer_endpoint_id=4,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        outcome = deploy_from_request(
            AlgorithmDeployRequest(request="person in zone on the sales floor"),
            settings,
            client=http,
        )
    dumped = outcome.model_dump_json()
    assert outcome.status == "deployed"
    assert outcome.portainer is not None
    assert outcome.portainer.stack_id == 17
    assert outcome.algorithm_id == "zone_presence"
    assert seen["path"] == "/api/stacks/create/standalone/string"
    assert seen["endpoint"] == "4"
    assert seen["key"] == "demo-key"
    assert seen["body"]["name"] == "sample-zone-presence"  # type: ignore[index]
    assert "demo-key" not in dumped


def test_portainer_transport_error_stays_in_the_result() -> None:
    """Report a connection failure without raising it to the caller.

    Returns
    -------
    None
        The test passes by assertion.
    """
    import httpx

    from app.skills.algorithm_deployer.logic import deploy_from_request
    from app.skills.algorithm_deployer.schema import AlgorithmDeployRequest

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    settings = Settings(
        llm_api_key="",
        portainer_url="http://portainer.internal:9000",
        portainer_api_key="demo-key",
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        outcome = deploy_from_request(
            AlgorithmDeployRequest(request="heatmap of movement"),
            settings,
            client=http,
        )
    assert outcome.status == "deploy_failed"
    assert outcome.portainer is not None
    assert outcome.portainer.deployed is False
    assert "ConnectError" in outcome.portainer.message

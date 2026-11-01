"""Schema and deterministic-tool tests."""

import asyncio

import pytest
from pydantic import ValidationError

from app.agent import AgentRunner, extract_json_object, repair_benchmark_arguments
from app.config import Settings
from app.schemas import BenchmarkInput, RunRequest, ToolError
from app.tools import TOOL_NAME, dispatch_tool
from app.tools.metrics_tool import evaluate_edge_config


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

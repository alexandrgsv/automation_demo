"""Tool registry and the dispatcher that validates arguments before a call."""

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.schemas import (
    BenchmarkInput,
    PRReviewInput,
    RiskScoreInput,
    TestExecutionInput,
    ToolError,
    ToolInvocation,
)
from app.tools.metrics_tool import evaluate_edge_config
from app.tools.test_runner_tool import EXECUTE_TOOL, execute_in_sandbox
from app.tools.pr_review_tool import (
    ANALYZE_TOOL,
    RISK_TOOL,
    STUB_TOOL,
    analyze_pr_diff,
    calculate_review_risk_score,
    generate_pytest_stub,
)

TOOL_NAME: str = "evaluate_edge_config"


class ToolSpec(dict[str, Any]):
    """Registry record for one tool.

    Keys are ``name``, ``description``, ``input_model``, and ``handler``.
    A dict subclass keeps construction obvious when the file is read on a call.
    """


def _build_registry() -> dict[str, ToolSpec]:
    """Build the process tool registry.

    Returns
    -------
    dict[str, ToolSpec]
        Map of tool name to spec.
    """
    return {
        TOOL_NAME: ToolSpec(
            name=TOOL_NAME,
            description=(
                "Estimate FPS and latency for an edge inference config. "
                "device must be cpu, cuda, or tensorrt. "
                "Do not invent FPS numbers; return this tool's output."
            ),
            input_model=BenchmarkInput,
            handler=evaluate_edge_config,
        ),
        ANALYZE_TOOL: ToolSpec(
            name=ANALYZE_TOOL,
            description=(
                "Scan a pull request diff for missing return annotations, "
                "broad exception handlers, and TODO or FIXME markers."
            ),
            input_model=PRReviewInput,
            handler=analyze_pr_diff,
        ),
        STUB_TOOL: ToolSpec(
            name=STUB_TOOL,
            description="Build a deterministic pytest skeleton from the changed file name.",
            input_model=PRReviewInput,
            handler=generate_pytest_stub,
        ),
        EXECUTE_TOOL: ToolSpec(
            name=EXECUTE_TOOL,
            description=(
                "Run one QA script against one implementation in a fresh namespace. "
                "Imports and file access are not available."
            ),
            input_model=TestExecutionInput,
            handler=execute_in_sandbox,
        ),
        RISK_TOOL: ToolSpec(
            name=RISK_TOOL,
            description=(
                "Score merge risk from analyze_pr_diff issues. "
                "Low approves. Medium and High request changes."
            ),
            input_model=RiskScoreInput,
            handler=calculate_review_risk_score,
        ),
    }


REGISTRY: dict[str, ToolSpec] = _build_registry()


def list_tool_names() -> list[str]:
    """Return registered tool names in sorted order.

    Returns
    -------
    list[str]
        Tool names the dispatcher accepts.
    """
    return sorted(REGISTRY)


def openai_tools() -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool definitions from the registry.

    Returns
    -------
    list[dict[str, Any]]
        Payload for the ``tools`` field of a chat completion request.
    """
    definitions: list[dict[str, Any]] = []
    for spec in REGISTRY.values():
        input_model: type[BaseModel] = spec["input_model"]
        schema: dict[str, Any] = input_model.model_json_schema()
        schema.pop("title", None)
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": schema,
                },
            }
        )
    return definitions


def _validation_message(details: list[dict[str, Any]]) -> str:
    """Join Pydantic errors into one sentence.

    Parameters
    ----------
    details : list[dict[str, Any]]
        Items produced by ``ValidationError.json``.

    Returns
    -------
    str
        A single sentence suitable for a model or a demo response.
    """
    parts: list[str] = []
    for item in details:
        location = ".".join(str(piece) for piece in item.get("loc", []))
        text = str(item.get("msg", "invalid value"))
        parts.append(f"{location}: {text}" if location else text)
    joined = "; ".join(parts) if parts else "invalid arguments"
    return f"Tool arguments failed validation. {joined}"


def dispatch_tool(name: str, arguments: Any) -> ToolInvocation:
    """Validate arguments and run one tool.

    A schema failure is returned as :class:`app.schemas.ToolInvocation`.
    It is not raised, so the HTTP handler can keep answering 200 while the
    orchestrator decides whether to repair the call.

    Parameters
    ----------
    name : str
        Tool name from the model or the mock runner.
    arguments : Any
        JSON-like arguments. Only a dict is forwarded to Pydantic.

    Returns
    -------
    ToolInvocation
        Either the tool output or a structured error.
    """
    spec: ToolSpec | None = REGISTRY.get(name)
    if spec is None:
        return ToolInvocation(
            tool_name=name,
            arguments=arguments if isinstance(arguments, dict) else {},
            ok=False,
            error=ToolError(
                type="unknown_tool",
                message=f"Unknown tool: {name}. Known tools: {', '.join(list_tool_names())}.",
            ),
        )

    if not isinstance(arguments, dict):
        return ToolInvocation(
            tool_name=name,
            arguments={},
            ok=False,
            error=ToolError(
                type="invalid_arguments",
                message="Tool arguments must be a JSON object.",
            ),
        )

    input_model: type[BaseModel] = spec["input_model"]
    try:
        parsed: BaseModel = input_model.model_validate(arguments)
    except ValidationError as exc:
        details: list[dict[str, Any]] = json.loads(exc.json())
        return ToolInvocation(
            tool_name=name,
            arguments=arguments,
            ok=False,
            error=ToolError(
                type="validation_error",
                message=_validation_message(details),
                details=details,
            ),
        )

    handler: Callable[..., BaseModel] = spec["handler"]
    try:
        result: BaseModel = handler(parsed)
    except (ValueError, KeyError) as exc:
        return ToolInvocation(
            tool_name=name,
            arguments=arguments,
            ok=False,
            error=ToolError(type="tool_failure", message=str(exc)),
        )

    return ToolInvocation(
        tool_name=name,
        arguments=arguments,
        ok=True,
        output=result.model_dump(mode="json"),
    )

"""HTTP contracts for the edge agent service.

Skill argument and result models live next to their skill, under
``app.skills``. They are re-exported here so existing imports keep working.
Input models reject unknown fields. Tool-argument failures stay inside the
agent response so a bad device name does not become an HTTP 500.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.skills.adversarial_review.schema import (
    AdversarialCycleSummary,
    AdversarialReviewRequest,
    AdversarialReviewResponse,
    LoopIteration,
    TestExecutionInput,
    TestExecutionResult,
)
from app.skills.edge_fps_evaluator.schema import BenchmarkInput, EdgeBenchmarkResult
from app.skills.pr_code_reviewer.schema import (
    DiffAnalysis,
    PRReviewInput,
    PRReviewSummary,
    PytestStub,
    ReviewMetrics,
    RiskScore,
    RiskScoreInput,
)

__all__ = [
    "AdversarialCycleSummary",
    "AdversarialReviewRequest",
    "AdversarialReviewResponse",
    "AgentEvent",
    "AgentResponse",
    "BenchmarkInput",
    "ChatMessage",
    "DiffAnalysis",
    "EdgeBenchmarkResult",
    "HealthResponse",
    "LoopIteration",
    "PRReviewInput",
    "PRReviewResponse",
    "PRReviewSummary",
    "PytestStub",
    "ReviewMetrics",
    "RiskScore",
    "RiskScoreInput",
    "RunRequest",
    "TestExecutionInput",
    "TestExecutionResult",
    "ToolError",
    "ToolInvocation",
]


class ChatMessage(BaseModel):
    """One turn stored in the session window.

    Attributes
    ----------
    role : {"user", "assistant"}
        Speaker of this turn.
    content : str
        Plain text. Tool payloads are not stored here.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"] = Field(description="Speaker of this turn.")
    content: str = Field(min_length=1, max_length=8000, description="Plain-text turn.")


class ToolError(BaseModel):
    """Structured tool failure returned to the orchestrator.

    Attributes
    ----------
    type : {"validation_error", "unknown_tool", "invalid_arguments", "tool_failure"}
        Machine-readable failure class.
    message : str
        Single sentence the agent can show or feed back to a model.
    details : list[dict[str, Any]]
        Pydantic error items when ``type`` is ``validation_error``.
    """

    type: Literal["validation_error", "unknown_tool", "invalid_arguments", "tool_failure"] = Field(
        description="Machine-readable failure class."
    )
    message: str = Field(description="Human-readable failure summary.")
    details: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Schema errors. Empty when the failure is not a validation error.",
    )


class ToolInvocation(BaseModel):
    """One attempt to call a tool, including a failed validation.

    Attributes
    ----------
    tool_name : str
        Registry name, for example ``evaluate_edge_config``.
    arguments : dict[str, Any]
        Arguments submitted on this attempt.
    ok : bool
        True when the tool returned a result.
    output : dict[str, Any] | None
        Tool result. Absent when ``ok`` is False.
    error : ToolError | None
        Structured failure. Absent when ``ok`` is True.
    """

    tool_name: str = Field(description="Registered tool name.")
    arguments: dict[str, Any] = Field(description="Arguments submitted on this attempt.")
    ok: bool = Field(description="True when the tool returned a result.")
    output: dict[str, Any] | None = Field(default=None, description="Tool result object.")
    error: ToolError | None = Field(default=None, description="Structured failure, if the call did not run.")


class RunRequest(BaseModel):
    """Body of ``POST /api/v1/run`` and ``POST /api/v1/stream``.

    Attributes
    ----------
    message : str
        New user turn. This is the instruction for the orchestrator.
    session_id : str | None
        Key for the in-process sliding window. Generated when omitted.
    tool_arguments : dict[str, Any] | None
        Optional explicit arguments for ``evaluate_edge_config``.
        Mock mode prefers these over JSON found inside ``message``.
    messages : list[ChatMessage] | None
        Prior turns. When set, they replace the stored window for this call.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "Check this edge config against the FPS target.",
                    "session_id": "demo-1",
                    "tool_arguments": {
                        "batch_size": 2,
                        "device": "cuda",
                        "target_fps": 30,
                    },
                }
            ]
        },
    )

    message: str = Field(
        min_length=1,
        max_length=8000,
        description="New user turn. Ask for an edge config check, or include a JSON object.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Session key for the sliding window. A UUID is generated when omitted.",
    )
    tool_arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Explicit evaluate_edge_config arguments. "
            "Invalid values stay in the agent response as a tool error."
        ),
    )
    messages: list[ChatMessage] | None = Field(
        default=None,
        description=(
            "Prior turns, oldest first. When set, they replace the server window "
            "for this session. Do not repeat the new turn here; put it in message."
        ),
    )


class AgentResponse(BaseModel):
    """Synchronous answer from the orchestrator.

    Attributes
    ----------
    result : str
        Final text for this turn.
    steps_taken : int
        Orchestrator iterations. See the field description for the exact count.
    tool_invocations : list[ToolInvocation]
        Every tool attempt, including failed validation and a repaired retry.
    latency_ms : float
        Wall time of this turn, measured inside the process.
    mode : {"mock", "llm"}
        ``mock`` when no key is set or the provider call failed over.
    session_id : str
        Session key used for the sliding window.
    """

    result: str = Field(description="Final text for this turn.")
    steps_taken: int = Field(
        ge=0,
        description=(
            "Orchestrator iterations for this turn. "
            "Mock mode counts each tool attempt and the final answer. "
            "LLM mode counts each chat completion request."
        ),
    )
    tool_invocations: list[ToolInvocation] = Field(
        description="Tool attempts in order, including validation failures."
    )
    latency_ms: float = Field(ge=0, description="Server-side wall time for this turn, in milliseconds.")
    mode: Literal["mock", "llm"] = Field(description="mock when no provider was used, otherwise llm.")
    session_id: str = Field(description="Session key for a follow-up turn.")


class AgentEvent(BaseModel):
    """One Server-Sent Event emitted by ``POST /api/v1/stream``.

    Attributes
    ----------
    event : {"step", "tool", "result"}
        SSE event name.
    data : dict[str, Any]
        JSON payload. ``result`` carries a full ``AgentResponse``.
    """

    event: Literal["step", "tool", "result"] = Field(description="SSE event name.")
    data: dict[str, Any] = Field(description="JSON payload for this event.")


class HealthResponse(BaseModel):
    """Liveness payload for ``GET /health``.

    Attributes
    ----------
    status : {"ok"}
        Process is serving requests.
    environment : str
        Value of the ``ENVIRONMENT`` setting.
    llm_mode : {"mock", "llm"}
        Which runner a new turn will try first.
    workers_ready : bool
        True when the tool registry imported and can accept ``/api/v1/run``.
    registered_tools : list[str]
        Tool names the dispatcher can call.
    """

    status: Literal["ok"] = Field(description="Process is serving.")
    environment: str = Field(description="Deployment label from settings.")
    llm_mode: Literal["mock", "llm"] = Field(description="Runner selected for a new turn.")
    workers_ready: bool = Field(description="True when the in-process worker can run tools.")
    registered_tools: list[str] = Field(description="Tools loaded in this process.")


class PRReviewResponse(BaseModel):
    """Body of ``POST /api/v1/pr-review``.

    The pipeline is deterministic. ``mode`` stays ``mock`` because this route
    does not call a chat provider.

    Attributes
    ----------
    status : {"success"}
        The three tools finished and the summary was built.
    pr_id : int
        Identifier copied from the request.
    mode : {"mock"}
        Always mock. The audit rules are fixed in code.
    metrics : ReviewMetrics
        Time-saved label and the cycle sentence.
    review : PRReviewSummary
        Risk, issues, tests, and the merge recommendation.
    tool_invocations : list[ToolInvocation]
        The three tool calls, in pipeline order.
    """

    status: Literal["success"] = Field(description="The pipeline finished.")
    pr_id: int = Field(description="Pull request identifier from the request.")
    mode: Literal["mock"] = Field(description="This route always uses the deterministic audit.")
    metrics: ReviewMetrics = Field(description="Time-saved label and cycle sentence.")
    review: PRReviewSummary = Field(description="Combined audit.")
    tool_invocations: list[ToolInvocation] = Field(description="analyze, stub, then risk, in that order.")

"""Orchestrator: sliding-window memory, tool retry, and mock fallback.

Mock mode is the default. It runs when ``LLM_API_KEY`` is empty and also when
a configured provider raises a transport or protocol error. The FPS function
stays pure. The device-alias repair is declared on the edge FPS skill, and
this module applies that repair once.
"""

import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from app.config import Settings
from app.schemas import (
    AgentEvent,
    AgentResponse,
    ChatMessage,
    RunRequest,
    ToolError,
    ToolInvocation,
)
from app.skills.edge_fps_evaluator.skill import EDGE_FPS_SKILL
from app.tools import TOOL_NAME, dispatch_tool, openai_tools

logger: logging.Logger = logging.getLogger(__name__)

# The protocol file in the skill directory is the source of this prompt.
SYSTEM_PROMPT: str = EDGE_FPS_SKILL.instructions()


class LLMUnavailable(Exception):
    """The chat provider could not return a usable completion."""


class SessionMemory:
    """In-process sliding window of chat messages.

    The buffer is per session and capped at ``window`` messages. It is not
    shared across processes. A restart clears it.

    Parameters
    ----------
    window : int
        Maximum number of messages kept for one session. Must be at least 1.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window: int = window
        self._sessions: dict[str, list[ChatMessage]] = {}
        self._lock: threading.Lock = threading.Lock()

    def get(self, session_id: str) -> list[ChatMessage]:
        """Return a copy of the stored turns.

        Parameters
        ----------
        session_id : str
            Session key.

        Returns
        -------
        list[ChatMessage]
            Oldest first. Empty when the session is unknown.
        """
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def replace(self, session_id: str, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Replace the session with ``messages``, trimmed to the window.

        Parameters
        ----------
        session_id : str
            Session key.
        messages : list[ChatMessage]
            Full history, oldest first.

        Returns
        -------
        list[ChatMessage]
            The stored suffix.
        """
        with self._lock:
            trimmed: list[ChatMessage] = list(messages[-self._window :])
            self._sessions[session_id] = trimmed
            return list(trimmed)

    def extend(self, session_id: str, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Append turns and drop anything past the window.

        Parameters
        ----------
        session_id : str
            Session key.
        messages : list[ChatMessage]
            Turns to append, oldest first.

        Returns
        -------
        list[ChatMessage]
            The stored suffix after the append.
        """
        with self._lock:
            bucket: list[ChatMessage] = self._sessions.setdefault(session_id, [])
            bucket.extend(messages)
            if len(bucket) > self._window:
                del bucket[: len(bucket) - self._window]
            return list(bucket)


class TurnOutcome:
    """Final text and tool trace for one orchestrator path.

    Parameters
    ----------
    text : str
        Answer that will be stored as the assistant turn.
    steps : int
        Iterations counted for this path.
    invocations : list[ToolInvocation]
        Tool attempts that belong in the response.
    mode : {"mock", "llm"}
        Which path produced ``text``.
    """

    def __init__(
        self,
        text: str,
        steps: int,
        invocations: list[ToolInvocation],
        mode: Literal["mock", "llm"],
    ) -> None:
        self.text: str = text
        self.steps: int = steps
        self.invocations: list[ToolInvocation] = invocations
        self.mode: Literal["mock", "llm"] = mode


class OutcomeBox:
    """Mutable slot filled by the LLM generator before it returns.

    An async generator cannot return a rich object cleanly. The generator
    writes the outcome here, and the caller reads it after iteration.
    """

    def __init__(self) -> None:
        self.outcome: TurnOutcome | None = None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object embedded in ``text``.

    Parameters
    ----------
    text : str
        User message, possibly with prose around one object.

    Returns
    -------
    dict[str, Any] | None
        The object, or None when no JSON object is present.
    """
    decoder: json.JSONDecoder = json.JSONDecoder()
    start: int = text.find("{")
    while start != -1:
        try:
            value: Any
            value, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def repair_benchmark_arguments(
    arguments: dict[str, Any],
    error: ToolError,
) -> dict[str, Any] | None:
    """Apply the edge FPS skill's one-step device repair.

    The alias table lives on the skill. Only the device string is rewritten,
    and only onto ``cpu``, ``cuda``, or ``tensorrt``. Batch size and target
    FPS are left alone so a bad number is reported instead of being clamped.

    Parameters
    ----------
    arguments : dict[str, Any]
        Arguments that failed validation.
    error : ToolError
        Structured failure from the dispatcher.

    Returns
    -------
    dict[str, Any] | None
        A new argument dict, or None when this policy cannot repair the call.
    """
    return EDGE_FPS_SKILL.repair_arguments(arguments, error)


def summarize_invocations(
    invocations: list[ToolInvocation],
    *,
    fallback_note: str | None,
) -> str:
    """Turn tool attempts into the mock-mode answer.

    Parameters
    ----------
    invocations : list[ToolInvocation]
        Attempts in call order. The last successful output wins.
    fallback_note : str | None
        Sentence prepended when a live provider was skipped or failed.

    Returns
    -------
    str
        Plain-text answer for ``AgentResponse.result``.
    """
    success: ToolInvocation | None = next(
        (item for item in reversed(invocations) if item.ok and item.output is not None),
        None,
    )
    text: str
    if success is not None and success.output is not None:
        output: dict[str, Any] = success.output
        verdict: str = "meets" if output["meets_requirements"] else "misses"
        bottleneck: str = (
            "A bottleneck was detected."
            if output["bottleneck_detected"]
            else "No bottleneck was detected."
        )
        text = (
            f"Config device={output['device']} batch_size={output['batch_size']} "
            f"estimates {output['estimated_fps']} FPS "
            f"(latency {output['estimated_latency_ms']} ms) and {verdict} "
            f"the target of {output['target_fps']} FPS. {bottleneck}"
        )
    elif invocations:
        last_error: ToolError | None = invocations[-1].error
        if last_error is not None:
            text = f"evaluate_edge_config was not executed. {last_error.message}"
        else:
            text = "evaluate_edge_config was not executed."
    else:
        text = (
            "Mock mode needs tool arguments. Pass tool_arguments or include a JSON object "
            "with batch_size, device, and target_fps."
        )
    if fallback_note:
        return f"{fallback_note} {text}"
    return text


class AgentRunner:
    """Run one user turn through the tool loop.

    Parameters
    ----------
    settings : Settings
        Process configuration. An empty API key selects mock mode.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._memory: SessionMemory = SessionMemory(settings.memory_window)
        self._http: httpx.AsyncClient | None = None

    @property
    def memory(self) -> SessionMemory:
        """Return the session buffer.

        Returns
        -------
        SessionMemory
            In-process sliding window owned by this runner.
        """
        return self._memory

    async def aclose(self) -> None:
        """Close the shared HTTP client, if a live call opened one.

        Returns
        -------
        None
            Nothing.
        """
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def run(self, request: RunRequest) -> AgentResponse:
        """Execute one turn and return the final response.

        Parameters
        ----------
        request : RunRequest
            Validated request body.

        Returns
        -------
        AgentResponse
            Result text, tool trace, and timing.

        Raises
        ------
        RuntimeError
            If the event stream ends without a result event. The stream is
            written to always emit one.
        """
        response: AgentResponse | None = None
        async for event in self.iter_events(request):
            if event.event == "result":
                response = AgentResponse.model_validate(event.data)
        if response is None:
            raise RuntimeError("Agent loop ended without a result.")
        return response

    async def iter_events(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        """Yield step, tool, and result events for one turn.

        The last event is always ``result``. Provider failures before any
        token is emitted fall back to the mock runner.

        Parameters
        ----------
        request : RunRequest
            Validated request body.

        Yields
        ------
        AgentEvent
            Progress events, then the full response.
        """
        started: float = time.perf_counter()
        session_id: str = request.session_id or str(uuid.uuid4())
        history: list[ChatMessage] = self._prepare_history(request, session_id)
        box: OutcomeBox = OutcomeBox()
        emitted: bool = False
        fallback_note: str | None = None

        try:
            if self._settings.llm_configured:
                try:
                    async for event in self._iter_llm(request, history, box):
                        emitted = True
                        yield event
                except LLMUnavailable as exc:
                    logger.warning("LLM unavailable, using mock mode: %s", exc.__class__.__name__)
                    fallback_note = (
                        f"LLM provider unavailable ({exc.__class__.__name__}); "
                        "mock runner answered instead."
                    )
                    if emitted:
                        # The client already saw a partial trace. Answer from the tool
                        # arguments anyway so the turn still finishes.
                        box.outcome = None

            if box.outcome is None:
                outcome, events = self._mock_turn(request, fallback_note=fallback_note)
                for event in events:
                    yield event
                box.outcome = outcome

            response: AgentResponse = self._finish(box.outcome, session_id, started)
            self._memory.extend(
                session_id,
                [ChatMessage(role="assistant", content=response.result)],
            )
            yield AgentEvent(event="result", data=response.model_dump(mode="json"))
        except Exception as exc:
            logger.exception("Agent loop failed")
            response = AgentResponse(
                result=f"Agent loop failed ({exc.__class__.__name__}). The service is still up.",
                steps_taken=0,
                tool_invocations=[],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                mode="mock",
                session_id=session_id,
            )
            yield AgentEvent(event="result", data=response.model_dump(mode="json"))

    def _prepare_history(self, request: RunRequest, session_id: str) -> list[ChatMessage]:
        """Store the new user turn and return the trimmed window.

        Parameters
        ----------
        request : RunRequest
            Validated request body.
        session_id : str
            Session key, already defaulted by the caller.

        Returns
        -------
        list[ChatMessage]
            Window contents after the user turn is appended, oldest first.
        """
        user_turn: ChatMessage = ChatMessage(role="user", content=request.message)
        if request.messages is not None:
            prior: list[ChatMessage] = list(request.messages)
        else:
            prior = self._memory.get(session_id)
        return self._memory.replace(session_id, [*prior, user_turn])

    def _finish(self, outcome: TurnOutcome, session_id: str, started: float) -> AgentResponse:
        """Build the response model for a completed turn.

        Parameters
        ----------
        outcome : TurnOutcome
            Text and tool trace.
        session_id : str
            Session key to echo back.
        started : float
            ``time.perf_counter`` value taken at the start of the turn.

        Returns
        -------
        AgentResponse
            Response ready to serialize.
        """
        elapsed_ms: float = round((time.perf_counter() - started) * 1000, 2)
        return AgentResponse(
            result=outcome.text,
            steps_taken=outcome.steps,
            tool_invocations=outcome.invocations,
            latency_ms=elapsed_ms,
            mode=outcome.mode,
            session_id=session_id,
        )

    def _mock_turn(
        self,
        request: RunRequest,
        *,
        fallback_note: str | None,
    ) -> tuple[TurnOutcome, list[AgentEvent]]:
        """Resolve arguments locally, validate, and repair the device once.

        Parameters
        ----------
        request : RunRequest
            Validated request body.
        fallback_note : str | None
            Sentence prepended when this path replaces a failed provider call.

        Returns
        -------
        tuple[TurnOutcome, list[AgentEvent]]
            The mock answer and the events that describe how it was reached.
        """
        events: list[AgentEvent] = []
        invocations: list[ToolInvocation] = []
        arguments: dict[str, Any] | None = request.tool_arguments
        if arguments is None:
            arguments = extract_json_object(request.message)

        if arguments is None:
            events.append(AgentEvent(event="step", data={"step": 1, "action": "answer", "mode": "mock"}))
            text: str = summarize_invocations([], fallback_note=fallback_note)
            return TurnOutcome(text, 1, [], "mock"), events

        step: int = 1
        events.append(
            AgentEvent(
                event="step",
                data={"step": step, "action": "validate_and_call", "tool_name": TOOL_NAME},
            )
        )
        first: ToolInvocation = dispatch_tool(TOOL_NAME, arguments)
        invocations.append(first)
        events.append(AgentEvent(event="tool", data=first.model_dump(mode="json")))

        if not first.ok and first.error is not None:
            repaired: dict[str, Any] | None = repair_benchmark_arguments(arguments, first.error)
            if repaired is not None:
                logger.info(
                    "Rebuilding %s arguments after validation error: device %r -> %r",
                    TOOL_NAME,
                    arguments.get("device"),
                    repaired.get("device"),
                )
                step = 2
                events.append(
                    AgentEvent(
                        event="step",
                        data={"step": step, "action": "repair", "tool_name": TOOL_NAME},
                    )
                )
                second: ToolInvocation = dispatch_tool(TOOL_NAME, repaired)
                invocations.append(second)
                events.append(AgentEvent(event="tool", data=second.model_dump(mode="json")))

        step += 1
        events.append(AgentEvent(event="step", data={"step": step, "action": "answer", "mode": "mock"}))
        text = summarize_invocations(invocations, fallback_note=fallback_note)
        return TurnOutcome(text, step, invocations, "mock"), events

    async def _iter_llm(
        self,
        request: RunRequest,
        history: list[ChatMessage],
        box: OutcomeBox,
    ) -> AsyncIterator[AgentEvent]:
        """Call an OpenAI-compatible chat API until it stops requesting tools.

        Parameters
        ----------
        request : RunRequest
            Validated request body. Explicit tool arguments are hinted to the model.
        history : list[ChatMessage]
            Trimmed session window, including the new user turn.
        box : OutcomeBox
            Filled with the turn outcome before this generator finishes.

        Yields
        ------
        AgentEvent
            A step event per model round-trip, then one tool event per call.

        Raises
        ------
        LLMUnavailable
            When the provider times out, returns a non-2xx status, or sends a body
            this client cannot read.
        """
        messages: list[dict[str, Any]] = self._llm_messages(history, request)
        invocations: list[ToolInvocation] = []
        tools: list[dict[str, Any]] = openai_tools()

        for step in range(1, self._settings.max_steps + 1):
            yield AgentEvent(
                event="step",
                data={"step": step, "action": "chat_completion", "mode": "llm"},
            )
            payload: dict[str, Any] = await self._chat(messages)
            choice: dict[str, Any] = self._first_choice(payload)
            tool_calls: list[dict[str, Any]] = list(choice.get("tool_calls") or [])
            if not tool_calls:
                content: str = str(choice.get("content") or "").strip()
                if not content:
                    content = summarize_invocations(invocations, fallback_note=None)
                box.outcome = TurnOutcome(content, step, invocations, "llm")
                return

            messages.append(choice)
            for call in tool_calls:
                invocation = self._invoke_model_call(call)
                invocations.append(invocation)
                yield AgentEvent(event="tool", data=invocation.model_dump(mode="json"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or ""),
                        "content": json.dumps(
                            invocation.output if invocation.ok else self._error_payload(invocation),
                            sort_keys=True,
                        ),
                    }
                )

        box.outcome = TurnOutcome(
            summarize_invocations(
                invocations,
                fallback_note="Step limit reached before a final model answer.",
            ),
            self._settings.max_steps,
            invocations,
            "llm",
        )

    def _llm_messages(
        self,
        history: list[ChatMessage],
        request: RunRequest,
    ) -> list[dict[str, Any]]:
        """Build the chat message list, adding explicit tool arguments as a hint.

        Parameters
        ----------
        history : list[ChatMessage]
            Trimmed session window.
        request : RunRequest
            Current turn. ``tool_arguments`` is appended to the last user message.

        Returns
        -------
        list[dict[str, Any]]
            Messages ready for a chat completion request.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        last_index: int = len(history) - 1
        for index, item in enumerate(history):
            content: str = item.content
            if (
                index == last_index
                and item.role == "user"
                and request.tool_arguments is not None
            ):
                suggestion: str = json.dumps(request.tool_arguments, sort_keys=True)
                content = f"{content}\n\nSuggested tool arguments:\n{suggestion}"
            messages.append({"role": item.role, "content": content})
        return messages

    async def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """POST one chat completion.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Provider message list, including tool results from earlier steps.

        Returns
        -------
        dict[str, Any]
            Decoded JSON body.

        Raises
        ------
        LLMUnavailable
            On transport errors, non-2xx responses, or a body that is not a JSON object.
        """
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self._settings.llm_timeout_s))
        url: str = self._settings.llm_base_url.rstrip("/") + "/chat/completions"
        request_body: dict[str, Any] = {
            "model": self._settings.llm_model,
            "temperature": 0,
            "messages": messages,
            "tools": openai_tools(),
            "tool_choice": "auto",
        }
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            response: httpx.Response = await self._http.post(url, json=request_body, headers=headers)
            response.raise_for_status()
            decoded: Any = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            raise LLMUnavailable(exc.__class__.__name__) from exc
        if not isinstance(decoded, dict):
            raise LLMUnavailable("Response body was not a JSON object.")
        return decoded

    def _first_choice(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Read the first assistant message from a chat completion body.

        Parameters
        ----------
        payload : dict[str, Any]
            Decoded provider response.

        Returns
        -------
        dict[str, Any]
            The assistant message object.

        Raises
        ------
        LLMUnavailable
            When ``choices[0].message`` is missing or not an object.
        """
        try:
            choice: Any = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable("Chat response did not include choices[0].message.") from exc
        if not isinstance(choice, dict):
            raise LLMUnavailable("Chat message was not a JSON object.")
        return choice

    def _invoke_model_call(self, call: dict[str, Any]) -> ToolInvocation:
        """Dispatch one tool call emitted by the model.

        Parameters
        ----------
        call : dict[str, Any]
            One item of ``message.tool_calls``.

        Returns
        -------
        ToolInvocation
            Dispatcher result. Invalid JSON arguments become a tool error.
        """
        function: Any = call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        name: str = str(function.get("name") or "")
        raw_arguments: Any = function.get("arguments") or "{}"
        parsed: Any
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed = raw_arguments
        else:
            parsed = raw_arguments
        return dispatch_tool(name, parsed)

    def _error_payload(self, invocation: ToolInvocation) -> dict[str, Any]:
        """Serialize a failed invocation for the next model turn.

        Parameters
        ----------
        invocation : ToolInvocation
            Failed attempt.

        Returns
        -------
        dict[str, Any]
            JSON object the model can use to rebuild the call.
        """
        if invocation.error is None:
            return {"ok": False, "message": "Tool failed without an error payload."}
        return invocation.error.model_dump(mode="json")

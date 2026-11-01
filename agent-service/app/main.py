"""ASGI entrypoint for the edge agent service.

Run with ``python -m app.main`` or ``uvicorn app.main:app``.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse

from app import __version__
from app.agent import AgentRunner
from app.config import Settings, get_settings
from app.schemas import AgentEvent, AgentResponse, HealthResponse, PRReviewInput, PRReviewResponse, RunRequest
from app.tools import list_tool_names
from app.tools.pr_review_tool import audit_code_changes


def configure_logging(level: str) -> None:
    """Configure the root logger once.

    Parameters
    ----------
    level : str
        Logging level name: debug, info, warning, or error.
    """
    root: logging.Logger = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_runner(settings: Settings) -> AgentRunner:
    """Construct the process-wide orchestrator.

    Parameters
    ----------
    settings : Settings
        Loaded process settings.

    Returns
    -------
    AgentRunner
        Runner that owns session memory for this process.
    """
    return AgentRunner(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the LLM client when the process stops.

    Parameters
    ----------
    app : FastAPI
        Application whose ``state.agent`` holds the runner.

    Yields
    ------
    None
        Control stays here while the server is running.
    """
    yield
    runner: AgentRunner = app.state.agent
    await runner.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    settings : Settings | None
        Settings override. Tests pass an isolated instance. Production uses
        :func:`app.config.get_settings`.

    Returns
    -------
    FastAPI
        Application with health, run, and stream routes.
    """
    resolved: Settings = settings if settings is not None else get_settings()
    configure_logging(resolved.log_level)
    runner: AgentRunner = build_runner(resolved)
    application: FastAPI = FastAPI(
        title="Edge Agent Service",
        version=__version__,
        description=(
            "Typed tool-calling loop for an edge inference config check, "
            "plus a deterministic PR diff audit. "
            "Invalid tool arguments come back inside the agent response. "
            "An empty LLM_API_KEY keeps both demos on fixed rules."
        ),
        lifespan=lifespan,
    )
    application.state.settings = resolved
    application.state.agent = runner

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect the service root to the interactive docs.

        Returns
        -------
        RedirectResponse
            Redirect to ``/docs``.
        """
        return RedirectResponse(url="/docs")

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Report process status and whether the tool worker is loaded.

        Returns
        -------
        HealthResponse
            Liveness payload. ``workers_ready`` is true when at least one tool
            is registered in this process.
        """
        names: list[str] = list_tool_names()
        llm_mode: Literal["mock", "llm"] = "llm" if resolved.llm_configured else "mock"
        return HealthResponse(
            status="ok",
            environment=resolved.environment,
            llm_mode=llm_mode,
            workers_ready=bool(names),
            registered_tools=names,
        )

    @application.post("/api/v1/run", response_model=AgentResponse, tags=["agent"])
    async def run_agent(body: RunRequest) -> AgentResponse:
        """Run one synchronous agent turn.

        A body that does not match :class:`app.schemas.RunRequest` is rejected
        with HTTP 422. A body whose tool arguments fail :class:`app.schemas.BenchmarkInput`
        still returns HTTP 200, with the validation error inside ``tool_invocations``.

        Parameters
        ----------
        body : RunRequest
            Validated JSON body.

        Returns
        -------
        AgentResponse
            Final text, steps, tool trace, and latency.
        """
        active: AgentRunner = application.state.agent
        return await active.run(body)

    @application.post("/api/v1/stream", tags=["agent"])
    async def stream_agent(body: RunRequest) -> StreamingResponse:
        """Stream orchestrator events as Server-Sent Events.

        Event order is ``step``, ``tool``, then ``result``. The ``result``
        payload is the same object as ``POST /api/v1/run``.

        Parameters
        ----------
        body : RunRequest
            Validated JSON body.

        Returns
        -------
        StreamingResponse
            ``text/event-stream`` body.
        """

        async def generate() -> AsyncIterator[str]:
            """Encode each orchestrator event as one SSE frame.

            Yields
            ------
            str
                One SSE frame, including the trailing blank line.
            """
            active: AgentRunner = application.state.agent
            async for event in active.iter_events(body):
                frame: AgentEvent = event
                yield f"event: {frame.event}\ndata: {frame.model_dump_json()}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/api/v1/pr-review", response_model=PRReviewResponse, tags=["review"])
    async def review_pr(payload: PRReviewInput) -> PRReviewResponse:
        """Audit a diff, draft pytest names, and score the merge risk.

        The route always runs the deterministic tools. A body that fails
        :class:`app.schemas.PRReviewInput` is HTTP 422.

        Parameters
        ----------
        payload : PRReviewInput
            Pull request id, file path, and diff text.

        Returns
        -------
        PRReviewResponse
            Risk, findings, test names, and the three tool traces.
        """
        return audit_code_changes(payload)

    return application


app: FastAPI = create_app()


def main() -> None:
    """Start uvicorn on the configured host and port.

    The container command is ``python -m app.main``. ``PORT`` comes from settings.
    """
    settings: Settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()

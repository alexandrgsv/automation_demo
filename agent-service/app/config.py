"""Process settings loaded from the environment and an optional ``.env`` file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the edge agent service.

    Environment variables use the field name in upper case, for example
    ``LLM_API_KEY``. An empty key selects mock mode so the demo does not
    call an external provider.

    Attributes
    ----------
    environment : str
        Deployment label returned by ``/health``.
    port : int
        TCP port used when the process is started with ``python -m app.main``.
    log_level : {"debug", "info", "warning", "error"}
        Level for the standard-library logger and uvicorn.
    llm_base_url : str
        OpenAI-compatible origin, without a trailing path.
    llm_api_key : str
        Bearer token. Blank means the orchestrator stays in mock mode.
    llm_model : str
        Chat model name sent to the provider.
    llm_timeout_s : float
        Per-request timeout for the chat completion call.
    max_steps : int
        Maximum chat completions per turn when a key is configured.
    memory_window : int
        Maximum chat messages kept per session.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(default="local", description="Deployment label.")
    port: int = Field(default=8000, ge=1, le=65535, description="HTTP port.")
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Log level name.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API origin.",
    )
    llm_api_key: str = Field(default="", description="Bearer token. Empty selects mock mode.")
    llm_model: str = Field(default="gpt-4o-mini", description="Chat model name.")
    llm_timeout_s: float = Field(default=20.0, gt=0, description="Chat request timeout in seconds.")
    max_steps: int = Field(default=4, ge=1, le=8, description="Maximum model round-trips per turn.")
    memory_window: int = Field(
        default=8,
        ge=1,
        le=100,
        description="How many chat messages to keep for one session.",
    )

    @property
    def llm_configured(self) -> bool:
        """Return True when a non-blank API key is configured.

        Returns
        -------
        bool
            True when live chat completions should be attempted.
        """
        return bool(self.llm_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, cached for the life of the process.

    Returns
    -------
    Settings
        Settings loaded from the environment and ``.env``.
    """
    return Settings()

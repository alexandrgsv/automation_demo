"""Argument and result contract for the edge FPS skill."""

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkInput(BaseModel):
    """Arguments for the deterministic edge-config benchmark.

    Attributes
    ----------
    batch_size : int
        Inference batch size. Values outside 1..64 are rejected.
    device : str
        Execution target. Only ``cpu``, ``cuda``, and ``tensorrt`` are valid.
    target_fps : float
        FPS the config is required to reach.
    """

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(
        gt=0,
        le=64,
        description="Inference batch size. Must be from 1 to 64.",
        examples=[2],
    )
    device: str = Field(
        pattern=r"^(cpu|cuda|tensorrt)$",
        description="Execution target: cpu, cuda, or tensorrt.",
        examples=["cuda"],
    )
    target_fps: float = Field(
        gt=0,
        description="Required FPS. The tool compares its estimate with this value.",
        examples=[30],
    )


class EdgeBenchmarkResult(BaseModel):
    """Deterministic estimate produced by ``evaluate_edge_config``.

    Attributes
    ----------
    device : str
        Device that was evaluated.
    batch_size : int
        Batch size that was evaluated.
    estimated_latency_ms : float
        Stacked latency for the whole batch, in milliseconds.
    estimated_fps : float
        Throughput implied by that latency.
    target_fps : float
        Threshold copied from the request.
    meets_requirements : bool
        True when ``estimated_fps`` is at least ``target_fps``.
    bottleneck_detected : bool
        True when the estimate misses the target.
    """

    device: str = Field(description="Device that was evaluated.")
    batch_size: int = Field(description="Batch size that was evaluated.")
    estimated_latency_ms: float = Field(description="Estimated batch latency in milliseconds.")
    estimated_fps: float = Field(description="Estimated throughput in frames per second.")
    target_fps: float = Field(description="FPS threshold from the request.")
    meets_requirements: bool = Field(description="True when the estimate reaches the target.")
    bottleneck_detected: bool = Field(description="True when the estimate misses the target.")

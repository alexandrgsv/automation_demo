"""Deterministic edge-config benchmark.

The numbers are a fixed formula so a demo payload always prints the same FPS.
Replace :func:`evaluate_edge_config` with a real profiler without changing
:class:`app.skills.edge_fps_evaluator.schema.BenchmarkInput`.
"""

from app.skills.edge_fps_evaluator.schema import BenchmarkInput, EdgeBenchmarkResult

# Per-sample latency in milliseconds at batch size 1.
# Batching does not overlap samples; the batch multiplies this constant.
BASE_LATENCY_MS: dict[str, float] = {
    "cpu": 45.0,
    "cuda": 12.5,
    "tensorrt": 6.0,
}


def evaluate_edge_config(input_data: BenchmarkInput) -> EdgeBenchmarkResult:
    """Estimate FPS for one edge config and compare it with the target.

    Latency stacks with the batch size. The same input always returns the
    same estimate, and this function does not touch a device.

    ``estimated_latency_ms = BASE_LATENCY_MS[device] * batch_size``

    ``estimated_fps = 1000 / estimated_latency_ms``

    Parameters
    ----------
    input_data : BenchmarkInput
        Validated batch size, device, and FPS target.

    Returns
    -------
    EdgeBenchmarkResult
        Estimated latency, FPS, and whether the target is met.

    Raises
    ------
    ValueError
        If ``device`` is not in :data:`BASE_LATENCY_MS`. The schema normally
        rejects that value before this function runs.
    """
    try:
        base_latency_ms: float = BASE_LATENCY_MS[input_data.device]
    except KeyError as exc:
        raise ValueError(f"Unsupported device: {input_data.device}") from exc

    estimated_latency_ms: float = base_latency_ms * input_data.batch_size
    estimated_fps: float = 1000.0 / estimated_latency_ms
    meets_requirements: bool = estimated_fps >= input_data.target_fps
    return EdgeBenchmarkResult(
        device=input_data.device,
        batch_size=input_data.batch_size,
        estimated_latency_ms=round(estimated_latency_ms, 2),
        estimated_fps=round(estimated_fps, 2),
        target_fps=input_data.target_fps,
        meets_requirements=meets_requirements,
        bottleneck_detected=not meets_requirements,
    )

"""Edge FPS skill: schema, device-repair protocol, and the FPS function."""

from typing import Any

from pydantic import BaseModel

from app.schemas import ToolError
from app.skills.base import BaseSkill, ToolBinding
from app.skills.edge_fps_evaluator.logic import evaluate_edge_config
from app.skills.edge_fps_evaluator.schema import BenchmarkInput, EdgeBenchmarkResult

# Common names a caller might use before the schema's exact tokens.
_DEVICE_ALIASES: dict[str, str] = {
    "gpu": "cuda",
    "nvidia": "cuda",
    "cuda:0": "cuda",
    "trt": "tensorrt",
    "tensor-rt": "tensorrt",
}
_ALLOWED_DEVICES: frozenset[str] = frozenset({"cpu", "cuda", "tensorrt"})

_TOOL_DESCRIPTION: str = (
    "Estimate FPS and latency for an edge inference config. "
    "device must be cpu, cuda, or tensorrt. "
    "Do not invent FPS numbers; return this tool's output."
)


class EdgeFPSEvaluatorSkill(BaseSkill):
    """Validate an edge config and estimate FPS with a fixed formula.

    The registered tool name stays ``evaluate_edge_config``. A failed device
    string can be repaired once through :meth:`repair_arguments`.
    """

    name: str = "edge_fps_evaluator"
    description: str = (
        "Evaluates an edge inference config and checks it against a target FPS. "
        "Repairs a device alias such as GPU onto cuda, once."
    )

    @property
    def tools(self) -> tuple[ToolBinding, ...]:
        """Return the single FPS tool.

        Returns
        -------
        tuple[ToolBinding, ...]
            The ``evaluate_edge_config`` binding.
        """
        return (
            ToolBinding(
                name="evaluate_edge_config",
                description=_TOOL_DESCRIPTION,
                input_model=BenchmarkInput,
                handler=evaluate_edge_config,
            ),
        )

    def execute(self, params: BaseModel) -> EdgeBenchmarkResult:
        """Estimate FPS for one validated config.

        Parameters
        ----------
        params : BaseModel
            A :class:`BenchmarkInput`, or a model with the same fields.

        Returns
        -------
        EdgeBenchmarkResult
            Latency, FPS, and whether the target is met.
        """
        parsed: BenchmarkInput = BenchmarkInput.model_validate(params, from_attributes=True)
        return evaluate_edge_config(parsed)

    def repair_arguments(
        self,
        arguments: dict[str, Any],
        error: ToolError,
    ) -> dict[str, Any] | None:
        """Rewrite a failed device string onto ``cpu``, ``cuda``, or ``tensorrt``.

        Only a validation error is eligible. Batch size and target FPS stay as
        submitted so a bad number is reported instead of being clamped.

        Parameters
        ----------
        arguments : dict[str, Any]
            Arguments that failed validation.
        error : ToolError
            Structured failure from the dispatcher.

        Returns
        -------
        dict[str, Any] | None
            A new argument dict, or None when the device cannot be mapped.
        """
        if error.type != "validation_error":
            return None
        device: Any = arguments.get("device")
        if not isinstance(device, str):
            return None
        lowered: str = device.strip().lower()
        canonical: str | None = _DEVICE_ALIASES.get(lowered)
        if canonical is None and lowered in _ALLOWED_DEVICES:
            canonical = lowered
        if canonical is None or canonical == device:
            return None
        repaired: dict[str, Any] = dict(arguments)
        repaired["device"] = canonical
        return repaired


EDGE_FPS_SKILL: EdgeFPSEvaluatorSkill = EdgeFPSEvaluatorSkill()

"""Compatibility import for the edge FPS function.

The skill lives in ``app.skills.edge_fps_evaluator``.
"""

from app.skills.edge_fps_evaluator.logic import BASE_LATENCY_MS, evaluate_edge_config

__all__ = ["BASE_LATENCY_MS", "evaluate_edge_config"]

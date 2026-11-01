"""Skill packages for the edge agent.

Each subdirectory is one skill: a Pydantic contract in ``schema.py``, the
protocol in ``prompt.md``, and the deterministic function in ``logic.py``.
The package init does not import those modules. Schema modules must stay
importable while ``app.schemas`` is still loading.
"""

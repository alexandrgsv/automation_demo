"""Skill contract: metadata, a typed input, and deterministic execution.

A skill is the unit the orchestrator knows. It is not only a function. It
carries the argument schema, the protocol text, and the repair rule.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import getfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.schemas import ToolError


class ToolBinding:
    """One registered tool that belongs to a skill.

    Parameters
    ----------
    name : str
        Registry name the dispatcher accepts.
    description : str
        Sentence shown to a model in the tool list.
    input_model : type[BaseModel]
        Pydantic contract for the arguments.
    handler : Callable[..., BaseModel]
        Deterministic function. It receives a validated model.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: Callable[..., BaseModel],
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_model: type[BaseModel] = input_model
        self.handler: Callable[..., BaseModel] = handler


class BaseSkill(ABC):
    """Contract, protocol, and deterministic function for one agent skill.

    ``name`` and ``description`` say when the orchestrator should use it.
    ``tools`` lists the callable contracts. ``instructions`` loads ``prompt.md``
    from the skill directory. ``repair_arguments`` is the optional one-step
    parameter repair. ``execute`` runs the skill.
    """

    name: str
    description: str

    @property
    @abstractmethod
    def tools(self) -> tuple[ToolBinding, ...]:
        """Return the tools this skill registers.

        Returns
        -------
        tuple[ToolBinding, ...]
            Tool contracts, in call order.
        """

    @abstractmethod
    def execute(self, params: BaseModel) -> BaseModel:
        """Run the skill on a validated input.

        Parameters
        ----------
        params : BaseModel
            Arguments that already match this skill's input schema.

        Returns
        -------
        BaseModel
            Deterministic result.
        """

    def repair_arguments(
        self,
        arguments: dict[str, Any],
        error: ToolError,
    ) -> dict[str, Any] | None:
        """Rewrite failed arguments once, or decline.

        The default skill does not invent a repair. A skill that can map an
        alias onto a legal value overrides this method.

        Parameters
        ----------
        arguments : dict[str, Any]
            Arguments that failed validation.
        error : ToolError
            Structured failure from the dispatcher.

        Returns
        -------
        dict[str, Any] | None
            Replacement arguments, or None when this skill cannot repair the call.
        """
        return None

    def instructions(self) -> str:
        """Load the skill protocol from ``prompt.md``.

        Returns
        -------
        str
            Protocol text. The description is used when the file is absent.
        """
        path: Path = Path(getfile(type(self))).resolve().parent / "prompt.md"
        if not path.is_file():
            return self.description
        return path.read_text(encoding="utf-8").strip()

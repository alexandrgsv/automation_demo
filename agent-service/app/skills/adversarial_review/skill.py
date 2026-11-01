"""Developer-versus-QA skill: a sandbox tool and a fixed two-pass loop."""

from pydantic import BaseModel

from app.skills.adversarial_review.logic import (
    EXECUTE_TOOL,
    execute_in_sandbox,
    run_adversarial_dev_loop,
)
from app.skills.adversarial_review.schema import (
    AdversarialReviewRequest,
    AdversarialReviewResponse,
    TestExecutionInput,
)
from app.skills.base import BaseSkill, ToolBinding


class AdversarialReviewSkill(BaseSkill):
    """Fail a naive throughput function, then pass the guarded one.

    ``execute_in_sandbox`` is the registered tool. :meth:`execute` runs the
    scripted loop and does not accept caller source.
    """

    name: str = "adversarial_review"
    description: str = (
        "Runs a fixed developer snippet against a fixed QA script, then repeats "
        "the script on a hardened snippet."
    )

    @property
    def tools(self) -> tuple[ToolBinding, ...]:
        """Return the sandbox runner.

        Returns
        -------
        tuple[ToolBinding, ...]
            The ``execute_in_sandbox`` binding.
        """
        return (
            ToolBinding(
                name=EXECUTE_TOOL,
                description=(
                    "Run one QA script against one implementation in a fresh namespace. "
                    "Imports and file access are not available."
                ),
                input_model=TestExecutionInput,
                handler=execute_in_sandbox,
            ),
        )

    def execute(self, params: BaseModel) -> AdversarialReviewResponse:
        """Run the naive pass and the hardened pass.

        Parameters
        ----------
        params : BaseModel
            An :class:`AdversarialReviewRequest`. Only ``task`` is read.

        Returns
        -------
        AdversarialReviewResponse
            The first failure, the second pass, and the named edge cases.
        """
        parsed: AdversarialReviewRequest = AdversarialReviewRequest.model_validate(
            params,
            from_attributes=True,
        )
        return run_adversarial_dev_loop(parsed.task)


ADVERSARIAL_REVIEW_SKILL: AdversarialReviewSkill = AdversarialReviewSkill()

"""Sample algorithm skill: retrieve a config, then create its Portainer stack."""

from pydantic import BaseModel

from app.config import get_settings
from app.skills.algorithm_deployer.logic import (
    deploy_document,
    deploy_from_request,
    document_by_id,
    retrieve_algorithm,
)
from app.skills.algorithm_deployer.schema import (
    AlgorithmDeployRequest,
    AlgorithmDeployResponse,
    AlgorithmDocument,
    DeployAlgorithmInput,
    PortainerDeployResult,
    RetrieveAlgorithmInput,
    RetrieveAlgorithmResult,
)
from app.skills.base import BaseSkill, ToolBinding


def retrieve_algorithm_config(payload: RetrieveAlgorithmInput) -> RetrieveAlgorithmResult:
    """Score the sample corpus for one sentence.

    Parameters
    ----------
    payload : RetrieveAlgorithmInput
        Deployment sentence.

    Returns
    -------
    RetrieveAlgorithmResult
        Document scores and the winner, if any phrase matched.
    """
    return retrieve_algorithm(payload.query)


def deploy_algorithm_stack(payload: DeployAlgorithmInput) -> PortainerDeployResult:
    """Deploy the config stored for one algorithm id.

    The config is read from the corpus. The tool arguments do not carry it.

    Parameters
    ----------
    payload : DeployAlgorithmInput
        Algorithm id and optional Portainer endpoint.

    Returns
    -------
    PortainerDeployResult
        Mock skip, or the live Portainer outcome.
    """
    document: AlgorithmDocument
    document, _name = document_by_id(payload.algorithm_id)
    _compose, result = deploy_document(
        document,
        get_settings(),
        endpoint_id=payload.endpoint_id,
    )
    return result


class AlgorithmDeployerSkill(BaseSkill):
    """Pick one of three sample analytics configs and hand it to Portainer.

    Retrieval reads ``knowledge/*.json``. A request that matches nothing does
    not create a stack.
    """

    name: str = "algorithm_deployer"
    description: str = (
        "Selects a sample line-crossing, zone-presence, or heatmap config from "
        "a fixed corpus and creates a Portainer stack for it."
    )

    @property
    def tools(self) -> tuple[ToolBinding, ...]:
        """Return the retrieve tool and the deploy tool.

        Returns
        -------
        tuple[ToolBinding, ...]
            Retrieval first, then the stack call.
        """
        return (
            ToolBinding(
                name="retrieve_algorithm_config",
                description=(
                    "Match a deployment sentence to one sample algorithm document. "
                    "Returns no selection when no phrase matches."
                ),
                input_model=RetrieveAlgorithmInput,
                handler=retrieve_algorithm_config,
            ),
            ToolBinding(
                name="deploy_algorithm_stack",
                description=(
                    "Create a Portainer stack for a sample algorithm id. "
                    "The config comes from the corpus, not from the arguments."
                ),
                input_model=DeployAlgorithmInput,
                handler=deploy_algorithm_stack,
            ),
        )

    def execute(self, params: BaseModel) -> AlgorithmDeployResponse:
        """Retrieve a config and deploy it with the process Portainer settings.

        Parameters
        ----------
        params : BaseModel
            An :class:`AlgorithmDeployRequest`.

        Returns
        -------
        AlgorithmDeployResponse
            Scores, config, and the Portainer outcome.
        """
        parsed: AlgorithmDeployRequest = AlgorithmDeployRequest.model_validate(
            params,
            from_attributes=True,
        )
        return deploy_from_request(parsed, get_settings())


ALGORITHM_DEPLOYER_SKILL: AlgorithmDeployerSkill = AlgorithmDeployerSkill()
